"""
JointCamemBERTav2 — 3-head NLU model for Anima.

Heads:
  1. Speech Act classification (13 classes) on [CLS]
  2. Domain classification (15 classes) on [CLS]
  3. Slot filling BIO (37 labels) on all tokens + optional CRF

Loss: λ1 * Focal(speech_act) + λ2 * Focal(domain) + λ3 * CRF_NLL(slots)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with optional label smoothing.

    FL(p) = -(1-p)^γ * log(p)
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = -100,
                 smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[mask]
        valid_targets = targets[mask]

        ce_loss = F.cross_entropy(
            valid_logits, valid_targets,
            reduction="none",
            label_smoothing=self.smoothing,
        )
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        return (focal_weight * ce_loss).mean()


def _make_head(in_features: int, num_classes: int, hidden_dim: int,
               dropout: float = 0.1) -> nn.Module:
    """Build a 2-layer classification head."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def pack_for_crf(slot_logits: torch.Tensor,
                 slot_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack non-(-100) positions into contiguous sequences for CRF.

    pytorch-crf requires the mask to be contiguous (all True then all
    False). Since real label positions are scattered among -100 positions,
    we pack them left-aligned and build a contiguous mask.

    Fully vectorized — no Python loops over the batch dimension.

    Returns: packed_logits, packed_labels, packed_mask (all same shape)
    """
    real_mask = (slot_labels != -100)  # (batch, seq_len)
    counts = real_mask.sum(dim=1)      # (batch,)
    max_real = counts.max().item()

    if max_real == 0:
        batch = slot_logits.size(0)
        device = slot_logits.device
        n_tags = slot_logits.size(-1)
        return (torch.zeros(batch, 1, n_tags, device=device, dtype=slot_logits.dtype),
                torch.zeros(batch, 1, dtype=torch.long, device=device),
                torch.zeros(batch, 1, dtype=torch.bool, device=device))

    batch = slot_logits.size(0)
    n_tags = slot_logits.size(-1)
    device = slot_logits.device

    packed_logits = torch.zeros(batch, max_real, n_tags, device=device,
                                dtype=slot_logits.dtype)
    packed_labels = torch.zeros(batch, max_real, dtype=torch.long, device=device)

    packed_pos = (real_mask.long().cumsum(dim=1) - 1).clamp(min=0)
    row_idx = torch.arange(batch, device=device).unsqueeze(1).expand_as(real_mask)

    src_rows = row_idx[real_mask]
    src_cols = packed_pos[real_mask]
    packed_logits[src_rows, src_cols] = slot_logits[real_mask]
    packed_labels[src_rows, src_cols] = slot_labels[real_mask]

    pos_range = torch.arange(max_real, device=device).unsqueeze(0)
    packed_mask = pos_range < counts.unsqueeze(1)

    return packed_logits, packed_labels, packed_mask


class JointCamemBERTav2(nn.Module):
    """CamemBERTav2-base with 3 classification heads + optional CRF for slots."""

    def __init__(
        self,
        model_name: str = "almanach/camembertav2-base",
        num_speech_acts: int = 13,
        num_domains: int = 15,
        num_slot_labels: int = 37,
        dropout: float = 0.1,
        use_crf: bool = False,
        head_hidden_dim: int = 256,
        focal_gamma: float = 2.0,
        smoothing: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.speech_act_head = _make_head(hidden_size, num_speech_acts, head_hidden_dim, dropout)
        self.domain_head = _make_head(hidden_size, num_domains, head_hidden_dim, dropout)
        self.slot_head = _make_head(hidden_size, num_slot_labels, head_hidden_dim, dropout)

        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_slot_labels, batch_first=True)

        self.focal_cls = FocalLoss(gamma=focal_gamma, smoothing=smoothing)
        self.focal_slot = FocalLoss(gamma=focal_gamma)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speech_act_labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
        loss_weights: dict | None = None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        cls_output = self.dropout(hidden[:, 0, :])  # (batch, hidden)
        seq_output = self.dropout(hidden)  # (batch, seq_len, hidden)

        speech_act_logits = self.speech_act_head(cls_output)  # (batch, 13)
        domain_logits = self.domain_head(cls_output)  # (batch, 15)
        slot_logits = self.slot_head(seq_output)  # (batch, seq_len, 37)

        loss = None
        if speech_act_labels is not None and domain_labels is not None and slot_labels is not None:
            w = loss_weights or {"speech_act": 1.0, "domain": 1.0, "slots": 2.0}

            sa_loss = self.focal_cls(speech_act_logits, speech_act_labels)
            dom_loss = self.focal_cls(domain_logits, domain_labels)

            if self.use_crf:
                # Pack real-label positions into contiguous sequences
                packed_logits, packed_labels, packed_mask = pack_for_crf(
                    slot_logits, slot_labels
                )
                # Drop rows with zero real labels (all-IGNORE examples)
                # so they don't corrupt CRF transition learning
                row_has_labels = packed_mask.any(dim=1)
                if row_has_labels.any():
                    slot_loss = -self.crf(
                        packed_logits[row_has_labels],
                        packed_labels[row_has_labels],
                        mask=packed_mask[row_has_labels],
                        reduction="mean",
                    )
                else:
                    slot_loss = torch.tensor(0.0, device=slot_logits.device,
                                             requires_grad=True)
            else:
                slot_loss = self.focal_slot(
                    slot_logits.view(-1, slot_logits.size(-1)),
                    slot_labels.view(-1),
                )

            loss = w["speech_act"] * sa_loss + w["domain"] * dom_loss + w["slots"] * slot_loss

        return {
            "loss": loss,
            "speech_act_logits": speech_act_logits,
            "domain_logits": domain_logits,
            "slot_logits": slot_logits,
        }
