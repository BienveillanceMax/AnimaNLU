"""
Training script for JointCamemBERTav2 multi-head NLU model.

Loads config.yaml, dataset from data/{train,dev}/, trains with
HuggingFace Trainer + custom loss, early stopping on eval_sentence_acc.
"""

import math
import re
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from collections import Counter
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler

from labels import (
    SPEECH_ACTS, DOMAINS, SLOT_LABELS,
    SPEECH_ACT_L2I, DOMAIN_L2I, SLOT_LABEL_L2I,
)
from model import JointCamemBERTav2, pack_for_crf


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────
# Dataset — pre-tokenized at init, not per-item
# ──────────────────────────────────────────────
def _align_slot_labels(word_ids, tags, slot_label_l2i):
    """Align BIO tags to subword tokens.

    - Special tokens (word_id=None) → -100 (CRF will mask these out)
    - First subword of a word → the word's BIO label
    - Continuation subwords → I-{slot_type} if inside an entity, O if word is O
      (so the CRF sees consistent B→I→I sequences across subwords)
    - IGNORE tags → -100 for the word AND all its subwords
    """
    slot_labels = []
    prev_word_id = None
    current_tag = "O"  # track the tag for continuation subwords
    current_ignored = False  # track if current word is IGNORE
    for word_id in word_ids:
        if word_id is None:
            # [CLS], [SEP], [PAD] → masked out
            slot_labels.append(-100)
        elif word_id != prev_word_id:
            # First subword of a new word
            tag = tags[word_id] if word_id < len(tags) else "O"
            if tag == "IGNORE":
                slot_labels.append(-100)
                current_tag = "O"
                current_ignored = True
            else:
                slot_labels.append(slot_label_l2i.get(tag, 0))
                current_tag = tag
                current_ignored = False
        else:
            # Continuation subword
            if current_ignored:
                slot_labels.append(-100)
            elif current_tag == "O":
                slot_labels.append(slot_label_l2i["O"])
            elif current_tag.startswith("B-") or current_tag.startswith("I-"):
                i_tag = "I-" + current_tag[2:]
                slot_labels.append(slot_label_l2i.get(i_tag, 0))
            else:
                slot_labels.append(slot_label_l2i["O"])
        prev_word_id = word_id
    return slot_labels


class NluDataset(Dataset):
    """Pre-tokenized dataset — tokenization happens once at init, not per __getitem__."""

    def __init__(self, data_dir: Path, tokenizer, max_length: int = 64):
        texts = (data_dir / "seq.in").read_text(encoding="utf-8").strip().split("\n")
        bio_tags_raw = (data_dir / "seq.out").read_text(encoding="utf-8").strip().split("\n")
        speech_acts = (data_dir / "speech_act").read_text(encoding="utf-8").strip().split("\n")
        domains = (data_dir / "domain").read_text(encoding="utf-8").strip().split("\n")

        assert len(texts) == len(bio_tags_raw) == len(speech_acts) == len(domains)

        # Pre-tokenize everything in one batch
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.input_ids = encodings["input_ids"]          # (N, max_length)
        self.attention_mask = encodings["attention_mask"]  # (N, max_length)

        # Pre-compute labels
        self.speech_act_labels = torch.tensor(
            [SPEECH_ACT_L2I[sa] for sa in speech_acts], dtype=torch.long
        )
        self.domain_labels = torch.tensor(
            [DOMAIN_L2I[dom] for dom in domains], dtype=torch.long
        )

        # Pre-align slot labels
        self.slot_labels = torch.full_like(self.input_ids, -100, dtype=torch.long)
        for i in range(len(texts)):
            tags = bio_tags_raw[i].split()
            word_ids = encodings.word_ids(batch_index=i)
            aligned = _align_slot_labels(word_ids, tags, SLOT_LABEL_L2I)
            self.slot_labels[i] = torch.tensor(aligned, dtype=torch.long)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "speech_act_labels": self.speech_act_labels[idx].item(),
            "domain_labels": self.domain_labels[idx].item(),
            "slot_labels": self.slot_labels[idx],
        }


# ──────────────────────────────────────────────
# DDP-compatible weighted sampler
# ──────────────────────────────────────────────
class DistributedWeightedSampler(Sampler):
    """WeightedRandomSampler that shards across DDP ranks.

    Each rank draws from a disjoint subset of the weighted sample indices,
    so no two GPUs see the same example in the same step. Re-shuffles
    per epoch via set_epoch().
    """

    def __init__(self, weights, num_samples, num_replicas=None, rank=None):
        import torch.distributed as dist
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.total_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples_per_rank = math.ceil(self.total_samples / self.num_replicas)
        self.total_padded = self.num_samples_per_rank * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + 42)
        indices = torch.multinomial(
            self.weights, self.total_padded, replacement=True, generator=g
        ).tolist()
        start = self.rank * self.num_samples_per_rank
        return iter(indices[start:start + self.num_samples_per_rank])

    def __len__(self):
        return self.num_samples_per_rank


# ──────────────────────────────────────────────
# Layer-wise LR decay
# ──────────────────────────────────────────────
def _build_param_groups(model, base_lr, layer_lr_decay, weight_decay, crf_lr_multiplier=50.0):
    """
    Build optimizer param groups with layer-wise LR decay.

    Layer 0 (bottom) gets lr * decay^12, layer 11 (top) gets lr * decay^1.
    Embeddings get lr * decay^12. Heads get lr * 1.0.
    CRF params get their own group at base_lr * crf_lr_multiplier.
    No weight decay on bias and LayerNorm.

    Why CRF LR multiplier: transition/start/end params start at 0 (unlike
    pretrained encoder). At base_lr ~3e-5 and 'mean' reduction, the 1369
    transition params develop std≈0.06 over 40 epochs — not enough to
    separate legal from illegal transitions. 50x multiplier gives them
    the aggressive learning they need, similar to training a classifier
    head from scratch.
    """
    # CRF transition/start/end parameters are sparse structural priors, not
    # representation weights — weight decay pulls them toward 0, which erases
    # learned transition constraints (Run 3: std=0.057 after 40 epochs).
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias",
                "crf.transitions", "crf.start_transitions", "crf.end_transitions"}
    crf_param_names = {"crf.transitions", "crf.start_transitions", "crf.end_transitions"}
    num_layers = model.encoder.config.num_hidden_layers  # 12

    # Group parameters by layer depth (CRF params pulled out into their own group)
    groups = {}  # depth → list of (name, param)
    crf_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name in crf_param_names:
            crf_params.append((name, param))
            continue

        # Determine depth: heads = 0 (full LR), encoder layers = 1..12, embeddings = 13
        # DeBERTa-V2 params are under encoder.encoder.layer.{i}.*
        # (self.encoder = AutoModel, which contains .encoder = DebertaV2Encoder)
        if name.startswith("encoder.embeddings."):
            depth = num_layers
        elif name.startswith("encoder.encoder.layer."):
            match = re.match(r"encoder\.encoder\.layer\.(\d+)\.", name)
            layer_idx = int(match.group(1))
            depth = num_layers - layer_idx
        elif name.startswith("encoder.encoder."):
            # encoder.encoder.rel_embeddings, encoder.encoder.LayerNorm
            depth = num_layers
        else:
            # Heads (speech_act_head, domain_head, slot_head), dropout
            depth = 0

        groups.setdefault(depth, []).append((name, param))

    # Build param groups (encoder + heads)
    param_groups = []
    for depth, params in sorted(groups.items()):
        lr = base_lr * (layer_lr_decay ** depth)
        decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay)]
        no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay)]

        if decay_params:
            param_groups.append({
                "params": [p for _, p in decay_params],
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if no_decay_params:
            param_groups.append({
                "params": [p for _, p in no_decay_params],
                "lr": lr,
                "weight_decay": 0.0,
            })

    # Dedicated CRF group — high LR, no weight decay
    if crf_params:
        param_groups.append({
            "params": [p for _, p in crf_params],
            "lr": base_lr * crf_lr_multiplier,
            "weight_decay": 0.0,
        })

    return param_groups


# ──────────────────────────────────────────────
# Custom Trainer
# ──────────────────────────────────────────────
class JointNluTrainer(Trainer):
    """Trainer that passes loss_weights, focal_gamma, and CRF config to the model."""

    def __init__(self, *args, loss_weights=None, sample_weights=None,
                 layer_lr_decay=1.0, crf_lr_multiplier=50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights or {"speech_act": 1.0, "domain": 1.0, "slots": 2.0}
        self._sample_weights = sample_weights
        self._layer_lr_decay = layer_lr_decay
        self._crf_lr_multiplier = crf_lr_multiplier

    def _get_train_sampler(self, train_dataset=None):
        if self._sample_weights is not None:
            if self.args.parallel_mode.value == "distributed":
                return DistributedWeightedSampler(
                    weights=self._sample_weights,
                    num_samples=len(self._sample_weights),
                )
            return WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=len(self._sample_weights),
                replacement=True,
            )
        return super()._get_train_sampler()

    def create_optimizer(self):
        """Override to apply layer-wise LR decay and a dedicated CRF param group."""
        # Always use the custom optimizer when CRF is present, even if
        # layer_lr_decay=1.0, so the CRF LR multiplier takes effect.
        has_crf = any(n.startswith("crf.") for n, _ in self.model.named_parameters())
        if self._layer_lr_decay < 1.0 or has_crf:
            param_groups = _build_param_groups(
                self.model,
                base_lr=self.args.learning_rate,
                layer_lr_decay=self._layer_lr_decay,
                weight_decay=self.args.weight_decay,
                crf_lr_multiplier=self._crf_lr_multiplier,
            )
            self.optimizer = torch.optim.AdamW(param_groups)
        else:
            super().create_optimizer()
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speech_act_labels=inputs["speech_act_labels"],
            domain_labels=inputs["domain_labels"],
            slot_labels=inputs["slot_labels"],
            loss_weights=self.loss_weights,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to apply CRF Viterbi decode during evaluation.

        Pack real-label positions, CRF-decode the packed sequence, then
        scatter results back to full seq_len so compute_metrics works.
        """
        loss, outputs, labels = super().prediction_step(
            model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys,
        )

        if prediction_loss_only:
            return (loss, None, None)

        # Unpack logits tuple: (sa_logits, dom_logits, slot_logits)
        sa_logits, dom_logits, slot_logits = outputs

        unwrapped = getattr(model, "module", model)
        if hasattr(unwrapped, "use_crf") and unwrapped.use_crf and hasattr(unwrapped, "crf"):
            slot_labels = inputs["slot_labels"]
            packed_logits, _, packed_mask = pack_for_crf(
                slot_logits, slot_labels
            )

            # Filter to rows that have real labels (CRF requires first timestep on)
            row_has_labels = packed_mask.any(dim=1)
            if row_has_labels.any():
                decoded_valid = unwrapped.crf.decode(
                    packed_logits[row_has_labels],
                    mask=packed_mask[row_has_labels],
                )

                real_mask = (slot_labels != -100)
                viterbi_logits = torch.full_like(slot_logits, -1e4)
                valid_indices = row_has_labels.nonzero(as_tuple=True)[0]
                for di, bi in enumerate(valid_indices):
                    idx = real_mask[bi].nonzero(as_tuple=True)[0]
                    n = min(len(idx), len(decoded_valid[di]))
                    if n > 0:
                        tags = torch.tensor(decoded_valid[di][:n], device=slot_logits.device)
                        viterbi_logits[bi, idx[:n], tags] = 1e4

                outputs = (sa_logits, dom_logits, viterbi_logits)

        return (loss, outputs, labels)


# ──────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────
@dataclass
class NluDataCollator:
    """Collates NluDataset items into batches."""

    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "speech_act_labels": torch.tensor([f["speech_act_labels"] for f in features], dtype=torch.long),
            "domain_labels": torch.tensor([f["domain_labels"] for f in features], dtype=torch.long),
            "slot_labels": torch.stack([f["slot_labels"] for f in features]),
        }
        return batch


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute per-head accuracy for logging and early stopping.

    Note: slot_logits here may be CRF-Viterbi-decoded (when use_crf=True),
    with argmax returning the decoded tag directly — see prediction_step.
    """
    predictions, labels = eval_pred
    sa_logits, dom_logits, slot_logits = predictions
    sa_labels, dom_labels, slot_labels = labels

    sa_preds = np.argmax(sa_logits, axis=-1)
    dom_preds = np.argmax(dom_logits, axis=-1)

    sa_acc = (sa_preds == sa_labels).mean()
    dom_acc = (dom_preds == dom_labels).mean()

    # Slot token-level accuracy (ignoring -100). Kept for visibility but
    # NOT used in composite because it saturates near 0.95 due to O-majority
    # (82% of real tokens are O) — doesn't reflect entity F1.
    slot_preds = np.argmax(slot_logits, axis=-1)
    mask = slot_labels != -100
    slot_acc = (slot_preds[mask] == slot_labels[mask]).mean() if mask.any() else 0.0

    # Slot entity-level F1 via seqeval — the metric that actually tracks
    # slot performance. Only real-label positions are included.
    pred_tag_seqs, gold_tag_seqs = [], []
    for i in range(slot_preds.shape[0]):
        row_mask = mask[i]
        if not row_mask.any():
            continue
        pred_seq = [SLOT_LABELS[t] for t in slot_preds[i][row_mask]]
        gold_seq = [SLOT_LABELS[t] for t in slot_labels[i][row_mask]]
        pred_tag_seqs.append(pred_seq)
        gold_tag_seqs.append(gold_seq)
    if pred_tag_seqs:
        from seqeval.metrics import f1_score as _seq_f1
        slot_f1 = _seq_f1(gold_tag_seqs, pred_tag_seqs, zero_division=0)
    else:
        slot_f1 = 0.0

    # Sentence accuracy: all 3 heads correct
    sa_correct = sa_preds == sa_labels
    dom_correct = dom_preds == dom_labels
    slot_correct = np.all((slot_preds == slot_labels) | (slot_labels == -100), axis=-1)
    sent_acc = (sa_correct & dom_correct & slot_correct).mean()

    # Composite: geometric mean of the 3 head accuracies, using entity F1
    # instead of token accuracy for the slot dimension. This ensures early
    # stopping actually tracks slot performance rather than the trivial
    # O-majority baseline.
    composite = (sa_acc * dom_acc * slot_f1) ** (1.0 / 3.0) if slot_f1 > 0 else 0.0

    return {
        "speech_act_acc": sa_acc,
        "domain_acc": dom_acc,
        "slot_token_acc": slot_acc,
        "slot_entity_f1": slot_f1,
        "sentence_acc": sent_acc,
        "composite_acc": composite,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    set_seed(CONFIG["seed"])

    data_dir = Path(CONFIG["paths"]["data_dir"])
    output_dir = Path(CONFIG["paths"]["output_dir"])
    model_name = CONFIG["model"]["name"]
    max_length = CONFIG["model"]["max_seq_length"]

    # GPU diagnostic
    print(f"Model: {model_name}")
    print(f"Max seq length: {max_length}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 0:
        for i in range(n_gpus):
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB)")
        if n_gpus > 1:
            print(f"Multi-GPU: {n_gpus} devices — Trainer will use DDP automatically")
    else:
        print("WARNING: No GPU detected — training will be very slow on CPU")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Datasets (pre-tokenized — no CPU bottleneck during training)
    print("Pre-tokenizing train set...")
    train_dataset = NluDataset(data_dir / "train", tokenizer, max_length)
    print("Pre-tokenizing dev set...")
    dev_dataset = NluDataset(data_dir / "dev", tokenizer, max_length)
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    # Model
    mc = CONFIG["model"]
    use_crf = mc.get("use_crf", False)
    head_hidden_dim = mc.get("head_hidden_dim", 256)
    lc = CONFIG["loss"]
    gradient_checkpointing = mc.get("gradient_checkpointing", False)

    # Class weighting for imbalanced speech_act / domain heads.
    # With `class_weighting: sqrt_inverse`, rare classes get a higher per-step
    # loss contribution without fully inverting frequency (which would drown
    # majority classes). `none` disables. Weights are L1-normalized so their
    # mean equals 1.0 — this keeps the loss magnitude comparable to unweighted.
    sa_weights, dom_weights, slot_weights = None, None, None
    weighting_mode = lc.get("class_weighting", "none")

    def _compute_weights(counts: Counter, n_classes: int, mode: str) -> torch.Tensor:
        raw = torch.ones(n_classes)
        for idx in range(n_classes):
            c = counts.get(idx, 0)
            if c == 0:
                raw[idx] = 0.0  # unseen class — no gradient signal anyway
            elif mode == "sqrt_inverse":
                raw[idx] = 1.0 / math.sqrt(c)
            elif mode == "inverse":
                raw[idx] = 1.0 / c
            else:
                raise ValueError(f"Unknown class_weighting: {mode}")
        # Normalize so mean weight = 1.0 over classes that are present
        present = raw > 0
        raw[present] = raw[present] * (present.sum().float() / raw[present].sum())
        return raw

    if weighting_mode != "none":
        sa_counts = Counter(train_dataset.speech_act_labels.tolist())
        dom_counts = Counter(train_dataset.domain_labels.tolist())
        sa_weights = _compute_weights(sa_counts, len(SPEECH_ACTS), weighting_mode)
        dom_weights = _compute_weights(dom_counts, len(DOMAINS), weighting_mode)
        print(f"Class weighting: {weighting_mode}")
        print(f"  SA weights (min/max/mean): "
              f"{sa_weights[sa_weights > 0].min():.2f} / "
              f"{sa_weights.max():.2f} / "
              f"{sa_weights[sa_weights > 0].mean():.2f}")
        # Print the per-class speech-act weights (small number, useful)
        for i, sa in enumerate(SPEECH_ACTS):
            print(f"    {sa:20s} count={sa_counts.get(i, 0):5d}  weight={sa_weights[i]:.3f}")

    # Slot-label class weighting. Separate knob because the slot head runs
    # through the CRF, which has no per-class weight support — we add a
    # weighted CE auxiliary loss to push rare BIO labels up at emission time.
    # `slot_emission_aux_weight: 0` disables; 0.3–0.5 is typical — enough to
    # shift the emission prior without overwhelming the CRF's structural signal.
    slot_weighting_mode = lc.get("slot_class_weighting", "none")
    slot_aux_weight = lc.get("slot_emission_aux_weight", 0.0)
    if slot_weighting_mode != "none" and slot_aux_weight > 0:
        # Flatten all real slot labels (ignore -100 and pad tokens)
        flat = train_dataset.slot_labels.view(-1)
        real = flat[flat != -100].tolist()
        slot_counts = Counter(real)
        slot_weights = _compute_weights(slot_counts, len(SLOT_LABELS), slot_weighting_mode)
        print(f"Slot class weighting: {slot_weighting_mode} "
              f"(aux weight × {slot_aux_weight})")
        total = sum(slot_counts.values())
        # Log the top-5 upweighted and O for reference
        ranked = sorted(range(len(SLOT_LABELS)),
                        key=lambda i: -slot_weights[i].item())
        o_idx = SLOT_LABELS.index("O") if "O" in SLOT_LABELS else None
        print(f"  Top-10 upweighted slot labels:")
        for idx in ranked[:10]:
            c = slot_counts.get(idx, 0)
            pct = 100 * c / total if total else 0
            print(f"    {SLOT_LABELS[idx]:24s} count={c:6d} ({pct:5.2f}%)  weight={slot_weights[idx]:.3f}")
        if o_idx is not None:
            c = slot_counts.get(o_idx, 0)
            pct = 100 * c / total if total else 0
            print(f"  For reference:")
            print(f"    {'O':24s} count={c:6d} ({pct:5.2f}%)  weight={slot_weights[o_idx]:.3f}")

    model = JointCamemBERTav2(
        model_name=model_name,
        num_speech_acts=len(SPEECH_ACTS),
        num_domains=len(DOMAINS),
        num_slot_labels=len(SLOT_LABELS),
        use_crf=use_crf,
        head_hidden_dim=head_hidden_dim,
        focal_gamma=lc["focal_gamma"],
        smoothing=lc.get("smoothing", 0.0),
        gradient_checkpointing=gradient_checkpointing,
        speech_act_class_weights=sa_weights,
        domain_class_weights=dom_weights,
        slot_class_weights=slot_weights,
        slot_emission_aux_weight=slot_aux_weight,
        slot_labels=SLOT_LABELS,
    )
    print(f"CRF: {'enabled' if use_crf else 'disabled'}")
    print(f"Gradient checkpointing: {'enabled' if gradient_checkpointing else 'disabled'}")
    print(f"Head hidden dim: {head_hidden_dim}")

    # Training args
    tc = CONFIG["training"]
    early_metric = tc.get("early_stopping_metric", "eval_loss")
    greater_is_better = not early_metric.endswith("_loss")

    # Auto-scale LR with GPU count (linear scaling rule)
    base_lr = tc["learning_rate"]
    if n_gpus > 1:
        scaled_lr = base_lr * math.sqrt(n_gpus)
        print(f"LR scaling: {base_lr:.1e} × √{n_gpus} = {scaled_lr:.1e}")
        base_lr = scaled_lr

    # Auto-detect best mixed precision: bf16 on Ampere+ (sm_80+), fp16 on older
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            use_bf16 = True
            print(f"Mixed precision: bf16 (compute capability {capability[0]}.{capability[1]})")
        else:
            use_fp16 = True
            print(f"Mixed precision: fp16 (compute capability {capability[0]}.{capability[1]})")

    # Compute warmup_steps from ratio (warmup_ratio is deprecated in transformers v5.2).
    effective_batch = (
        tc["per_device_batch_size"]
        * tc["gradient_accumulation_steps"]
        * max(n_gpus, 1)
    )
    total_steps = tc["num_epochs"] * math.ceil(
        len(train_dataset) / effective_batch
    )
    warmup_steps = int(total_steps * tc["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        per_device_eval_batch_size=tc["per_device_batch_size"] * 2,
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=base_lr,
        warmup_steps=warmup_steps,
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        lr_scheduler_type=tc.get("lr_scheduler_type", "linear"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=early_metric,
        greater_is_better=greater_is_better,
        save_total_limit=2,
        logging_steps=50,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        seed=CONFIG["seed"],
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
    )

    # Loss config
    loss_weights = lc["weights"]
    layer_lr_decay = tc.get("layer_lr_decay", 1.0)
    crf_lr_multiplier = tc.get("crf_lr_multiplier", 50.0)

    print(f"Effective batch size: {effective_batch} "
          f"({tc['per_device_batch_size']} × {tc['gradient_accumulation_steps']} accum × {max(n_gpus, 1)} GPU)")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps} ({tc['warmup_ratio']:.0%})")
    print(f"LR scheduler: {tc.get('lr_scheduler_type', 'linear')}")
    print(f"Layer LR decay: {layer_lr_decay}")
    print(f"CRF LR multiplier: {crf_lr_multiplier}x (applied to transition params only)")
    print(f"Label smoothing: {lc.get('smoothing', 0.0)}")
    print(f"Early stopping on: {early_metric} (greater_is_better={greater_is_better})")

    # Sqrt-inverse combo sampling: moderate rebalancing without starving
    # majority classes. Weight = 1/sqrt(count) instead of 1/count.
    # This softly upweights rare combos without making common combos invisible.
    combo_labels = list(zip(
        train_dataset.speech_act_labels.tolist(),
        train_dataset.domain_labels.tolist(),
    ))
    combo_counts = Counter(combo_labels)
    sample_weights = [1.0 / math.sqrt(combo_counts[c]) for c in combo_labels]
    n_combos = len(combo_counts)
    smallest = min(combo_counts.values())
    largest = max(combo_counts.values())
    print(f"Sqrt-inverse combo sampling: {n_combos} combos, "
          f"range {smallest}–{largest} examples, ratio {largest/smallest:.0f}:1")

    # Trainer
    trainer = JointNluTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=NluDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tc["early_stopping_patience"])],
        loss_weights=loss_weights,
        sample_weights=sample_weights,
        layer_lr_decay=layer_lr_decay,
        crf_lr_multiplier=crf_lr_multiplier,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save best model
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")

    # Final eval
    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")

    # Dump full training history (epoch-by-epoch) — needed to diagnose
    # whether training plateaued, early-stopped, or was still improving.
    import json
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
