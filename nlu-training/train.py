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
from torch.utils.data import Dataset, WeightedRandomSampler

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
    torch.backends.cudnn.benchmark = False


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
# Layer-wise LR decay
# ──────────────────────────────────────────────
def _build_param_groups(model, base_lr, layer_lr_decay, weight_decay):
    """
    Build optimizer param groups with layer-wise LR decay.

    Layer 0 (bottom) gets lr * decay^12, layer 11 (top) gets lr * decay^1.
    Embeddings get lr * decay^12. Heads and CRF get lr * 1.0.
    No weight decay on bias and LayerNorm.
    """
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    num_layers = model.encoder.config.num_hidden_layers  # 12

    # Group parameters by layer depth
    groups = {}  # depth → list of (name, param)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine depth: heads/crf = 0 (full LR), encoder layers = 1..12, embeddings = 13
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
            # Heads (speech_act_head, domain_head, slot_head), CRF, dropout
            depth = 0

        groups.setdefault(depth, []).append((name, param))

    # Build param groups
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

    return param_groups


# ──────────────────────────────────────────────
# Custom Trainer
# ──────────────────────────────────────────────
class JointNluTrainer(Trainer):
    """Trainer that passes loss_weights, focal_gamma, and CRF config to the model."""

    def __init__(self, *args, loss_weights=None, sample_weights=None,
                 layer_lr_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights or {"speech_act": 1.0, "domain": 1.0, "slots": 2.0}
        self._sample_weights = sample_weights
        self._layer_lr_decay = layer_lr_decay

    def _get_train_sampler(self, train_dataset=None):
        if self._sample_weights is not None:
            return WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=len(self._sample_weights),
                replacement=True,
            )
        return super()._get_train_sampler()

    def create_optimizer(self):
        """Override to apply layer-wise LR decay."""
        if self._layer_lr_decay < 1.0:
            param_groups = _build_param_groups(
                self.model,
                base_lr=self.args.learning_rate,
                layer_lr_decay=self._layer_lr_decay,
                weight_decay=self.args.weight_decay,
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

        if hasattr(model, "use_crf") and model.use_crf and hasattr(model, "crf"):
            slot_labels = inputs["slot_labels"]
            packed_logits, _, packed_mask = pack_for_crf(
                slot_logits, slot_labels
            )

            # Filter to rows that have real labels (CRF requires first timestep on)
            row_has_labels = packed_mask.any(dim=1)
            if row_has_labels.any():
                decoded_valid = model.crf.decode(
                    packed_logits[row_has_labels],
                    mask=packed_mask[row_has_labels],
                )

                # Scatter decoded tags back to full seq_len positions
                real_mask = (slot_labels != -100)
                viterbi_logits = torch.full_like(slot_logits, -1e4)
                valid_indices = row_has_labels.nonzero(as_tuple=True)[0]
                for di, bi in enumerate(valid_indices):
                    idx = real_mask[bi].nonzero(as_tuple=True)[0]
                    for j, pos in enumerate(idx):
                        if j < len(decoded_valid[di]):
                            viterbi_logits[bi, pos, decoded_valid[di][j]] = 1e4

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
    """Compute per-head accuracy for logging and early stopping."""
    predictions, labels = eval_pred
    sa_logits, dom_logits, slot_logits = predictions
    sa_labels, dom_labels, slot_labels = labels

    sa_preds = np.argmax(sa_logits, axis=-1)
    dom_preds = np.argmax(dom_logits, axis=-1)

    sa_acc = (sa_preds == sa_labels).mean()
    dom_acc = (dom_preds == dom_labels).mean()

    # Slot accuracy (ignoring -100)
    slot_preds = np.argmax(slot_logits, axis=-1)
    mask = slot_labels != -100
    slot_acc = (slot_preds[mask] == slot_labels[mask]).mean() if mask.any() else 0.0

    # Sentence accuracy: all 3 heads correct
    sa_correct = sa_preds == sa_labels
    dom_correct = dom_preds == dom_labels
    slot_correct = np.all((slot_preds == slot_labels) | (slot_labels == -100), axis=-1)
    sent_acc = (sa_correct & dom_correct & slot_correct).mean()

    # Weighted composite: geometric mean of the 3 head accuracies.
    # Ensures early stopping balances all heads — no single head can
    # inflate the metric while another collapses.
    composite = (sa_acc * dom_acc * slot_acc) ** (1.0 / 3.0) if slot_acc > 0 else 0.0

    return {
        "speech_act_acc": sa_acc,
        "domain_acc": dom_acc,
        "slot_token_acc": slot_acc,
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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    model = JointCamemBERTav2(
        model_name=model_name,
        num_speech_acts=len(SPEECH_ACTS),
        num_domains=len(DOMAINS),
        num_slot_labels=len(SLOT_LABELS),
        use_crf=use_crf,
        head_hidden_dim=head_hidden_dim,
        focal_gamma=lc["focal_gamma"],
        smoothing=lc.get("smoothing", 0.0),
    )
    print(f"CRF: {'enabled' if use_crf else 'disabled'}")
    print(f"Head hidden dim: {head_hidden_dim}")

    # Training args
    tc = CONFIG["training"]
    early_metric = tc.get("early_stopping_metric", "eval_loss")
    greater_is_better = not early_metric.endswith("_loss")

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

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        per_device_eval_batch_size=tc["per_device_batch_size"] * 2,
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        lr_scheduler_type=tc.get("lr_scheduler_type", "linear"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=early_metric,
        greater_is_better=greater_is_better,
        save_total_limit=3,
        logging_steps=50,
        dataloader_num_workers=0,  # pre-tokenized data = no CPU work to parallelize
        dataloader_pin_memory=True,
        seed=CONFIG["seed"],
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
    )

    # Loss config
    loss_weights = lc["weights"]
    layer_lr_decay = tc.get("layer_lr_decay", 1.0)

    print(f"LR scheduler: {tc.get('lr_scheduler_type', 'linear')}")
    print(f"Layer LR decay: {layer_lr_decay}")
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


if __name__ == "__main__":
    main()
