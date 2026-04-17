"""
Evaluate the trained JointCamemBERTav2 model on the test set.

Reports:
- Speech act accuracy (target > 95%)
- Domain accuracy (target > 93%)
- Slot F1 entity-level via seqeval (target > 90%)
- Sentence accuracy (target > 82%)
- Confusion matrices per head
- Per-combo breakdown
- BIO violation count (should be 0 with CRF)
"""

import yaml
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1

from labels import (
    SPEECH_ACTS, DOMAINS, SLOT_LABELS,
    SPEECH_ACT_I2L, DOMAIN_I2L, SLOT_LABEL_I2L,
)
from model import JointCamemBERTav2, pack_for_crf
from train import NluDataset, set_seed

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def _count_bio_violations(tag_seq):
    """Count BIO violations: I-X without matching B-X/I-X predecessor."""
    violations = 0
    prev_tag = "O"
    for tag in tag_seq:
        if tag.startswith("I-"):
            slot_type = tag[2:]
            if prev_tag != f"B-{slot_type}" and prev_tag != f"I-{slot_type}":
                violations += 1
        prev_tag = tag
    return violations


def evaluate(model, dataset, device, use_crf=False):
    """Run inference on entire dataset, return all predictions and labels."""
    model.eval()
    all_sa_preds, all_sa_labels = [], []
    all_dom_preds, all_dom_labels = [], []
    all_slot_preds, all_slot_labels = [], []

    batch_size = 64
    n = len(dataset)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            input_ids = dataset.input_ids[start:end].to(device)
            attention_mask = dataset.attention_mask[start:end].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Speech act
            sa_preds = outputs["speech_act_logits"].argmax(-1).cpu().tolist()
            all_sa_preds.extend(sa_preds)
            all_sa_labels.extend(dataset.speech_act_labels[start:end].tolist())

            # Domain
            dom_preds = outputs["domain_logits"].argmax(-1).cpu().tolist()
            all_dom_preds.extend(dom_preds)
            all_dom_labels.extend(dataset.domain_labels[start:end].tolist())

            # Slots
            slot_logits = outputs["slot_logits"]  # (batch, seq_len, 37)

            if use_crf and hasattr(model, "crf"):
                slot_labels_batch = dataset.slot_labels[start:end]
                packed_logits, _, packed_mask = pack_for_crf(
                    slot_logits, slot_labels_batch.to(device)
                )

                # Filter to rows with real labels (CRF requires first timestep on)
                row_has_labels = packed_mask.any(dim=1)
                real_mask = (slot_labels_batch != -100)
                bs = end - start

                if row_has_labels.any():
                    decoded_valid = model.crf.decode(
                        packed_logits[row_has_labels],
                        mask=packed_mask[row_has_labels],
                    )

                    # Map decoded results back to original batch indices
                    valid_indices = row_has_labels.nonzero(as_tuple=True)[0].tolist()
                    decoded_by_row = {}
                    for di, bi in enumerate(valid_indices):
                        decoded_by_row[bi] = decoded_valid[di]

                    for i in range(bs):
                        slot_labels_i = slot_labels_batch[i].numpy()
                        real_positions = real_mask[i].nonzero(as_tuple=True)[0].tolist()

                        if i in decoded_by_row:
                            decoded_i = decoded_by_row[i]
                            pred_seq, label_seq = [], []
                            for k, j in enumerate(real_positions):
                                p = decoded_i[k] if k < len(decoded_i) else 0
                                pred_seq.append(SLOT_LABEL_I2L[p])
                                label_seq.append(SLOT_LABEL_I2L[slot_labels_i[j]])
                            all_slot_preds.append(pred_seq)
                            all_slot_labels.append(label_seq)
                        else:
                            all_slot_preds.append([])
                            all_slot_labels.append([])
                else:
                    for i in range(bs):
                        all_slot_preds.append([])
                        all_slot_labels.append([])
            else:
                # Argmax fallback
                slot_preds_batch = slot_logits.argmax(-1).cpu().numpy()
                for i in range(end - start):
                    slot_labels_i = dataset.slot_labels[start + i].numpy()
                    preds_i = slot_preds_batch[i]

                    pred_seq, label_seq = [], []
                    for p, l in zip(preds_i, slot_labels_i):
                        if l != -100:
                            pred_seq.append(SLOT_LABEL_I2L[p])
                            label_seq.append(SLOT_LABEL_I2L[l])
                    all_slot_preds.append(pred_seq)
                    all_slot_labels.append(label_seq)

    return all_sa_preds, all_sa_labels, all_dom_preds, all_dom_labels, all_slot_preds, all_slot_labels


def print_confusion_matrix(labels, preds, label_names, title):
    """Print a simple confusion matrix."""
    n = len(label_names)
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, preds):
        matrix[true][pred] += 1

    print(f"\n{'='*60}")
    print(f"  Confusion Matrix: {title}")
    print(f"{'='*60}")
    # Header
    print(f"{'':20s}", end="")
    for name in label_names:
        print(f"{name[:8]:>9s}", end="")
    print()
    # Rows
    for i, name in enumerate(label_names):
        row_sum = matrix[i].sum()
        if row_sum == 0:
            continue
        print(f"{name:20s}", end="")
        for j in range(n):
            val = matrix[i][j]
            if val > 0:
                print(f"{val:9d}", end="")
            else:
                print(f"{'·':>9s}", end="")
        acc = matrix[i][i] / row_sum if row_sum > 0 else 0
        print(f"  ({acc:.0%})")


def main():
    set_seed(CONFIG["seed"])

    data_dir = Path(CONFIG["paths"]["data_dir"])
    output_dir = Path(CONFIG["paths"]["output_dir"])
    model_name = CONFIG["model"]["name"]
    max_length = CONFIG["model"]["max_seq_length"]
    use_crf = CONFIG["model"].get("use_crf", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CRF: {'enabled' if use_crf else 'disabled'}")

    # Load model
    best_dir = output_dir / "best_model"
    model = JointCamemBERTav2(
        model_name=model_name,
        use_crf=use_crf,
        head_hidden_dim=CONFIG["model"].get("head_hidden_dim", 256),
    )
    from safetensors.torch import load_file
    model.load_state_dict(load_file(best_dir / "model.safetensors", device=str(device)))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load test set
    test_dataset = NluDataset(data_dir / "test", tokenizer, max_length)
    print(f"Test set: {len(test_dataset)} examples")

    # Evaluate
    sa_preds, sa_labels, dom_preds, dom_labels, slot_preds, slot_labels = evaluate(
        model, test_dataset, device, use_crf=use_crf
    )

    # ── Metrics ──
    sa_acc = sum(p == l for p, l in zip(sa_preds, sa_labels)) / len(sa_labels)
    dom_acc = sum(p == l for p, l in zip(dom_preds, dom_labels)) / len(dom_labels)
    slot_f1 = seq_f1(slot_labels, slot_preds)

    # Sentence accuracy
    sent_correct = 0
    for i in range(len(sa_preds)):
        if (sa_preds[i] == sa_labels[i] and
                dom_preds[i] == dom_labels[i] and
                slot_preds[i] == slot_labels[i]):
            sent_correct += 1
    sent_acc = sent_correct / len(sa_preds)

    # BIO violation count
    total_violations = 0
    sequences_with_violations = 0
    for pred_seq in slot_preds:
        v = _count_bio_violations(pred_seq)
        total_violations += v
        if v > 0:
            sequences_with_violations += 1

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Speech Act Accuracy : {sa_acc:.4f}  {'PASS' if sa_acc > 0.95 else 'FAIL'} (target > 0.95)")
    print(f"  Domain Accuracy     : {dom_acc:.4f}  {'PASS' if dom_acc > 0.93 else 'FAIL'} (target > 0.93)")
    print(f"  Slot F1 (entity)    : {slot_f1:.4f}  {'PASS' if slot_f1 > 0.90 else 'FAIL'} (target > 0.90)")
    print(f"  Sentence Accuracy   : {sent_acc:.4f}  {'PASS' if sent_acc > 0.82 else 'FAIL'} (target > 0.82)")
    print(f"  BIO Violations      : {total_violations} total in {sequences_with_violations}/{len(slot_preds)} sequences")
    print(f"{'='*60}")

    # Confusion matrices
    print_confusion_matrix(sa_labels, sa_preds, SPEECH_ACTS, "Speech Acts")
    print_confusion_matrix(dom_labels, dom_preds, DOMAINS, "Domains")

    # Slot report (seqeval)
    print(f"\n{'='*60}")
    print(f"  Slot Classification Report (seqeval)")
    print(f"{'='*60}")
    print(seq_report(slot_labels, slot_preds))

    # Per-combo report
    print(f"\n{'='*60}")
    print(f"  Per-Combo Accuracy")
    print(f"{'='*60}")
    combo_stats = defaultdict(lambda: {"total": 0, "sa_correct": 0, "dom_correct": 0, "both_correct": 0})
    for i in range(len(sa_preds)):
        combo = f"{SPEECH_ACT_I2L[sa_labels[i]]} × {DOMAIN_I2L[dom_labels[i]]}"
        combo_stats[combo]["total"] += 1
        if sa_preds[i] == sa_labels[i]:
            combo_stats[combo]["sa_correct"] += 1
        if dom_preds[i] == dom_labels[i]:
            combo_stats[combo]["dom_correct"] += 1
        if sa_preds[i] == sa_labels[i] and dom_preds[i] == dom_labels[i]:
            combo_stats[combo]["both_correct"] += 1

    for combo in sorted(combo_stats.keys()):
        s = combo_stats[combo]
        sa_a = s["sa_correct"] / s["total"]
        dom_a = s["dom_correct"] / s["total"]
        both_a = s["both_correct"] / s["total"]
        print(f"  {combo:35s}  n={s['total']:4d}  sa={sa_a:.2f}  dom={dom_a:.2f}  both={both_a:.2f}")


if __name__ == "__main__":
    main()
