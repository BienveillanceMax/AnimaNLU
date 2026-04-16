"""
Prepare NLU training data from MASSIVE + supplements.

1. Download MASSIVE French via HuggingFace datasets
2. Remap intents → (speech_act, domain)
3. Remap slots BIO → our 18 types (unmapped → -100)
4. Split MASSIVE 80/10/10 stratified by (speech_act, domain)
5. Load supplements from data/supplements/ → TRAIN ONLY
6. Write 4 files per split: seq.in, seq.out, speech_act, domain
"""

import json
import os
import re
import random
import yaml
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from labels import (
    SPEECH_ACTS, DOMAINS, SLOT_LABELS,
    SPEECH_ACT_L2I, DOMAIN_L2I, SLOT_LABEL_L2I,
    MASSIVE_INTENT_MAP, MASSIVE_SLOT_MAP,
    write_label_files,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["seed"]
DATA_DIR = Path(CONFIG["paths"]["data_dir"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Precompiled patterns for text quality filter
_RE_TRUNCATED = re.compile(r'[a-zà-ü]-$')
_RE_MULTI_SENTENCE = re.compile(r'\w{2,}[.!?]\s+[A-ZÀ-Ü]')
_RE_GARBLED_SMS = re.compile(r'\bsa (va|fait|sert)\b', re.IGNORECASE)


# ──────────────────────────────────────────────
# Step 1: Load MASSIVE French
# ──────────────────────────────────────────────
def load_massive_fr():
    """Download and filter MASSIVE for French examples."""
    from datasets import load_dataset

    print("Loading MASSIVE dataset...")
    ds = load_dataset("AmazonScience/massive", revision="refs/convert/parquet")

    # Get intent name mapping from features
    intent_names = ds["train"].features["intent"].names

    fr_examples = []
    for split_name in ["train", "validation", "test"]:
        split = ds[split_name].filter(lambda x: x["locale"] == "fr-FR")
        for ex in split:
            intent_name = intent_names[ex["intent"]]
            fr_examples.append({
                "utt": ex["utt"],
                "annot_utt": ex["annot_utt"],
                "intent": intent_name,
            })

    print(f"  Loaded {len(fr_examples)} French examples from MASSIVE")
    return fr_examples


# ──────────────────────────────────────────────
# Step 2: Parse MASSIVE annotated utterances
# ──────────────────────────────────────────────
def parse_annot_utt(annot_utt: str) -> tuple[list[str], list[str]]:
    """
    Parse MASSIVE annotated utterance into words and MASSIVE BIO tags.

    MASSIVE format: "réveille-moi à [time : neuf heures du matin] le [date : vendredi]"
    → words: ["réveille-moi", "à", "neuf", "heures", "du", "matin", "le", "vendredi"]
    → tags:  ["O", "O", "B-time", "I-time", "I-time", "I-time", "O", "B-date"]
    """
    words = []
    tags = []
    i = 0
    text = annot_utt.strip()

    while i < len(text):
        if text[i] == "[":
            # Parse slot: [slot_type : slot_value]
            end = text.index("]", i)
            content = text[i+1:end]
            slot_type, slot_value = content.split(":", 1)
            slot_type = slot_type.strip()
            slot_value = slot_value.strip()

            slot_words = slot_value.split()
            for j, w in enumerate(slot_words):
                words.append(w)
                prefix = "B-" if j == 0 else "I-"
                tags.append(f"{prefix}{slot_type}")

            i = end + 1
        elif text[i] == " ":
            i += 1
        else:
            # Regular word
            end = i
            while end < len(text) and text[end] not in " [":
                end += 1
            words.append(text[i:end])
            tags.append("O")
            i = end

    return words, tags


# ──────────────────────────────────────────────
# Step 3: Remap to our taxonomy
# ──────────────────────────────────────────────
def remap_example(utt: str, annot_utt: str, intent: str):
    """
    Remap a MASSIVE example to our taxonomy.

    Returns None if intent not in our map (shouldn't happen).
    """
    if intent not in MASSIVE_INTENT_MAP:
        return None

    speech_act, domain = MASSIVE_INTENT_MAP[intent]
    words, massive_tags = parse_annot_utt(annot_utt)

    # Remap slot tags
    our_tags = []
    for tag in massive_tags:
        if tag == "O":
            our_tags.append("O")
        else:
            prefix = tag[:2]  # "B-" or "I-"
            massive_slot = tag[2:]
            our_slot = MASSIVE_SLOT_MAP.get(massive_slot)
            if our_slot is not None:
                our_tags.append(f"{prefix}{our_slot}")
            else:
                # Unmapped slot → special marker, will become -100 in training
                our_tags.append("IGNORE")

    return {
        "text": " ".join(words),
        "bio_tags": our_tags,
        "speech_act": speech_act,
        "domain": domain,
    }


# ──────────────────────────────────────────────
# Step 4: Load supplements
# ──────────────────────────────────────────────
def _load_yaml_dir(directory: Path, warn_mismatches=False):
    """Load all YAML files from a directory into a flat example list."""
    examples = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "examples" not in data:
            continue
        sa, dom = data["speech_act"], data["domain"]
        for ex in data["examples"]:
            text = ex["text"]
            words = text.split()
            bio_tags = ex.get("bio_tags", ["O"] * len(words))
            if len(bio_tags) != len(words):
                if warn_mismatches:
                    print(f"  WARNING: tag count mismatch in {yaml_file.name}: "
                          f"'{text}' has {len(words)} words but {len(bio_tags)} tags, skipping")
                continue
            examples.append({"text": text, "bio_tags": bio_tags, "speech_act": sa, "domain": dom})
    return examples


def load_supplements():
    """Load supplement data from data/supplements/*.yaml files."""
    supp_dir = DATA_DIR / "supplements"
    if not supp_dir.exists():
        print("  No supplements directory found, skipping.")
        return []
    examples = _load_yaml_dir(supp_dir, warn_mismatches=True)
    print(f"  Loaded {len(examples)} supplement examples from {supp_dir}")
    return examples


def _is_clean_text(text: str) -> bool:
    """Reject multi-sentence, garbled, and truncated utterances."""
    stripped = text.rstrip()
    if stripped.endswith('—') or _RE_TRUNCATED.search(stripped):
        return False
    if _RE_MULTI_SENTENCE.search(text):
        return False
    if _RE_GARBLED_SMS.search(text):
        return False
    return True


def _filter_and_resolve_conflicts(examples, name):
    """Remove garbled text and resolve label conflicts.

    Classification conflicts (different SA/DOM): first label wins.
    Slot tag conflicts (same SA/DOM, different tags): prefer version with most entities.
    """
    clean = []
    seen = {}  # key → (label_tuple, entity_count, index_in_clean)
    removed_dirty = 0
    removed_conflict = 0
    removed_tag_dedup = 0
    for ex in examples:
        if not _is_clean_text(ex["text"]):
            removed_dirty += 1
            continue
        key = ex["text"].lower().strip()
        label = (ex["speech_act"], ex["domain"])
        entity_count = sum(1 for t in ex["bio_tags"] if t.startswith("B-"))
        if key in seen:
            prev_label, prev_entity_count, prev_idx = seen[key]
            if prev_label != label:
                removed_conflict += 1
                continue
            # Same SA/DOM — keep version with more entities
            if entity_count > prev_entity_count:
                clean[prev_idx] = ex
                seen[key] = (label, entity_count, prev_idx)
            removed_tag_dedup += 1
            continue
        else:
            idx = len(clean)
            seen[key] = (label, entity_count, idx)
        clean.append(ex)
    print(f"  {name}: removed {removed_dirty} dirty + {removed_conflict} conflicts "
          f"+ {removed_tag_dedup} tag dedup → {len(clean)}")
    return clean


# ──────────────────────────────────────────────
# Step 5: Split and write
# ──────────────────────────────────────────────
def stratified_split(examples, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """Split examples stratified by (speech_act, domain) combo."""
    combo_labels = [f"{ex['speech_act']}_{ex['domain']}" for ex in examples]

    # First split: train+dev vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=SEED)
    train_dev_idx, test_idx = next(sss1.split(examples, combo_labels))

    # Second split: train vs dev (from the train+dev portion)
    train_dev_examples = [examples[i] for i in train_dev_idx]
    train_dev_labels = [combo_labels[i] for i in train_dev_idx]

    dev_fraction = dev_ratio / (train_ratio + dev_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=SEED)
    train_idx_inner, dev_idx_inner = next(sss2.split(train_dev_examples, train_dev_labels))

    train = [train_dev_examples[i] for i in train_idx_inner]
    dev = [train_dev_examples[i] for i in dev_idx_inner]
    test = [examples[i] for i in test_idx]

    return train, dev, test


def write_split(examples, split_dir: Path):
    """Write the 4 files for a split."""
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "seq.in", "w", encoding="utf-8") as f_in, \
         open(split_dir / "seq.out", "w", encoding="utf-8") as f_out, \
         open(split_dir / "speech_act", "w", encoding="utf-8") as f_sa, \
         open(split_dir / "domain", "w", encoding="utf-8") as f_dom:

        for ex in examples:
            f_in.write(ex["text"] + "\n")
            f_out.write(" ".join(ex["bio_tags"]) + "\n")
            f_sa.write(ex["speech_act"] + "\n")
            f_dom.write(ex["domain"] + "\n")


def print_stats(name, examples):
    """Print distribution stats for a split."""
    combo_counter = Counter(f"{ex['speech_act']} × {ex['domain']}" for ex in examples)
    sa_counter = Counter(ex["speech_act"] for ex in examples)
    dom_counter = Counter(ex["domain"] for ex in examples)

    print(f"\n{'='*60}")
    print(f"  {name}: {len(examples)} examples")
    print(f"{'='*60}")

    print(f"\n  Speech Acts:")
    for sa in SPEECH_ACTS:
        count = sa_counter.get(sa, 0)
        print(f"    {sa:20s}: {count:5d}")

    print(f"\n  Domains:")
    for dom in DOMAINS:
        count = dom_counter.get(dom, 0)
        print(f"    {dom:20s}: {count:5d}")

    print(f"\n  Top combos:")
    for combo, count in combo_counter.most_common(30):
        flag = " ⚠ LOW" if count < 30 else ""
        print(f"    {combo:35s}: {count:5d}{flag}")

    # Check for combos with < 30 examples
    low = [(c, n) for c, n in combo_counter.items() if n < 30]
    if low:
        print(f"\n  ⚠ {len(low)} combos with < 30 examples")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    set_seed(SEED)

    # 1. Load and remap MASSIVE
    massive_examples = load_massive_fr()
    remapped = []
    skipped = 0
    for ex in massive_examples:
        result = remap_example(ex["utt"], ex["annot_utt"], ex["intent"])
        if result is not None:
            remapped.append(result)
        else:
            skipped += 1

    print(f"  Remapped: {len(remapped)}, Skipped: {skipped}")

    # 2. Split MASSIVE only (dev/test = pure human data)
    train, dev, test = stratified_split(remapped)
    print(f"  MASSIVE split — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    # 3. Load handwritten supplements → TRAIN ONLY (upsampled)
    supplements = load_supplements()
    if supplements:
        upsample = CONFIG.get("supplement_upsample_factor", 1)
        for _ in range(upsample):
            train.extend(supplements)
        print(f"  Supplements: {len(supplements)} × {upsample} = {len(supplements) * upsample}")
        print(f"  After supplements — train: {len(train)}")

    # 4. Load Sonnet-classified dialogue extracts (from classify_results/)
    #    100% train (Sonnet labels too noisy for eval)
    classify_results_dir = DATA_DIR / "classify_results"
    if classify_results_dir.exists():
        dialogue_all = []
        seen = set()
        skipped_long = 0
        skipped_stmt_q = 0
        for f in sorted(classify_results_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                classify_data = json.load(fh)
            for ex in classify_data:
                # Validate speech_act and domain are in our taxonomy
                sa, dom = ex.get("speech_act"), ex.get("domain")
                if sa not in SPEECH_ACT_L2I or dom not in DOMAIN_L2I:
                    continue
                key = ex["text"].lower().strip()
                if key in seen or len(key.split()) < 2:
                    continue
                # Fix 4: Quality filters for Sonnet data
                if len(key.split()) > 20:
                    skipped_long += 1
                    continue
                if sa == "Statement" and "?" in ex["text"]:
                    skipped_stmt_q += 1
                    continue
                seen.add(key)
                words = ex["text"].split()
                dialogue_all.append({
                    "text": ex["text"],
                    "bio_tags": ["IGNORE"] * len(words),
                    "speech_act": sa,
                    "domain": dom,
                })

        print(f"  Sonnet classified: {len(dialogue_all)} unique "
              f"(skipped {skipped_long} long + {skipped_stmt_q} Statement-with-?)")

        # Fix 3: Cap Sonnet-Social to reduce domain bias
        sonnet_cap = CONFIG.get("sonnet_social_cap", 2000)
        sonnet_social = [ex for ex in dialogue_all if ex["domain"] == "Social"]
        sonnet_other = [ex for ex in dialogue_all if ex["domain"] != "Social"]
        original_social = len(sonnet_social)
        if len(sonnet_social) > sonnet_cap:
            random.shuffle(sonnet_social)
            sonnet_social = sonnet_social[:sonnet_cap]
        dialogue_all = sonnet_other + sonnet_social
        print(f"    Social capped: {original_social} → {len(sonnet_social)}")

        # Fix 2: All to train, none to eval
        train.extend(dialogue_all)
        print(f"    → train: {len(dialogue_all)} (0 to eval)")

    # 4b. Load old regex-extracted dialogues (backwards compat, smaller set)
    dialogues_train_dir = DATA_DIR / "dialogues_train"
    dialogues_eval_dir = DATA_DIR / "dialogues_eval"

    if dialogues_train_dir.exists():
        dt = _load_yaml_dir(dialogues_train_dir)
        train.extend(dt)
        print(f"  Dialogue regex train: {len(dt)} examples")

    if dialogues_eval_dir.exists():
        de = _load_yaml_dir(dialogues_eval_dir)
        dev.extend(de)
        test.extend(de)
        print(f"  Dialogue regex eval: {len(de)} examples → both dev and test")

    # 4c. Quality filters and leakage removal
    train = _filter_and_resolve_conflicts(train, "train")
    dev = _filter_and_resolve_conflicts(dev, "dev")
    test = _filter_and_resolve_conflicts(test, "test")

    # Remove train↔eval leakage
    eval_keys = set(ex["text"].lower().strip() for ex in dev + test)
    before = len(train)
    train = [ex for ex in train if ex["text"].lower().strip() not in eval_keys]
    print(f"  Leakage: removed {before - len(train)} train examples shared with dev/test")

    # 5. Balance training set: cap + floor per combo
    bal = CONFIG.get("balancing", {})
    max_per = bal.get("max_per_combo", 99999)
    min_per = bal.get("min_per_combo", 0)

    if max_per < 99999 or min_per > 0:
        by_combo = defaultdict(list)
        for ex in train:
            by_combo[f"{ex['speech_act']}|{ex['domain']}"].append(ex)

        balanced_train = []
        for combo, examples in sorted(by_combo.items()):
            n = len(examples)
            if n > max_per:
                # Downsample
                random.shuffle(examples)
                balanced_train.extend(examples[:max_per])
            elif n < min_per:
                # Upsample by repeating
                repeats = (min_per // n) + 1
                upsampled = (examples * repeats)[:min_per]
                balanced_train.extend(upsampled)
            else:
                balanced_train.extend(examples)

        print(f"  Balancing: {len(train)} → {len(balanced_train)} "
              f"(cap={max_per}, floor={min_per})")
        train = balanced_train

    # 6. Entity balance: cap entity-free examples to reduce CRF dilution
    entity_examples, no_entity_examples = [], []
    for ex in train:
        if any(t.startswith("B-") or t.startswith("I-") for t in ex["bio_tags"]):
            entity_examples.append(ex)
        else:
            no_entity_examples.append(ex)
    entity_cap = len(entity_examples)
    if len(no_entity_examples) > entity_cap:
        random.shuffle(no_entity_examples)
        no_entity_examples = no_entity_examples[:entity_cap]
    print(f"  Entity balance: {len(entity_examples)} with entities, "
          f"{len(no_entity_examples)} without (capped to match)")
    train = entity_examples + no_entity_examples

    print(f"  Final — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    # 7. Shuffle train
    random.shuffle(train)

    # 8. Write splits
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        write_split(split_data, DATA_DIR / split_name)

    # 9. Write label files
    write_label_files(str(DATA_DIR / "labels"))

    # 10. Stats
    print_stats("TRAIN", train)
    print_stats("DEV", dev)
    print_stats("TEST", test)

    # 11. Identify train-only combos (synthetic, no eval data)
    train_combo_counts = Counter(f"{ex['speech_act']} × {ex['domain']}" for ex in train)
    eval_combos = set(f"{ex['speech_act']} × {ex['domain']}" for ex in dev + test)
    train_only = set(train_combo_counts) - eval_combos
    if train_only:
        print(f"\n{'='*60}")
        print(f"  ⚠ TRAIN-ONLY combos (no eval data — synthetic only):")
        for combo in sorted(train_only):
            print(f"    {combo}: {train_combo_counts[combo]} examples")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
