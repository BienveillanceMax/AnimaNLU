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
    MASSIVE_INTENT_MAP, MASSIVE_INTENT_SKIP, MASSIVE_SLOT_MAP,
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
    if intent not in MASSIVE_INTENT_MAP or intent in MASSIVE_INTENT_SKIP:
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


# ──────────────────────────────────────────────
# Time-phrase BIO normalization
# ──────────────────────────────────────────────
# The raw data is inconsistent about whether tokens like "du matin", "ce soir",
# "à midi" are included in time_value spans. An audit
# (see diag_time_conventions.py) showed ~80%+ of each pattern uses the same
# convention, with the rest being annotation noise. We apply the majority
# convention uniformly to all splits so train and test agree.
#
# Rules (driven by the audit):
#   1. '<N> heures du <matin|soir|...>' → 'du <part>' becomes I-time_value
#      (83% of 311 occurrences follow this; normalize the 17% holdouts).
#   2. '<demain|aujourd'hui|<weekday>> <matin|soir|...>' → if first word is
#      B-date_value, force second to B-time_value (2 separate spans).
#      (50 vs 7 majority).
#   3. 'ce/cette <matin|soir|...>' → force B-time_value | I-time_value
#      (95 vs 52 holdouts spread across O|B, O|O, etc).
#   4. 'à <midi|minuit>' → force O | B-time_value (38 vs 3).
_TIME_PARTS = {"matin", "soir", "après-midi", "apres-midi", "nuit",
               "midi", "minuit", "matinée", "soirée"}
_TIME_WEEKDAYS = {"lundi", "mardi", "mercredi", "jeudi", "vendredi",
                  "samedi", "dimanche"}
_TIME_DATE_PRONOUNS = {"demain", "hier", "aujourd'hui"} | _TIME_WEEKDAYS

# Rule 5: French number words that can precede 'heures' in a clock-time phrase.
# Audit Pattern C: gold inconsistency on '(à) <num> heures [du <part>]'.
# Majority convention is: 'à' OUT, '<num> heures' IN, optional 'du <part>' IN.
_FR_NUMERALS = {
    "une", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf", "vingt", "vingt-et-une", "vingt-et-un",
    "vingt-deux", "vingt-trois", "vingt-quatre", "zéro",
}
# Rule 6: multi-day period tokens. Audit Pattern E shows gold overwhelmingly
# uses date_value for durations like 'trois jours', 'une semaine', 'un mois'.
# Only demote when preceded by a cardinal/quantifier, NOT by a demonstrative
# like 'ce/cette/la/le' (those signal a specific window → keep time_value).
_PERIOD_TOKENS = {"jour", "jours", "journée", "journées", "semaine", "semaines",
                  "mois", "année", "années", "an", "ans", "week-end", "weekend"}
_PERIOD_QUANTIFIERS = _FR_NUMERALS | {
    "quelques", "plusieurs", "prochains", "prochaines", "derniers", "dernières",
    "dernier", "dernière", "prochain", "prochaine",
}
# Rule 7: timezone tokens that gold inconsistently labels time_value but are
# semantically locations. Audit Pattern F identifies these; model predicts
# location (reasonable). Demote to O to stop training on nonsense spans.
_TIMEZONE_TOKENS = {"pacifique", "montagne", "pacific", "atlantique",
                    "centrale", "orientale", "occidentale"}


# ──────────────────────────────────────────────
# person_name BIO normalization
# ──────────────────────────────────────────────
# Audit finding: ~17 cases of `de <name>` / `d' <name>` where gold inconsistently
# includes the preposition. Majority convention: preposition OUT.
# Apply uniformly to all splits.
_PERSON_DE_PREPS = {"de", "d'", "du", "des"}


def normalize_person_name_bio(words: list[str],
                              tags: list[str]) -> list[str]:
    """Drop leading 'de'/'d'' preposition from person_name spans.

    Example:
      words:  ['email', 'de', 'paul']
      before: [O,        B-person_name, I-person_name]
      after:  [O,        O,             B-person_name]
    """
    if len(words) != len(tags):
        return tags
    tags = list(tags)
    lw = [w.lower() for w in words]
    n = len(tags)
    i = 0
    while i < n:
        if not tags[i].startswith("B-person_name"):
            i += 1
            continue
        j = i + 1
        while j < n and tags[j] == "I-person_name":
            j += 1
        if lw[i] in _PERSON_DE_PREPS and i + 1 < j:
            tags[i] = "O"
            tags[i + 1] = "B-person_name"
        elif lw[i] in _PERSON_DE_PREPS:
            # Lone 'de' tagged person_name — clear it
            tags[i] = "O"
        i = j
    return tags


# ──────────────────────────────────────────────
# reminder_content BIO normalization
# ──────────────────────────────────────────────
# Audit finding: gold mixes five incompatible conventions for "liste de X":
#   A. (majority, ~25) liste+de O, content tagged — e.g. "liste de [courses]"
#   B. (~7) whole phrase tagged — "[liste de vacances]"
#   C. (~2) "de X" tagged without "liste"
#   D/E. "choses" sometimes in span, sometimes out
# Convention A matches the v2 supplement. We normalize toward A in BOTH train
# AND test/dev so the model is scored against a consistent ground truth. This
# is the fix the audit called "strong recommendation: re-normalize gold".
_LISTE_WORDS = {"liste", "listes"}
_DE_WORDS = {"de", "d'", "du", "des"}


def normalize_reminder_content_bio(words: list[str],
                                   tags: list[str]) -> list[str]:
    """Normalize 'liste de X' reminder_content spans to Convention A.

    Rule: if a B/I-reminder_content span starts on 'liste' and the following
    token is a `de`-family word, push the span start past `liste de` onto the
    content noun.

    Example transformation:
      words:  ['ouvre', 'ma', 'liste', 'de', 'vacances']
      before: [O,       O,    B-rc,    I-rc, I-rc]
      after:  [O,       O,    O,       O,    B-rc]

    Leaves Convention-A spans untouched (they never start on 'liste').
    """
    if len(words) != len(tags):
        return tags
    tags = list(tags)
    lw = [w.lower() for w in words]
    n = len(tags)

    i = 0
    while i < n:
        t = tags[i]
        if not t.startswith("B-reminder_content"):
            i += 1
            continue
        # Walk to find span end
        j = i + 1
        while j < n and tags[j] == "I-reminder_content":
            j += 1
        # Span is [i, j). Check Convention B/C patterns.
        if lw[i] in _LISTE_WORDS and i + 1 < j and lw[i + 1] in _DE_WORDS:
            # Convention B: span includes 'liste de'. Push start to i+2.
            if i + 2 < j:
                tags[i] = "O"
                tags[i + 1] = "O"
                tags[i + 2] = "B-reminder_content"
                # tags[i+3:j] already I-reminder_content, leave them
            else:
                # Span was only 'liste de' — clear it entirely (degenerate case)
                for k in range(i, j):
                    tags[k] = "O"
        elif lw[i] in _DE_WORDS:
            # Convention C: span starts on 'de' — drop the preposition.
            tags[i] = "O"
            if i + 1 < j:
                tags[i + 1] = "B-reminder_content"
            # else span was just a lone 'de', it's now all-O
        i = j

    return tags


def normalize_time_bio(words: list[str], tags: list[str]) -> list[str]:
    """Apply the four time-phrase normalization rules. Returns new tag list."""
    if len(words) != len(tags):
        return tags
    tags = list(tags)
    lw = [w.lower() for w in words]

    for i, w in enumerate(lw):
        if w not in _TIME_PARTS:
            continue
        prev1 = lw[i-1] if i >= 1 else ""
        prev2 = lw[i-2] if i >= 2 else ""
        tag_i = tags[i]
        tag_p1 = tags[i-1] if i >= 1 else ""
        tag_p2 = tags[i-2] if i >= 2 else ""

        # Rule 1: 'heures du <part>' — extend the time_value span over 'du <part>'.
        # Fires when 'heures' is already inside a time_value span.
        if prev2 == "heures" and prev1 == "du" and tag_p2 in ("B-time_value", "I-time_value"):
            tags[i-1] = "I-time_value"
            tags[i] = "I-time_value"
            continue

        # Rule 3: 'ce/cette <part>' — one time_value span, demonstrative included.
        if prev1 in ("ce", "cette"):
            # Only rewrite if not already part of a larger span (e.g. event_name).
            # We only override O/partial time_value/date_value labelings.
            if tag_p1 in ("O", "B-time_value", "I-time_value", "B-date_value", "I-date_value") \
               and tag_i in ("O", "B-time_value", "I-time_value", "B-date_value", "I-date_value"):
                tags[i-1] = "B-time_value"
                tags[i] = "I-time_value"
            continue

        # Rule 2: '<demain|hier|aujourd'hui|<weekday>> <part>' — 2 separate spans.
        # Fires only when pronoun is labeled as a date_value; leaves other
        # structures (scene_name, event_name, etc) alone.
        if prev1 in _TIME_DATE_PRONOUNS and tag_p1 in ("B-date_value", "I-date_value"):
            # Force date span to end at prev1, and start a fresh time_value at i.
            tags[i] = "B-time_value"
            continue

        # Rule 4: 'à midi/minuit' — à is outside.
        if prev1 == "à" and w in {"midi", "minuit"}:
            # Only override if existing labels are within our reach (don't
            # break a larger span that happens to include 'à').
            if tag_p1 in ("O", "B-time_value", "I-time_value") \
               and tag_i in ("O", "B-time_value", "I-time_value"):
                tags[i-1] = "O"
                tags[i] = "B-time_value"
            continue

    # Rule 5: '(à) <num> heures [du <part>]' boundary normalization.
    # Audit shows gold disagrees on whether '<num>' or 'heures' or 'à' belongs
    # in the span. Majority convention: numeral + heures (both in), à excluded.
    # We only normalize when the phrase is already partially marked time_value
    # — we don't hallucinate new spans where gold says all O.
    for i, w in enumerate(lw):
        if w != "heures":
            continue
        prev1 = lw[i-1] if i >= 1 else ""
        if prev1 not in _FR_NUMERALS:
            continue
        # Is there any time_value tag anywhere in positions i-1 or i?
        partial = tags[i-1] in ("B-time_value", "I-time_value") \
                  or tags[i] in ("B-time_value", "I-time_value")
        if not partial:
            continue
        # Normalize: force numeral=B-time_value, heures=I-time_value.
        # Check position i-2: if 'à', push it out.
        if i >= 2 and lw[i-2] == "à" and tags[i-2] in ("O", "B-time_value", "I-time_value"):
            tags[i-2] = "O"
        tags[i-1] = "B-time_value"
        tags[i] = "I-time_value"

    # Rule 6: multi-day duration tokens default to date_value (gold convention).
    # '<quantifier> <period>' or '<period>' standalone when time_value predicted.
    # Find each time_value span that contains a period token and is NOT anchored
    # by a demonstrative determiner, then retag the whole span as date_value.
    i = 0
    while i < len(lw):
        if tags[i] != "B-time_value":
            i += 1
            continue
        # Walk to span end
        j = i + 1
        while j < len(lw) and tags[j] == "I-time_value":
            j += 1
        # Does this span contain a period token?
        span_has_period = any(lw[k] in _PERIOD_TOKENS for k in range(i, j))
        if not span_has_period:
            i = j
            continue
        # Does the span START with a demonstrative-like determiner?
        # Examples to PRESERVE as time_value: 'cette semaine', 'le week-end',
        # 'la semaine', 'toute la semaine'. Check the first 1-2 tokens.
        first = lw[i]
        second = lw[i + 1] if i + 1 < j else ""
        starts_with_det = first in {"ce", "cette", "le", "la", "les"} \
                          or (first in {"toute", "tout"} and second in {"la", "le", "les"})
        if starts_with_det:
            i = j
            continue
        # Demote the whole span to date_value
        tags[i] = "B-date_value"
        for k in range(i + 1, j):
            tags[k] = "I-date_value"
        i = j

    # Rule 7: timezone tokens demoted out of time_value (semantic: they're
    # locations, not times). Audit Pattern F. Convert B/I-time_value → O.
    for i, w in enumerate(lw):
        if w not in _TIMEZONE_TOKENS:
            continue
        if tags[i] in ("B-time_value", "I-time_value"):
            tags[i] = "O"

    return tags


def write_split(examples, split_dir: Path):
    """Write the 4 files for a split. Applies BIO normalization on the way out."""
    split_dir.mkdir(parents=True, exist_ok=True)

    time_normalized = 0
    rc_normalized = 0
    pn_normalized = 0
    with open(split_dir / "seq.in", "w", encoding="utf-8") as f_in, \
         open(split_dir / "seq.out", "w", encoding="utf-8") as f_out, \
         open(split_dir / "speech_act", "w", encoding="utf-8") as f_sa, \
         open(split_dir / "domain", "w", encoding="utf-8") as f_dom:

        for ex in examples:
            words = ex["text"].split()
            original = ex["bio_tags"]
            after_time = normalize_time_bio(words, original)
            if after_time != original:
                time_normalized += 1
            after_rc = normalize_reminder_content_bio(words, after_time)
            if after_rc != after_time:
                rc_normalized += 1
            after_pn = normalize_person_name_bio(words, after_rc)
            if after_pn != after_rc:
                pn_normalized += 1
            f_in.write(ex["text"] + "\n")
            f_out.write(" ".join(after_pn) + "\n")
            f_sa.write(ex["speech_act"] + "\n")
            f_dom.write(ex["domain"] + "\n")

    if time_normalized:
        print(f"  [{split_dir.name}] normalized {time_normalized} time-BIO sequences")
    if rc_normalized:
        print(f"  [{split_dir.name}] normalized {rc_normalized} reminder_content-BIO sequences")
    if pn_normalized:
        print(f"  [{split_dir.name}] normalized {pn_normalized} person_name-BIO sequences")


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
    classify_annotated_dir = DATA_DIR / "classify_results_annotated"
    if classify_results_dir.exists():
        # Pre-load BIO annotations keyed by text.
        # Missing keys or length mismatches fall back to all-O (not IGNORE),
        # so the slot head still trains on O-transition statistics the CRF needs.
        annotations_by_text: dict[str, list[str]] = {}
        if classify_annotated_dir.exists():
            for af in sorted(classify_annotated_dir.glob("*.json")):
                with open(af, encoding="utf-8") as fh:
                    for a in json.load(fh):
                        annotations_by_text[a["text"]] = a["bio_tags"]

        dialogue_all = []
        seen = set()
        skipped_long = 0
        skipped_stmt_q = 0
        annotated_hits = 0
        annotated_misses = 0
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
                bio_tags = annotations_by_text.get(ex["text"])
                if bio_tags is None or len(bio_tags) != len(words):
                    bio_tags = ["O"] * len(words)
                    annotated_misses += 1
                else:
                    annotated_hits += 1
                # Social all-O entries poison the slot head with O-dominance:
                # 82% of classify_results are all-O, 83% of those are Social.
                # Mask them out of slot loss (IGNORE→-100 in _align_slot_labels)
                # while keeping the SA+Domain signal intact.
                if dom == "Social" and all(t == "O" for t in bio_tags):
                    bio_tags = ["IGNORE"] * len(words)
                dialogue_all.append({
                    "text": ex["text"],
                    "bio_tags": bio_tags,
                    "speech_act": sa,
                    "domain": dom,
                })
        print(f"  classify_results BIO annotations: {annotated_hits} hits, "
              f"{annotated_misses} fell back to all-O")

        print(f"  Sonnet classified: {len(dialogue_all)} unique "
              f"(skipped {skipped_long} long + {skipped_stmt_q} Statement-with-?)")

        # Fix 3: Cap Sonnet-Social to reduce domain bias.
        # Prefer entity-bearing entries when truncating — dropping them loses the
        # very annotations we just produced, and the dialogues_train YAML duplicates
        # (all-O) would then win the conflict resolver by default.
        sonnet_cap = CONFIG.get("sonnet_social_cap", 2000)
        sonnet_social = [ex for ex in dialogue_all if ex["domain"] == "Social"]
        sonnet_other = [ex for ex in dialogue_all if ex["domain"] != "Social"]
        original_social = len(sonnet_social)
        if len(sonnet_social) > sonnet_cap:
            with_ent = [ex for ex in sonnet_social
                        if any(t != "O" for t in ex["bio_tags"])]
            without_ent = [ex for ex in sonnet_social
                           if all(t == "O" for t in ex["bio_tags"])]
            random.shuffle(with_ent)
            random.shuffle(without_ent)
            kept = with_ent[:sonnet_cap]
            if len(kept) < sonnet_cap:
                kept.extend(without_ent[:sonnet_cap - len(kept)])
            sonnet_social = kept
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
            unique_n = len({ex["text"].lower().strip() for ex in examples})
            if n > max_per:
                # Downsample
                random.shuffle(examples)
                balanced_train.extend(examples[:max_per])
            elif n < min_per:
                # Require real lexical diversity before upsampling —
                # repeating a single sentence 40× teaches memorization,
                # not generalization. Combos below the diversity floor
                # are dropped so the model doesn't learn a spurious rule.
                if unique_n < 5:
                    print(f"    skipping {combo}: only {unique_n} unique "
                          f"text(s) in {n} examples — insufficient diversity")
                    continue
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
