"""
Augment supplement YAML files:
1. Add comma variants of existing examples
2. Add filler/interjection variants (euh, ben, bah, enfin)
3. Add politeness variants (s'il te plaît)

Run once, then inspect results. Does NOT duplicate existing texts.
"""

import yaml
import re
from pathlib import Path
from copy import deepcopy

SUPP_DIR = Path("data/supplements")

# Comma insertion rules (pattern → replacement)
COMMA_RULES = [
    # "non X" → "non, X" (after negation opener)
    (r"^(non) ", r"\1, "),
    # "non pas X Y" → "non, pas X, Y" — handled by the above + below
    # "oui X" → "oui, X"
    (r"^(oui) ", r"\1, "),
    # "ouais X" → "ouais, X"
    (r"^(ouais) ", r"\1, "),
    # "bah X" → "bah, X"
    (r"^(bah|ben|bon|bof|euh) ", r"\1, "),
    # "allez X" → "allez, X"
    (r"^(allez) ", r"\1, "),
    # Before "s'il te plaît" at the end
    (r" (s'il te plaît)$", r", \1"),
    # "finalement X" → "finalement, X"
    (r"^(finalement|franchement|sincèrement|honnêtement) ", r"\1, "),
    # "en fait X" → "en fait, X"
    (r"^(en fait) ", r"\1, "),
]

# Filler prefixes to add
FILLER_PREFIXES = [
    ("euh ", ["O"]),
    ("ben ", ["O"]),
    ("bah ", ["O"]),
    ("bon ", ["O"]),
    ("enfin ", ["O"]),
    ("ah ", ["O"]),
    ("oh ", ["O"]),
]

# Politeness suffixes
POLITE_SUFFIXES = [
    (" s'il te plaît", ["O", "O", "O", "O"]),
    (" steuplé", ["O"]),
]


def apply_comma_rules(text):
    """Try each comma rule, return variant if any applied."""
    for pattern, repl in COMMA_RULES:
        new_text = re.sub(pattern, repl, text)
        if new_text != text:
            return new_text
    return None


def make_variants(text, bio_tags):
    """Generate variants of a (text, bio_tags) pair."""
    variants = []
    words = text.split()

    # 1. Comma variants
    comma_text = apply_comma_rules(text)
    if comma_text:
        # Commas don't change word count when split by space IF comma is attached
        # But "non," is still one token. Recount.
        comma_words = comma_text.split()
        if len(comma_words) == len(words):
            # Same word count — comma just attached to a word
            variants.append((comma_text, bio_tags[:]))
        else:
            # Comma became a separate token — adjust tags
            # Find where new tokens were inserted and add O tags
            new_tags = []
            j = 0
            for cw in comma_words:
                # Strip commas to match original
                clean = cw.rstrip(",").lstrip(",")
                if j < len(bio_tags) and clean:
                    new_tags.append(bio_tags[j])
                    # Only advance j if this was an actual word from original
                    if clean == words[j].rstrip(",") if j < len(words) else False:
                        j += 1
                    elif cw == ",":
                        new_tags[-1] = "O"  # standalone comma
                else:
                    new_tags.append("O")
            # Only use if tag count matches
            if len(new_tags) == len(comma_words):
                variants.append((comma_text, new_tags))

    # 2. Filler prefix (only for longer utterances, skip 1-word ones)
    if len(words) >= 3:
        for prefix, prefix_tags in FILLER_PREFIXES[:3]:  # Only first 3 fillers
            new_text = prefix + text
            new_tags = prefix_tags + bio_tags[:]
            variants.append((new_text, new_tags))
            break  # Only one filler variant per example

    # 3. Politeness suffix (only for commands/requests, skip short utterances)
    if len(words) >= 3 and not any(text.startswith(w) for w in ["merci", "oui", "non", "je suis", "j'ai", "ça"]):
        for suffix, suffix_tags in POLITE_SUFFIXES[:1]:
            new_text = text + suffix
            new_tags = bio_tags[:] + suffix_tags
            variants.append((new_text, new_tags))
            break

    return variants


def augment_file(yaml_path):
    """Augment a single YAML file with variants."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    existing_texts = {ex["text"].lower() for ex in data["examples"]}
    new_examples = []

    for ex in data["examples"]:
        for var_text, var_tags in make_variants(ex["text"], ex["bio_tags"]):
            if var_text.lower() not in existing_texts:
                # Validate tag count
                if len(var_text.split()) == len(var_tags):
                    new_examples.append({"text": var_text, "bio_tags": var_tags})
                    existing_texts.add(var_text.lower())

    data["examples"].extend(new_examples)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return len(new_examples)


def main():
    total_added = 0
    for f in sorted(SUPP_DIR.glob("*.yaml")):
        added = augment_file(f)
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        print(f"  {f.name:30s}  +{added:3d} variants → {len(data['examples']):3d} total")
        total_added += added

    print(f"\nTotal added: {total_added}")

    # Validate
    errors = 0
    total = 0
    for f in sorted(SUPP_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        total += len(data["examples"])
        for ex in data["examples"]:
            if len(ex["text"].split()) != len(ex["bio_tags"]):
                errors += 1
    print(f"Total examples: {total}, Errors: {errors}")


if __name__ == "__main__":
    main()
