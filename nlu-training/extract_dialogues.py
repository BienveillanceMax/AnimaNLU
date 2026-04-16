"""
Extract and classify utterances from FrancophonIA/french_dialogues_series
to supplement our synthetic combos with real human dialogue.

Outputs:
  - data/dialogues_train/*.yaml  (training supplements)
  - data/dialogues_eval/*.yaml   (eval set for synthetic combos — NEVER in train)
"""

import re
import random
import yaml
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

random.seed(42)

OUT_TRAIN = Path("data/dialogues_train")
OUT_EVAL = Path("data/dialogues_eval")
OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_EVAL.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Keyword filters per target combo
# ──────────────────────────────────────────────
# Each filter: (speech_act, domain, patterns, anti_patterns)
# patterns: at least one must match (case-insensitive)
# anti_patterns: none must match
FILTERS = [
    ("ExpressEmotion", "Social", [
        r"^j'en ai marre$", r"^j'en peux plus$",
        r"^je suis (tellement |trop |vraiment )?(triste|content|fatigué|crevé|énervé|furieux|dégoûté|épuisé|stressé|déprimé|heureux|soulagé|anxieux|déçu|frustré|fier|inquiet)(\s|$|,|!)",
        r"^ça (me soûle|m'énerve|me rend (triste|fou|dingue|heureux|malade))(\s|$)",
        r"^j'ai (le cafard|le blues|la trouille|la pêche|honte|envie de pleurer)(\s|$)",
        r"^ça va (pas|mal|très mal|pas du tout)(\s|$|,)",
        r"^je me sens (seul|mal|bien|nul|perdu|triste|vide)(\s|$|,)",
        r"^je suis (au bout du rouleau|à bout|à cran|sur les nerfs|aux anges)",
    ], [r"^je suis (le|la|un|une|en train|sûr|certain)", r"\?$",
        r"(que|qui|qu'|dont|où|parce|car|si )", r"^je suis content.{15,}"]),
        # Exclude complex subordinate clauses — we want short emotional bursts

    ("Statement", "Social", [
        r"^j'aime (bien )?le ", r"^j'aime (bien )?la ", r"^j'aime (bien )?les ",
        r"^j'aime pas ", r"^je déteste les? ", r"^j'adore les? ", r"^je préfère le ",
        r"^j'ai (faim|soif|froid|chaud|sommeil)$",
        r"^il fait (froid|chaud|beau)$", r"^il fait (froid|chaud|beau) ",
        r"^il (pleut|neige)(\s|$)",
    ], [r"\?$", r"^j'aime(rais|rai)", r"(que |qui |qu'|parce|car )"]),
        # Tight: only simple declarative preferences/observations

    ("Confirm", "Social", [
        r"^oui$", r"^ouais$", r"^d'accord$", r"^ok$", r"^exactement$",
        r"^absolument$", r"^tout à fait$", r"^bien sûr$", r"^c'est ça$",
        r"^oui,? (c'est ça|exactement|tout à fait|bien sûr|d'accord)$",
        r"^oui,? (oui|merci|je sais|je comprends)",
    ], [r"\?$", r".{35,}"]),

    ("Deny", "Social", [
        r"^non$", r"^nan$", r"^pas du tout$", r"^absolument pas$",
        r"^hors de question$", r"^certainement pas$",
        r"^non,? (non|merci|jamais|pas question|c'est faux|c'est pas vrai|impossible)(\s|$|!)",
    ], [r"\?$", r".{40,}"]),

    ("Farewell", "Social", [
        r"^au revoir$", r"^au revoir[,!. ]", r"^bonne nuit$", r"^bonne nuit[,!. ]",
        r"^à demain$", r"^à plus$", r"^à bientôt$", r"^adieu$", r"^ciao$",
        r"^bonne soirée$", r"^bonne soirée[,!. ]",
        r"^à (tout à l'heure|la prochaine)$",
    ], [r".{30,}"]),
        # Very tight — only clear farewell formulas, no "je te laisse" (too ambiguous)

    ("Greeting", "Social", [
        r"^bonjour$", r"^bonjour[,!. ]", r"^bonsoir$", r"^bonsoir[,!. ]",
        r"^salut$", r"^salut[,!. ]", r"^coucou$", r"^hello$", r"^hey$",
    ], [r".{20,}"]),
        # Very short only — exclude greetings followed by character names

    ("PositiveFeedback", "Social", [
        r"^merci$", r"^merci[,!. ]", r"^super$", r"^parfait$", r"^génial$",
        r"^excellent$", r"^bravo$", r"^bravo[,!. ]",
        r"^c'est (super|génial|parfait|formidable|magnifique|incroyable)$",
        r"^c'est (super|génial|parfait|formidable),",
        r"^(bien joué|chapeau|nickel|impeccable)(\s|$|!)",
    ], [r"\?$", r".{35,}"]),

    ("NegativeFeedback", "Social", [
        r"^c'est (nul|pourri|ridicule|lamentable|n'importe quoi)(\s|$|!|,)",
        r"^(ça craint|n'importe quoi|c'est de la merde)(\s|$|!)",
        r"^(t'es nul|c'est pas terrible|ça marche pas)(\s|$|!)",
    ], [r"\?$", r".{35,}"]),

    ("Cancel", "System", [
        r"^laisse tomber$", r"^laisse tomber[,!. ]",
        r"^tant pis$", r"^tant pis[,!. ]",
        r"^oublie([ -]|$)", r"^annule",
        r"^j'ai changé d'avis",
        r"^(en fait|finalement),? non$",
        r"^pas la peine$",
    ], [r"\?$", r".{30,}"]),
        # No "rien" or "arrête" — too many false positives

    ("Question", "Meta", [
        r"^pourquoi (tu|t'as) (fait|dit|répondu|choisi|menti) ",
        r"^comment tu (sais|fais|fonctionnes)",
        r"^tu te (souviens|rappelles) ",
        r"^tu (m'écoutes|me comprends|mens|m'aimes)\s",
        r"^t'es (qui|quoi) ",
        r"^c'est quoi ton (nom|problème|but)",
    ], [r".{40,}"]),
        # Tight — only genuinely meta-conversational patterns
]


def matches(text, patterns, anti_patterns):
    """Check if text matches any pattern and no anti-pattern."""
    text_lower = text.lower().strip()
    if not any(re.search(p, text_lower) for p in patterns):
        return False
    if any(re.search(p, text_lower) for p in anti_patterns):
        return False
    return True


def clean_utterance(text):
    """Clean TV dialogue text for NLU."""
    text = text.strip()
    # Remove stage directions in brackets
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Remove trailing/leading punctuation excess
    text = text.strip(" .!…")
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading FrancophonIA/french_dialogues_series...")
    ds = load_dataset("FrancophonIA/french_dialogues_series")["train"]

    # Filter to actual dialogue with speakers
    dialogue = [ex for ex in ds if ex["speaker"] is not None and ex["speaker"] != "None"]
    print(f"Dialogue utterances: {len(dialogue)}")

    # Apply filters
    candidates = defaultdict(list)
    for ex in dialogue:
        text = clean_utterance(ex["utterance"])
        if not text or len(text.split()) < 1 or len(text.split()) > 15:
            continue
        for sa, dom, patterns, anti_patterns in FILTERS:
            if matches(text, patterns, anti_patterns):
                candidates[(sa, dom)].append(text)
                break  # First match wins

    print(f"\n=== Candidates per combo ===")
    for (sa, dom), texts in sorted(candidates.items()):
        unique = list(set(t.lower() for t in texts))
        print(f"  {sa:20s} × {dom:10s}: {len(texts)} total, {len(unique)} unique")

    # Sample and deduplicate
    TRAIN_PER_COMBO = 500
    EVAL_PER_COMBO = 100

    for (sa, dom), texts in sorted(candidates.items()):
        # Deduplicate (case-insensitive)
        seen = set()
        unique = []
        for t in texts:
            key = t.lower().strip()
            if key not in seen and len(key) > 1:
                seen.add(key)
                unique.append(t)

        random.shuffle(unique)

        # Split: first EVAL_PER_COMBO for eval, rest for train
        eval_texts = unique[:EVAL_PER_COMBO]
        train_texts = unique[EVAL_PER_COMBO:EVAL_PER_COMBO + TRAIN_PER_COMBO]

        combo_name = f"{sa.lower()}_{dom.lower()}"

        for out_dir, texts_list, label in [(OUT_EVAL, eval_texts, "eval"), (OUT_TRAIN, train_texts, "train")]:
            if not texts_list:
                continue
            examples = []
            for t in texts_list:
                words = t.split()
                examples.append({"text": t, "bio_tags": ["O"] * len(words)})

            data = {"speech_act": sa, "domain": dom, "examples": examples}
            path = out_dir / f"{combo_name}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            print(f"  {label:5s} {path.name:40s}: {len(examples)} examples")

    print("\nDone. Review the extracted examples for quality.")
    print(f"  Train supplements: {OUT_TRAIN}/")
    print(f"  Eval set:          {OUT_EVAL}/")


if __name__ == "__main__":
    main()
