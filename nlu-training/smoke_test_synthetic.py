"""
Smoke test for synthetic-only combos (no eval data from MASSIVE).

Hand-written examples — not recycled from training supplements.
Tests that the model at least classifies these combos correctly.
This is a sanity check, not a statistical evaluation.
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer

from labels import SPEECH_ACT_I2L, DOMAIN_I2L, SLOT_LABEL_I2L, SPEECH_ACTS, DOMAINS
from model import JointCamemBERTav2
from train import set_seed

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# ──────────────────────────────────────────────
# Hand-written test cases — NOT from supplements
# ──────────────────────────────────────────────
TEST_CASES = [
    # ExpressEmotion × Social
    ("j'ai vraiment le blues ce soir", "ExpressEmotion", "Social"),
    ("je suis tellement heureux là", "ExpressEmotion", "Social"),
    ("ça fait deux nuits que je dors mal", "ExpressEmotion", "Social"),
    ("je suis trop stressé avec les examens", "ExpressEmotion", "Social"),
    ("j'ai la patate aujourd'hui", "ExpressEmotion", "Social"),

    # Request × Home
    ("tu pourrais baisser la lumière dans le couloir", "Request", "Home"),
    ("est-ce que c'est possible de fermer les stores", "Request", "Home"),
    ("j'aimerais bien que tu montes le chauffage", "Request", "Home"),
    ("ce serait cool de tamiser un peu", "Request", "Home"),
    ("tu veux bien ouvrir la fenêtre du bureau", "Request", "Home"),

    # Request × Media
    ("j'aimerais bien écouter quelque chose de doux", "Request", "Media"),
    ("tu pourrais me mettre un podcast sur l'histoire", "Request", "Media"),
    ("est-ce qu'on pourrait avoir un peu de musique", "Request", "Media"),

    # Correction × Home
    ("non c'est pas le salon la cuisine", "Correction", "Home"),
    ("je parlais du garage pas du bureau", "Correction", "Home"),
    ("non l'autre pièce", "Correction", "Home"),

    # Correction × Media
    ("non pas ce style là plutôt du classique", "Correction", "Media"),
    ("je voulais la radio pas spotify", "Correction", "Media"),

    # Confirm × Social
    ("ouais c'est exactement ça", "Confirm", "Social"),
    ("tout à fait c'est ce que je voulais", "Confirm", "Social"),
    ("oui parfaitement", "Confirm", "Social"),
    ("ça me va très bien", "Confirm", "Social"),

    # Deny × Social
    ("non vraiment pas", "Deny", "Social"),
    ("c'est vraiment pas ça du tout", "Deny", "Social"),
    ("absolument pas c'est faux", "Deny", "Social"),

    # Cancel × System
    ("laisse tomber j'en ai plus besoin", "Cancel", "System"),
    ("non rien oublie ce que j'ai dit", "Cancel", "System"),
    ("finalement c'est pas la peine", "Cancel", "System"),

    # Farewell × Social
    ("allez je vais me pieuter", "Farewell", "Social"),
    ("à la revoyure", "Farewell", "Social"),
    ("je te dis à plus", "Farewell", "Social"),

    # PositiveFeedback × Social
    ("ah c'est vraiment bien ça", "PositiveFeedback", "Social"),
    ("t'es vraiment au top", "PositiveFeedback", "Social"),
    ("c'est pile poil ce qu'il me fallait", "PositiveFeedback", "Social"),

    # NegativeFeedback × Social
    ("franchement c'est pas ouf", "NegativeFeedback", "Social"),
    ("t'es complètement à côté de la plaque", "NegativeFeedback", "Social"),
    ("mais c'est pas du tout ce que je voulais", "NegativeFeedback", "Social"),

    # Question × Meta
    ("est-ce que tu apprends de nos échanges", "Question", "Meta"),
    ("comment tu fais pour te souvenir de tout", "Question", "Meta"),
    ("tu sais ce que j'aime comme musique", "Question", "Meta"),

    # Command × System
    ("mets-toi en sourdine deux minutes", "Command", "System"),
    ("parle un peu plus doucement s'il te plaît", "Command", "System"),
    ("redémarre toi complètement", "Command", "System"),

    # Statement × Social
    ("mon frère arrive ce weekend", "Statement", "Social"),
    ("je suis plutôt thé que café", "Statement", "Social"),
    ("les voisins du dessus sont bruyants", "Statement", "Social"),
]


def run_inference(model, tokenizer, text, device, max_length=64):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])

    sa_probs = torch.softmax(outputs["speech_act_logits"], dim=-1).squeeze()
    dom_probs = torch.softmax(outputs["domain_logits"], dim=-1).squeeze()

    sa_pred = sa_probs.argmax().item()
    dom_pred = dom_probs.argmax().item()
    sa_conf = sa_probs[sa_pred].item()
    dom_conf = dom_probs[dom_pred].item()

    return SPEECH_ACT_I2L[sa_pred], sa_conf, DOMAIN_I2L[dom_pred], dom_conf


def main():
    set_seed(CONFIG["seed"])

    output_dir = Path(CONFIG["paths"]["output_dir"])
    model_name = CONFIG["model"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_dir = output_dir / "best_model"
    model = JointCamemBERTav2(model_name=model_name)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(best_dir / "model.safetensors", device=str(device)))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))

    print(f"{'='*80}")
    print(f"  SMOKE TEST — Synthetic combos (hand-written, not from training)")
    print(f"{'='*80}\n")

    correct_sa, correct_dom, correct_both, total = 0, 0, 0, 0
    by_combo = {}

    for text, expected_sa, expected_dom in TEST_CASES:
        pred_sa, sa_conf, pred_dom, dom_conf = run_inference(model, tokenizer, text, device)

        sa_ok = pred_sa == expected_sa
        dom_ok = pred_dom == expected_dom
        both_ok = sa_ok and dom_ok

        correct_sa += sa_ok
        correct_dom += dom_ok
        correct_both += both_ok
        total += 1

        combo = f"{expected_sa} × {expected_dom}"
        if combo not in by_combo:
            by_combo[combo] = {"total": 0, "sa_ok": 0, "dom_ok": 0, "both_ok": 0, "failures": []}
        by_combo[combo]["total"] += 1
        by_combo[combo]["sa_ok"] += sa_ok
        by_combo[combo]["dom_ok"] += dom_ok
        by_combo[combo]["both_ok"] += both_ok

        status = "OK" if both_ok else "MISS"
        if not both_ok:
            by_combo[combo]["failures"].append(
                f'    "{text}" → {pred_sa}({sa_conf:.2f}) × {pred_dom}({dom_conf:.2f})'
            )
            print(f'  {status}  "{text}"')
            print(f'       expected: {expected_sa} × {expected_dom}')
            print(f'       got:      {pred_sa}({sa_conf:.2f}) × {pred_dom}({dom_conf:.2f})')
            print()

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Speech Act: {correct_sa}/{total} ({correct_sa/total:.0%})")
    print(f"  Domain:     {correct_dom}/{total} ({correct_dom/total:.0%})")
    print(f"  Both:       {correct_both}/{total} ({correct_both/total:.0%})")

    print(f"\n  Per-combo:")
    for combo in sorted(by_combo.keys()):
        s = by_combo[combo]
        status = "PASS" if s["both_ok"] == s["total"] else "PARTIAL" if s["both_ok"] > 0 else "FAIL"
        print(f"    {combo:35s}  {s['both_ok']}/{s['total']}  [{status}]")
        for f in s["failures"]:
            print(f)

    # Verdict
    both_pct = correct_both / total
    print(f"\n{'='*80}")
    if both_pct >= 0.7:
        print(f"  VERDICT: {'PASS' if both_pct >= 0.85 else 'ACCEPTABLE'} ({both_pct:.0%})")
    else:
        print(f"  VERDICT: FAIL ({both_pct:.0%}) — synthetic combos need more training data")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
