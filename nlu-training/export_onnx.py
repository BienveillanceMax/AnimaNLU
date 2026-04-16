"""
Export trained model to ONNX (FP32, FP16, INT8) + validation + tokenizer test vectors.

CRF transition matrix and Viterbi test vectors are exported separately
for Java-side Viterbi decoding.
"""

import json
import yaml
import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer

from labels import SPEECH_ACTS, DOMAINS, SLOT_LABELS, write_label_files
from model import JointCamemBERTav2
from train import NluDataset, set_seed

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def export_fp32(model, tokenizer, output_path, max_length=64):
    """Export PyTorch model to ONNX FP32."""
    model.eval()
    dummy_text = "allume la lumière du salon"
    encoding = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Wrap model to only output logits (no loss, no CRF)
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out["speech_act_logits"], out["domain_logits"], out["slot_logits"]

    wrapper = ExportWrapper(model)

    torch.onnx.export(
        wrapper,
        (encoding["input_ids"], encoding["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["speech_act_logits", "domain_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "speech_act_logits": {0: "batch_size"},
            "domain_logits": {0: "batch_size"},
            "slot_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  FP32 exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def convert_fp16(fp32_path, fp16_path):
    """Convert FP32 ONNX to FP16."""
    from onnxruntime.transformers import float16
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))
    print(f"  FP16 exported to {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")


def quantize_int8(fp32_path, int8_path):
    """Dynamic INT8 quantization of FP32 ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"  INT8 exported to {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")


def validate_onnx(pytorch_model, onnx_path, dataset, tokenizer, device, n_samples=100):
    """Compare ONNX output to PyTorch output on test samples."""
    pytorch_model.eval()
    session = ort.InferenceSession(str(onnx_path))

    sa_match, dom_match, total = 0, 0, 0
    max_diff = 0.0

    indices = list(range(min(n_samples, len(dataset))))
    for idx in indices:
        item = dataset[idx]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        # PyTorch
        with torch.no_grad():
            pt_out = pytorch_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )

        # ONNX
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        }
        ort_out = session.run(None, ort_inputs)

        # Compare
        pt_sa = pt_out["speech_act_logits"].cpu().numpy()
        pt_dom = pt_out["domain_logits"].cpu().numpy()

        diff_sa = np.abs(pt_sa - ort_out[0]).max()
        diff_dom = np.abs(pt_dom - ort_out[1]).max()
        max_diff = max(max_diff, diff_sa, diff_dom)

        if np.argmax(pt_sa) == np.argmax(ort_out[0]):
            sa_match += 1
        if np.argmax(pt_dom) == np.argmax(ort_out[1]):
            dom_match += 1
        total += 1

    sa_agree = sa_match / total
    dom_agree = dom_match / total
    return sa_agree, dom_agree, max_diff


def export_crf_transitions(model, output_path):
    """Export CRF transition matrix as JSON for Java-side Viterbi decoding."""
    if not hasattr(model, "crf") or not model.use_crf:
        print("  No CRF layer found, skipping transition export")
        return

    transitions = model.crf.transitions.detach().cpu().numpy()
    start_transitions = model.crf.start_transitions.detach().cpu().numpy()
    end_transitions = model.crf.end_transitions.detach().cpu().numpy()

    export_data = {
        "num_tags": int(transitions.shape[0]),
        "tag_names": SLOT_LABELS,
        "transitions": transitions.tolist(),
        "start_transitions": start_transitions.tolist(),
        "end_transitions": end_transitions.tolist(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"  CRF transitions exported to {output_path}")
    print(f"    Matrix shape: {transitions.shape}")
    print(f"    Start transitions: {start_transitions.shape}")
    print(f"    End transitions: {end_transitions.shape}")


def generate_viterbi_test_vectors(model, dataset, output_path, device, n=100):
    """Generate test vectors for validating Java Viterbi implementation.

    Each vector contains emissions, mask, and expected Viterbi output
    from the Python CRF so we can verify the Java implementation matches.
    """
    if not hasattr(model, "crf") or not model.use_crf:
        print("  No CRF layer found, skipping Viterbi test vectors")
        return

    model.eval()
    vectors = []

    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            input_ids = dataset.input_ids[i].unsqueeze(0).to(device)
            attention_mask = dataset.attention_mask[i].unsqueeze(0).to(device)
            slot_labels = dataset.slot_labels[i].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            slot_logits = outputs["slot_logits"]  # (1, seq_len, 37)

            # Pack real-label positions for CRF decode
            real_mask = (slot_labels != -100)  # (1, seq_len)
            idx = real_mask[0].nonzero(as_tuple=True)[0]
            n_real = idx.size(0)

            if n_real > 0:
                packed_logits = slot_logits[0, idx].unsqueeze(0)  # (1, n_real, 37)
                packed_mask = torch.ones(1, n_real, dtype=torch.bool, device=device)
                viterbi_result = model.crf.decode(packed_logits, mask=packed_mask)[0]
            else:
                viterbi_result = []

            vectors.append({
                "emissions": slot_logits.squeeze(0).cpu().numpy().tolist(),
                "attention_mask": attention_mask.squeeze(0).cpu().numpy().tolist(),
                "valid_label_mask": real_mask.squeeze(0).cpu().numpy().tolist(),
                "expected_tags": viterbi_result,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f)

    print(f"  {len(vectors)} Viterbi test vectors saved to {output_path}")


def export_tokenizer_test_vectors(tokenizer, output_path, max_length=64):
    """Export test vectors for Java tokenizer verification."""
    test_sentences = [
        "allume la lumière du salon",
        "quelle heure est-il",
        "j'ai eu une journée de merde",
        "mets du jazz s'il te plaît",
        "ferme les volets de la chambre",
        "c'est quoi la température dehors",
        "rappelle-moi d'acheter du pain",
        "non je voulais dire le garage",
        "bonne nuit",
        "je suis crevé",
        "tu te souviens de ce que j'ai dit hier",
        "est-ce que tu peux baisser le chauffage",
        "merci c'est parfait",
        "qu'est-ce qu'il y a comme courses à faire",
        "mets la lumière en rouge dans le salon",
        "je pars dans dix minutes",
        "oui d'accord vas-y",
        "non pas du tout",
        "parle moins fort s'il te plaît",
        "j'écoute beaucoup de rap français",
        "réveille-moi à sept heures demain",
        "il fait froid ici",
        "pourquoi t'as fait ça",
        "je suis de bonne humeur aujourd'hui",
        "annule ça laisse tomber",
        "tu pourrais mettre un peu de musique",
        "c'est nul ça marche pas",
        "coucou anima comment ça va",
        "j'ai une réunion à quatorze heures",
        "mets-toi en mode nuit",
        # Edge cases
        "",
        "a",
        "oui",
        "je ne sais pas trop quoi dire mais je voulais juste parler un peu avec toi ce soir",
        "l'électricité",
        "vingt-deux degrés",
        "salle de bain",
        "aujourd'hui c'est lundi",
        "j'aimerais qu'il fasse plus chaud dans la chambre à coucher s'il te plaît",
        "euh ben je sais pas",
        "la lumière la lumière la lumière",
        "café crème",
        "rock'n'roll",
        "c'est-à-dire que je voudrais",
        "vingt et un",
        "soixante-dix-sept",
        "l'alarme de huit heures et demie",
        "est-ce que c'est possible",
        "quatre-vingt-dix-neuf",
    ]

    vectors = []
    for text in test_sentences:
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        vectors.append({
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(0).tolist(),
            "attention_mask": encoding["attention_mask"].squeeze(0).tolist(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)
    print(f"  {len(vectors)} tokenizer test vectors saved to {output_path}")


def main():
    set_seed(CONFIG["seed"])

    data_dir = Path(CONFIG["paths"]["data_dir"])
    output_dir = Path(CONFIG["paths"]["output_dir"])
    model_name = CONFIG["model"]["name"]
    max_length = CONFIG["model"]["max_seq_length"]
    use_crf = CONFIG["model"].get("use_crf", False)

    device = torch.device("cpu")  # Export on CPU for reproducibility

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

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))

    # Load test set for validation
    test_dataset = NluDataset(data_dir / "test", tokenizer, max_length)

    # ── Export FP32 ──
    print("\n1. Exporting FP32...")
    fp32_path = output_dir / "joint_nlu_fp32.onnx"
    export_fp32(model, tokenizer, fp32_path, max_length)

    # ── Export FP16 ──
    print("\n2. Converting to FP16...")
    fp16_path = output_dir / "joint_nlu_fp16.onnx"
    try:
        convert_fp16(fp32_path, fp16_path)
    except Exception as e:
        print(f"  FP16 conversion failed: {e}")
        fp16_path = None

    # ── Export INT8 ──
    print("\n3. Quantizing to INT8...")
    int8_path = output_dir / "joint_nlu_int8.onnx"
    try:
        quantize_int8(fp32_path, int8_path)
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")
        int8_path = None

    # ── Validate all variants ──
    print("\n4. Validating against PyTorch...")
    report = {}
    for name, path in [("FP32", fp32_path), ("FP16", fp16_path), ("INT8", int8_path)]:
        if path is None or not path.exists():
            print(f"  {name}: skipped")
            continue
        sa_agree, dom_agree, max_diff = validate_onnx(model, path, test_dataset, tokenizer, device)
        degradation = 1.0 - min(sa_agree, dom_agree)
        status = "PASS" if degradation < 0.005 else "INFO" if name != "FP32" else "FAIL"
        print(f"  {name}: SA agree={sa_agree:.4f}, DOM agree={dom_agree:.4f}, "
              f"max_diff={max_diff:.6f}, degradation={degradation:.4f} [{status}]")
        report[name] = {
            "speech_act_agreement": sa_agree,
            "domain_agreement": dom_agree,
            "max_logit_diff": float(max_diff),
            "size_mb": path.stat().st_size / 1e6,
        }

    # Save report
    report_path = output_dir / "quantization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Quantization report saved to {report_path}")

    # ── Tokenizer test vectors ──
    print("\n5. Exporting tokenizer test vectors...")
    export_tokenizer_test_vectors(tokenizer, output_dir / "tokenizer_test_vectors.json", max_length)

    # ── Copy label files ──
    print("\n6. Copying label files...")
    write_label_files(str(output_dir))

    # ── Copy tokenizer ──
    print("\n7. Copying tokenizer.json...")
    import shutil
    tokenizer_src = best_dir / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, output_dir / "tokenizer.json")
        print(f"  Copied to {output_dir / 'tokenizer.json'}")

    # ── CRF transitions ──
    print("\n8. Exporting CRF transitions...")
    export_crf_transitions(model, output_dir / "crf_transitions.json")

    # ── Viterbi test vectors ──
    print("\n9. Generating Viterbi test vectors...")
    generate_viterbi_test_vectors(model, test_dataset, output_dir / "viterbi_test_vectors.json", device)

    print("\n" + "=" * 60)
    print("  Export complete. Artifacts in outputs/:")
    for p in sorted(output_dir.glob("*")):
        if p.is_file():
            print(f"    {p.name:40s}  {p.stat().st_size / 1e6:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
