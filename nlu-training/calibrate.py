"""
Temperature scaling calibration for speech act and domain heads.

Learns a scalar T per head on the dev set to minimize NLL.
calibrated_logits = logits / T
"""

import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from labels import SPEECH_ACTS, DOMAINS, SLOT_LABELS
from model import JointCamemBERTav2
from train import NluDataset, set_seed

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter for a classification head."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def collect_logits(model, dataset, device, batch_size=64):
    """Run batched inference, collect raw logits for speech act and domain heads."""
    model.eval()
    sa_logits_list, dom_logits_list = [], []
    n = len(dataset)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            input_ids = dataset.input_ids[start:end].to(device)
            attention_mask = dataset.attention_mask[start:end].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sa_logits_list.append(outputs["speech_act_logits"].cpu())
            dom_logits_list.append(outputs["domain_logits"].cpu())

    sa_logits = torch.cat(sa_logits_list, dim=0)
    sa_labels = dataset.speech_act_labels.clone()
    dom_logits = torch.cat(dom_logits_list, dim=0)
    dom_labels = dataset.domain_labels.clone()

    return sa_logits, sa_labels, dom_logits, dom_labels


def optimize_temperature(logits, labels, name, lr=0.01, max_iter=200):
    """Optimize temperature to minimize NLL on the given logits/labels."""
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = scaler.temperature.item()

    # ECE before/after
    ece_before = compute_ece(logits, labels)
    ece_after = compute_ece(logits / T, labels)

    print(f"  {name}: T={T:.4f}, ECE before={ece_before:.4f}, ECE after={ece_after:.4f}")
    return T


def compute_ece(logits, labels, n_bins=15):
    """Expected Calibration Error."""
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.float().mean() * abs(bin_acc - bin_conf)
    return ece.item()


def main():
    set_seed(CONFIG["seed"])

    data_dir = Path(CONFIG["paths"]["data_dir"])
    output_dir = Path(CONFIG["paths"]["output_dir"])
    model_name = CONFIG["model"]["name"]
    max_length = CONFIG["model"]["max_seq_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    best_dir = output_dir / "best_model"
    model = JointCamemBERTav2(model_name=model_name)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(best_dir / "model.safetensors", device=str(device)))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))

    # Load dev set (calibration set)
    dev_dataset = NluDataset(data_dir / "dev", tokenizer, max_length)
    print(f"Dev set: {len(dev_dataset)} examples")

    # Collect logits
    print("Collecting logits...")
    sa_logits, sa_labels, dom_logits, dom_labels = collect_logits(model, dev_dataset, device)

    # Optimize temperatures
    print("\nOptimizing temperatures...")
    T_sa = optimize_temperature(sa_logits, sa_labels, "Speech Act")
    T_dom = optimize_temperature(dom_logits, dom_labels, "Domain")

    # Save
    temps = {
        "speech_act_temperature": T_sa,
        "domain_temperature": T_dom,
    }
    out_path = output_dir / "temperatures.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(temps, f, indent=2)
    print(f"\nTemperatures saved to {out_path}")


if __name__ == "__main__":
    main()
