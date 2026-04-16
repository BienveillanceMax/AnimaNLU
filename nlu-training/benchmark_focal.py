"""
Benchmark FocalLoss vs CrossEntropyLoss for slot prediction.

Tests:
1. Forward pass time with various -100 ratios
2. FP16 stability check
3. Memory usage comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        loss = focal_weight * ce_loss
        mask = targets != self.ignore_index
        return loss[mask].mean() if mask.any() else loss.mean()


def benchmark_loss(loss_fn, logits, targets, name, n_runs=100):
    """Benchmark a loss function."""
    # Warmup
    for _ in range(10):
        _ = loss_fn(logits, targets)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(n_runs):
        loss = loss_fn(logits, targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / n_runs) * 1000

    print(f"{name:30s}: {avg_time_ms:.4f} ms/iter  (loss={loss.item():.4f})")
    return avg_time_ms


def test_performance():
    """Test FocalLoss vs CE with realistic NLU slot dimensions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Realistic dimensions from NLU model
    batch_size = 64
    seq_len = 64
    num_classes = 37

    # Test different ignore ratios
    ignore_ratios = [0.5, 0.7, 0.85, 0.95]  # % of tokens that are -100

    for ignore_ratio in ignore_ratios:
        print(f"\n{'='*60}")
        print(f"Ignore ratio: {ignore_ratio:.0%} (typical for NLU with subwords + IGNORE tags)")
        print('='*60)

        # Create synthetic data
        logits = torch.randn(batch_size * seq_len, num_classes, device=device)
        targets = torch.randint(0, num_classes, (batch_size * seq_len,), device=device)

        # Set ignore_ratio of targets to -100
        n_ignore = int(batch_size * seq_len * ignore_ratio)
        ignore_indices = torch.randperm(batch_size * seq_len)[:n_ignore]
        targets[ignore_indices] = -100

        # Benchmark FP32
        print("\nFP32:")
        focal = FocalLoss(gamma=2.0)
        focal_time = benchmark_loss(focal, logits, targets, "FocalLoss")

        ce_time = benchmark_loss(
            lambda l, t: F.cross_entropy(l, t, ignore_index=-100),
            logits, targets,
            "CrossEntropyLoss"
        )

        overhead_pct = ((focal_time - ce_time) / ce_time) * 100
        print(f"\nFocalLoss overhead: {overhead_pct:+.1f}%")

        # Benchmark FP16 (if CUDA available)
        if torch.cuda.is_available():
            print("\nFP16 (mixed precision):")
            logits_fp16 = logits.half()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                focal_time_fp16 = benchmark_loss(focal, logits_fp16, targets, "FocalLoss (FP16)")
                ce_time_fp16 = benchmark_loss(
                    lambda l, t: F.cross_entropy(l, t, ignore_index=-100),
                    logits_fp16, targets,
                    "CrossEntropyLoss (FP16)"
                )

            overhead_fp16 = ((focal_time_fp16 - ce_time_fp16) / ce_time_fp16) * 100
            print(f"\nFocalLoss overhead (FP16): {overhead_fp16:+.1f}%")


def test_fp16_stability():
    """Test numerical stability of FocalLoss in FP16."""
    if not torch.cuda.is_available():
        print("\nSkipping FP16 stability test (no CUDA)")
        return

    print(f"\n{'='*60}")
    print("FP16 Numerical Stability Test")
    print('='*60)

    device = torch.device("cuda")
    batch_size = 64
    seq_len = 64
    num_classes = 37

    logits_fp32 = torch.randn(batch_size * seq_len, num_classes, device=device)
    targets = torch.randint(0, num_classes, (batch_size * seq_len,), device=device)
    targets[torch.randperm(batch_size * seq_len)[:int(0.85 * batch_size * seq_len)]] = -100

    focal = FocalLoss(gamma=2.0)

    # FP32 baseline
    loss_fp32 = focal(logits_fp32, targets)

    # FP16
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        loss_fp16 = focal(logits_fp32.half(), targets)

    rel_error = abs(loss_fp32.item() - loss_fp16.item()) / loss_fp32.item()

    print(f"FP32 loss: {loss_fp32.item():.6f}")
    print(f"FP16 loss: {loss_fp16.item():.6f}")
    print(f"Relative error: {rel_error:.2%}")

    if rel_error > 0.05:
        print("⚠ WARNING: >5% relative error in FP16")
    else:
        print("✓ FP16 stable (< 5% error)")


def test_wasted_computation():
    """Check if FocalLoss computes focal weights for ignored tokens."""
    print(f"\n{'='*60}")
    print("Wasted Computation Analysis")
    print('='*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    seq_len = 64
    num_classes = 37

    logits = torch.randn(batch_size * seq_len, num_classes, device=device, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size * seq_len,), device=device)

    # 95% ignored (extreme case)
    targets[torch.randperm(batch_size * seq_len)[:int(0.95 * batch_size * seq_len)]] = -100

    focal = FocalLoss(gamma=2.0)

    print("\nFocalLoss implementation:")
    print("1. F.cross_entropy(reduction='none', ignore_index=-100)")
    print("   → CE returns 0 for ignored indices")
    print("2. p = torch.exp(-ce_loss)")
    print("   → Computes exp(0) = 1.0 for all ignored tokens")
    print("3. focal_weight = (1 - p) ** gamma")
    print("   → Computes (1-1)**2 = 0 for all ignored tokens")
    print("4. loss = focal_weight * ce_loss")
    print("   → Computes 0 * 0 = 0 for all ignored tokens")
    print("5. return loss[mask].mean()")
    print("   → Filters out ignored tokens")

    print("\n⚠ ISSUE: Steps 2-4 compute exp, subtraction, and power operations")
    print("  on ALL tokens, including the 95% that will be discarded.")
    print("  With 64*64=4096 tokens, 3891 operations are wasted per batch.")

    # Measure actual computation
    loss = focal(logits, targets)
    n_valid = (targets != -100).sum().item()
    n_total = targets.numel()
    waste_pct = (1 - n_valid / n_total) * 100

    print(f"\nValid tokens: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    print(f"Wasted focal ops: {waste_pct:.1f}%")


if __name__ == "__main__":
    test_performance()
    test_fp16_stability()
    test_wasted_computation()
