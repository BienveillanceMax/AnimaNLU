"""
Test optimized FocalLoss that avoids wasted computation on -100 tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class FocalLossOriginal(nn.Module):
    """Original implementation from model.py"""
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


class FocalLossOptimized(nn.Module):
    """Optimized: mask BEFORE computing focal weights"""
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Mask first
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Only compute CE on valid tokens
        valid_logits = logits[mask]
        valid_targets = targets[mask]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        loss = focal_weight * ce_loss

        return loss.mean()


def benchmark_both(batch_size=64, seq_len=64, num_classes=37, ignore_ratio=0.485):
    """Compare original vs optimized FocalLoss"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits = torch.randn(batch_size * seq_len, num_classes, device=device)
    targets = torch.randint(0, num_classes, (batch_size * seq_len,), device=device)

    n_ignore = int(batch_size * seq_len * ignore_ratio)
    ignore_indices = torch.randperm(batch_size * seq_len)[:n_ignore]
    targets[ignore_indices] = -100

    focal_orig = FocalLossOriginal(gamma=2.0)
    focal_opt = FocalLossOptimized(gamma=2.0)

    # Warmup
    for _ in range(10):
        _ = focal_orig(logits, targets)
        _ = focal_opt(logits, targets)

    # Benchmark original
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        loss_orig = focal_orig(logits, targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    time_orig = (time.perf_counter() - start) / 100 * 1000

    # Benchmark optimized
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        loss_opt = focal_opt(logits, targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    time_opt = (time.perf_counter() - start) / 100 * 1000

    speedup = (time_orig - time_opt) / time_orig * 100

    print(f"{'='*60}")
    print(f"Ignore ratio: {ignore_ratio:.1%}")
    print(f"{'='*60}")
    print(f"Original:  {time_orig:.4f} ms  (loss={loss_orig.item():.4f})")
    print(f"Optimized: {time_opt:.4f} ms  (loss={loss_opt.item():.4f})")
    print(f"Speedup:   {speedup:+.1f}%")
    print(f"Loss diff: {abs(loss_orig.item() - loss_opt.item()):.6f}")
    print()


if __name__ == "__main__":
    print("\nBenchmarking FocalLoss optimization")
    print("(masking before focal weight computation)\n")

    # Real-world ratio from analysis
    benchmark_both(ignore_ratio=0.485)

    # Worst case (supplements only)
    benchmark_both(ignore_ratio=1.0)
