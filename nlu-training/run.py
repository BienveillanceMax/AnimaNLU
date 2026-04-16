"""
Central orchestrator — runs the full NLU training pipeline end-to-end.

Usage:
    python run.py              # Run everything
    python run.py --from train # Resume from a specific step
    python run.py --only eval  # Run a single step
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("data",      "prepare_data.py",  "Prepare dataset (MASSIVE + supplements)"),
    ("train",     "train.py",         "Fine-tune JointCamemBERTav2"),
    ("eval",      "evaluate.py",      "Evaluate on test set"),
    ("calibrate", "calibrate.py",     "Temperature scaling per head"),
    ("export",    "export_onnx.py",   "Export ONNX (FP32/FP16/INT8) + test vectors"),
]


def run_step(script, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  → python {script}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n✗ FAILED: {script} (exit code {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)

    print(f"\n✓ {script} done ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Anima NLU training pipeline")
    parser.add_argument("--from", dest="from_step", help="Resume from step: " + ", ".join(s[0] for s in STEPS))
    parser.add_argument("--only", help="Run single step: " + ", ".join(s[0] for s in STEPS))
    args = parser.parse_args()

    step_names = [s[0] for s in STEPS]

    if args.only:
        if args.only not in step_names:
            print(f"Unknown step '{args.only}'. Available: {', '.join(step_names)}")
            sys.exit(1)
        idx = step_names.index(args.only)
        run_step(STEPS[idx][1], STEPS[idx][2])
        return

    start_idx = 0
    if args.from_step:
        if args.from_step not in step_names:
            print(f"Unknown step '{args.from_step}'. Available: {', '.join(step_names)}")
            sys.exit(1)
        start_idx = step_names.index(args.from_step)

    total_start = time.time()
    timings = {}

    for i, (name, script, desc) in enumerate(STEPS):
        if i < start_idx:
            continue
        timings[name] = run_step(script, desc)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete ({total:.0f}s total)")
    print(f"{'='*60}")
    for name, t in timings.items():
        print(f"  {name:12s}: {t:.1f}s")


if __name__ == "__main__":
    main()
