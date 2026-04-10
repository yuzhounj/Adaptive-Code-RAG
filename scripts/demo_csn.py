#!/usr/bin/env python3
"""
Demo: randomly sample 10 entries from the CodeSearchNet dataset and display them.

Usage:
    python scripts/demo_csn.py
    python scripts/demo_csn.py --n 5 --seed 123
"""
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse
import random
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.codesearchnet_loader import load_codesearchnet


SEP = "─" * 72


def truncate(text: str, max_lines: int = 12) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n  ... ({len(lines) - max_lines} more lines)"


def display(idx: int, problem):
    print(f"\n{SEP}")
    print(f"  #{idx+1}  {problem.task_id}  |  func: {problem.entry_point}()")
    print(SEP)

    print("\n[Docstring / Query]")
    doc = problem.prompt.strip()
    wrapped = textwrap.fill(doc, width=70, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

    print("\n[Code]")
    for line in truncate(problem.canonical_solution.strip()).splitlines():
        print("  " + line)

    print()


def main():
    parser = argparse.ArgumentParser(description="Demo: random CSN samples")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n",    type=int, default=10, help="Number of samples to display")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    print(f"Loading CodeSearchNet (max={config.data.codesearchnet_max_samples}, seed={seed})...")

    all_samples = load_codesearchnet(
        max_samples=config.data.codesearchnet_max_samples,
        cache_dir=config.data.cache_dir,
    )

    rng = random.Random(seed)
    samples = rng.sample(all_samples, min(args.n, len(all_samples)))

    print(f"Showing {len(samples)} random samples from {len(all_samples)} loaded entries.\n")

    for i, problem in enumerate(samples):
        display(i, problem)

    print(SEP)
    print(f"\nDone. Re-run with --seed {seed} to reproduce this selection.")


if __name__ == "__main__":
    main()
