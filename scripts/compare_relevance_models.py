#!/usr/bin/env python3
"""
Compare code generation performance of checkpoints trained with different LLM judges (3b/7b/14b).
Runs HumanEval evaluation for each checkpoint and prints a comparison table.

Usage:
    python scripts/compare_relevance_models.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import argparse
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.humaneval_loader import load_humaneval
from src.data.corpus_builder import load_humaneval_corpus
from src.retriever.encoder import CodeBERTEncoder
from src.retriever.retriever import DifferentiableRetriever
from src.generator.llm_client import LLMClient
from src.generator.prompt_builder import build_prompt
from src.reward.reward_fn import RewardFunction
from src.utils.metrics import compute_pass_at_k
from src.utils.checkpoint import load_checkpoint
from tqdm import tqdm


def evaluate_model(problems, retriever, llm_client, reward_fn, n_samples):
    """Run retrieval + generation for all problems."""
    all_rewards = []
    for problem in tqdm(problems, desc="Evaluating"):
        context = retriever.retrieve(problem)
        prompt = build_prompt(problem, context.snippets[:1])
        generated_codes = llm_client.generate(prompt, n=n_samples, temperature=0.0)
        rewards = reward_fn.compute(problem, generated_codes, snippets=context.snippets)
        all_rewards.append(rewards)
    return all_rewards


def evaluate_checkpoint(
    checkpoint_path: str,
    config,
    problems: List,
    device: torch.device,
    label: str = "",
) -> Tuple[float, float]:
    """Evaluate a single checkpoint, return (pass@1, pass@n_samples)."""
    encoder = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(device)

    if checkpoint_path:
        load_checkpoint(checkpoint_path, encoder)
        print(f"[{label}] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[{label}] Using pretrained encoder (no RL training)")

    retriever = DifferentiableRetriever(config=config.retriever, encoder=encoder)
    encoder.eval()

    # Build HumanEval corpus index with the current encoder
    humaneval_snippets = load_humaneval_corpus(problems)
    retriever.build_index(humaneval_snippets)

    llm_client = LLMClient(config=config.generator)
    reward_fn = RewardFunction(config=config.reward)
    n_samples = config.generator.n_samples

    with torch.no_grad():
        all_rewards = evaluate_model(problems, retriever, llm_client, reward_fn, n_samples)

    pass_at_1 = compute_pass_at_k(all_rewards, k=1)
    pass_at_n = compute_pass_at_k(all_rewards, k=n_samples)
    return pass_at_1, pass_at_n


def evaluate_baseline(
    config,
    problems: List,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate no-retrieval baseline."""
    llm_client = LLMClient(config=config.generator)
    reward_fn = RewardFunction(config=config.reward)
    n_samples = config.generator.n_samples

    all_rewards = []
    for problem in tqdm(problems, desc="Baseline"):
        prompt = build_prompt(problem, snippets=[])
        generated_codes = llm_client.generate(prompt, n=n_samples, temperature=0.0)
        rewards = reward_fn.compute(problem, generated_codes)
        all_rewards.append(rewards)

    pass_at_1 = compute_pass_at_k(all_rewards, k=1)
    pass_at_n = compute_pass_at_k(all_rewards, k=n_samples)
    return pass_at_1, pass_at_n


def make_bar_chart(results: dict, save_path: str):
    """
    results: { label: {"pass@1": float, "pass@{n_samples}": float} }
    Saves a bar chart comparing pass@1 across models.
    """
    labels = list(results.keys())
    pass1_vals = [results[l]["pass@1"] * 100 for l in labels]

    x = np.arange(len(labels))
    bar_w = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, pass1_vals, bar_w, color="#4C72B0", edgecolor="white", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("pass@1 (%)", fontsize=11)
    ax.set_title(f"HumanEval pass@1 — Checkpoints Trained with Different LLM Judges", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(pass1_vals) * 1.25 + 5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved chart → {save_path}")
    return fig


def write_to_tensorboard(results: dict, fig, n_samples: int, log_dir: str):
    """Write results to TensorBoard as scalars and a figure."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Bar chart as image
        writer.add_figure("eval/comparison_chart", fig, global_step=0)

        # Individual scalars (step = model index)
        for step, (label, metrics) in enumerate(results.items()):
            writer.add_scalar("eval/pass@1", metrics["pass@1"], global_step=step)
            if f"pass@{n_samples}" in metrics:
                writer.add_scalar(f"eval/pass@{n_samples}", metrics[f"pass@{n_samples}"], global_step=step)
            print(f"TensorBoard scalar written: step={step} ({label})")

        writer.close()
        print(f"TensorBoard logs → {log_dir}")
        print(f"Run:  tensorboard --logdir {log_dir}")
    except ImportError:
        print("TensorBoard not installed — skipping TB write.")


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints trained with different LLM judges")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None, help="JSON file to save results")
    parser.add_argument("--chart", default="outputs/relevance_comparison/comparison.png",
                        help="Path to save bar chart (default: outputs/relevance_comparison/comparison.png)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write results to TensorBoard (logs to outputs/logs/relevance_comparison)")
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    problems = load_humaneval(cache_dir=config.data.humaneval_dir)
    eval_size = config.data.humaneval_eval_size
    if eval_size < len(problems):
        problems = random.Random(42).sample(problems, eval_size)
    print(f"Evaluating on {len(problems)} HumanEval problems\n")

    # Define checkpoint paths (labels include newlines for chart display)
    checkpoints = [
        ("Baseline\n(no retrieval)", None),
        ("Pretrained\n(no RL)", None),  # No checkpoint means pretrained CodeBERT
        ("RL-Trained\n(14b judge)", "outputs/checkpoints/step_3000.pt"),
        ("RL-Trained\n(7b judge)", "outputs/checkpoints/7b/step_3000.pt"),
        ("RL-Trained\n(3b judge)", "outputs/checkpoints/3b/step_3000.pt"),
    ]

    results = {}

    # Baseline
    print("=== Baseline (no retrieval) ===")
    pass_at_1, pass_at_n = evaluate_baseline(config, problems, device)
    results["Baseline (no retrieval)"] = {"pass@1": pass_at_1, f"pass@{config.generator.n_samples}": pass_at_n}
    print(f"pass@1:  {pass_at_1:.4f}")
    print(f"pass@{config.generator.n_samples}: {pass_at_n:.4f}\n")

    # Pretrained (no checkpoint)
    print("=== Pretrained CodeBERT (no RL) ===")
    pass_at_1, pass_at_n = evaluate_checkpoint("", config, problems, device, label="Pretrained")
    results["Pretrained (no RL)"] = {"pass@1": pass_at_1, f"pass@{config.generator.n_samples}": pass_at_n}
    print(f"pass@1:  {pass_at_1:.4f}")
    print(f"pass@{config.generator.n_samples}: {pass_at_n:.4f}\n")

    # RL-trained checkpoints
    for label, ckpt_path in checkpoints[2:]:
        if ckpt_path is None:
            continue
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        print(f"=== {label} ===")
        pass_at_1, pass_at_n = evaluate_checkpoint(ckpt_path, config, problems, device, label=label)
        results[label] = {"pass@1": pass_at_1, f"pass@{config.generator.n_samples}": pass_at_n}
        print(f"pass@1:  {pass_at_1:.4f}")
        print(f"pass@{config.generator.n_samples}: {pass_at_n:.4f}\n")

    # Print comparison table (replace newlines with spaces for clean display)
    print("\n" + "=" * 70)
    print("COMPARISON OF RELEVANCE MODEL JUDGE CHECKPOINTS")
    print("=" * 70)
    print(f"{'Model':<30} {'pass@1':>10} {'pass@'+str(config.generator.n_samples):>10}")
    print("-" * 70)
    for label in [
        "Baseline\n(no retrieval)",
        "Pretrained\n(no RL)",
        "RL-Trained\n(14b judge)",
        "RL-Trained\n(7b judge)",
        "RL-Trained\n(3b judge)",
    ]:
        if label in results:
            r = results[label]
            clean_label = label.replace("\n", " ")
            print(f"{clean_label:<30} {r['pass@1']:>10.4f} {r[f'pass@{config.generator.n_samples}']:>10.4f}")

    # Save results to JSON if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Generate bar chart
    print("\n" + "=" * 70)
    print("Generating bar chart...")
    fig = make_bar_chart(results, args.chart)

    # Write to TensorBoard if requested
    if args.tensorboard:
        tb_dir = "outputs/logs/relevance_comparison"
        n_samples = config.generator.n_samples
        write_to_tensorboard(results, fig, n_samples, tb_dir)

    plt.close(fig)


if __name__ == "__main__":
    main()