#!/usr/bin/env python3
"""
对比三种配置在 HumanEval 上的通过率，用于汇报展示。

三列对比：
  1. Baseline          : qwen2.5:7b，无检索
  2. Pretrained        : qwen2.5:7b + CodeBERT 预训练权重
  3. RL-Trained        : qwen2.5:7b + CodeBERT + REINFORCE 训练后

输出：
  - outputs/eval_comparison/comparison.png  — 本地柱状图（pass@1）
  - outputs/logs/eval_comparison/           — TensorBoard 日志（图像 + 标量）
    查看：tensorboard --logdir outputs/logs/eval_comparison

前置条件：
  - Ollama 服务已启动（ollama serve）
  - HumanEval 数据已下载（data/humaneval/）

用法：
    # 只跑 baseline + pretrained（无 checkpoint 时）
    python scripts/compare_eval.py

    # 跑全部三列
    python scripts/compare_eval.py --checkpoint outputs/checkpoints/step_500.pt

    # 指定配置文件
    python scripts/compare_eval.py --config configs/default.yaml --checkpoint outputs/checkpoints/step_500.pt
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
from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _passed(rewards) -> bool:
    """A problem is considered passed if any generated sample executes correctly (reward == 1.0)."""
    return any(r == 1.0 for r in rewards)


def run_baseline(problems, llm_client, reward_fn, n_samples):
    """No retrieval — pure LLM. Returns (all_rewards, pass_dict)."""
    all_rewards = []
    pass_dict = {}
    for problem in tqdm(problems, desc="[1/3] Baseline (no retrieval)"):
        prompt = build_prompt(problem, snippets=[])
        generated_codes = llm_client.generate(prompt, n=n_samples, temperature=0.0)
        rewards = reward_fn.compute(problem, generated_codes)
        all_rewards.append(rewards)
        pass_dict[problem.task_id] = _passed(rewards)
    return all_rewards, pass_dict


def run_with_retriever(problems, retriever, llm_client, reward_fn, n_samples, desc, log_dir=None):
    """With retrieval (pretrained or RL-trained). Returns (all_rewards, pass_dict)."""
    all_rewards = []
    pass_dict = {}
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    for problem in tqdm(problems, desc=desc):
        context = retriever.retrieve(problem)
        prompt = build_prompt(problem, context.snippets[:1])
        generated_codes = llm_client.generate(prompt, n=n_samples, temperature=0.0)
        rewards = reward_fn.compute(problem, generated_codes, snippets=context.snippets)
        all_rewards.append(rewards)
        passed = _passed(rewards)
        pass_dict[problem.task_id] = passed

        if log_dir:
            task_id_safe = problem.task_id.replace("/", "_")
            fname = os.path.join(log_dir, f"{'PASS' if passed else 'FAIL'}_{task_id_safe}.txt")
            sim_scores = context.scores.tolist() if hasattr(context.scores, "tolist") else list(context.scores)
            lines = [f"# [{('PASS' if passed else 'FAIL')}] {problem.task_id}\n\n"]
            lines.append("## Problem Prompt\n\n```python\n" + problem.prompt + "\n```\n\n")
            lines.append("## Retrieved Snippets (only Snippet 1 is used in prompt)\n\n")
            for i, snippet in enumerate(context.snippets):
                sim = sim_scores[i] if i < len(sim_scores) else float("nan")
                used = " ← used in prompt" if i == 0 else ""
                lines.append(f"### Snippet {i+1}{used}  |  Similarity: {sim:.4f}\n\n")
                lines.append(f"Docstring: {snippet.docstring}\n\n")
                lines.append(f"```python\n{snippet.code}\n```\n\n")
            lines.append(f"## Generated Code\n\n```python\n{generated_codes[0] if generated_codes else ''}\n```\n")
            with open(fname, "w", encoding="utf-8") as f:
                f.writelines(lines)

    return all_rewards, pass_dict


def build_retriever(config, encoder, device, problems):
    """Build a DifferentiableRetriever with a fresh HumanEval index."""
    retriever = DifferentiableRetriever(config=config.retriever, encoder=encoder)
    humaneval_snippets = load_humaneval_corpus(problems)
    retriever.build_index(humaneval_snippets)
    return retriever


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_bar_chart(results: dict, save_path: str):
    """
    results: { label: {"pass@1": float} }
    Returns a matplotlib Figure.
    """
    labels = list(results.keys())
    pass1_vals = [results[l]["pass@1"] * 100 for l in labels]

    x = np.arange(len(labels))
    bar_w = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(x, pass1_vals, bar_w, color="#4C72B0", edgecolor="white", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("pass@1 (%)", fontsize=11)
    ax.set_title("HumanEval pass@1 — Three Configurations", fontsize=12, pad=12)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare baseline / pretrained / RL on HumanEval")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="RL-trained checkpoint (.pt). If omitted, only baseline + pretrained are run.")
    args = parser.parse_args()

    config = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # --- Problems ---
    problems = load_humaneval(cache_dir=config.data.humaneval_dir)
    eval_size = config.data.humaneval_eval_size
    if eval_size < len(problems):
        problems = random.Random(42).sample(problems, eval_size)
    print(f"Evaluating on {len(problems)} HumanEval problems\n")

    llm_client = LLMClient(config=config.generator)
    reward_fn = RewardFunction(config=config.reward)
    n_samples = config.generator.n_samples

    results = {}

    with torch.no_grad():
        # 1. Baseline — no retrieval
        rewards_baseline, _ = run_baseline(problems, llm_client, reward_fn, n_samples)
        results["Baseline\n(LLM only)"] = {
            "pass@1": compute_pass_at_k(rewards_baseline, k=1),
        }

        # 2. Pretrained CodeBERT (no RL)
        encoder_pre = CodeBERTEncoder(
            model_name=config.retriever.model_name,
            max_seq_len=config.retriever.max_seq_len,
        ).to(device)
        encoder_pre.eval()
        retriever_pre = build_retriever(config, encoder_pre, device, problems)
        rewards_pre, pass_pre = run_with_retriever(
            problems, retriever_pre, llm_client, reward_fn, n_samples,
            desc="[2/3] Pretrained CodeBERT",
            log_dir="outputs/eval_comparison/pretrained",
        )
        results["Pretrained\nCodeBERT"] = {
            "pass@1": compute_pass_at_k(rewards_pre, k=1),
        }

        # 3. RL-Trained CodeBERT (optional)
        pass_rl = None
        if args.checkpoint:
            encoder_rl = CodeBERTEncoder(
                model_name=config.retriever.model_name,
                max_seq_len=config.retriever.max_seq_len,
            ).to(device)
            load_checkpoint(args.checkpoint, encoder_rl)
            encoder_rl.eval()
            retriever_rl = build_retriever(config, encoder_rl, device, problems)
            rewards_rl, pass_rl = run_with_retriever(
                problems, retriever_rl, llm_client, reward_fn, n_samples,
                desc="[3/3] RL-Trained CodeBERT",
                log_dir="outputs/eval_comparison/rl_trained",
            )
            results["RL-Trained\nCodeBERT"] = {
                "pass@1": compute_pass_at_k(rewards_rl, k=1),
            }

    # --- Print table ---
    print("\n=== Results ===")
    print(f"{'Configuration':<25}  {'pass@1':>8}")
    print("-" * 36)
    for label, metrics in results.items():
        clean_label = label.replace("\n", " ")
        print(f"{clean_label:<25}  {metrics['pass@1']:>8.4f}")

    # --- Regression analysis: pretrained pass but RL fail ---
    if pass_rl is not None:
        regressed = sorted(
            task_id for task_id, passed in pass_pre.items()
            if passed and not pass_rl.get(task_id, True)
        )
        print(f"\n=== Pretrained ✓ → RL-Trained ✗  ({len(regressed)} problems) ===")
        if regressed:
            for task_id in regressed:
                print(f"  {task_id}")
        else:
            print("  (none)")

        gained = sorted(
            task_id for task_id, passed in pass_rl.items()
            if passed and not pass_pre.get(task_id, True)
        )
        print(f"\n=== Pretrained ✗ → RL-Trained ✓  ({len(gained)} problems) ===")
        if gained:
            for task_id in gained:
                print(f"  {task_id}")
        else:
            print("  (none)")

    # --- Plot ---
    chart_path = "outputs/eval_comparison/comparison.png"
    fig = make_bar_chart(results, chart_path)

    # --- Write to TensorBoard ---
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = "outputs/logs/eval_comparison"
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

        # Bar chart as image
        writer.add_figure("eval/comparison_chart", fig, global_step=0)

        # Individual scalars (step = config index)
        for step, (label, metrics) in enumerate(results.items()):
            clean = label.replace("\n", " ")
            writer.add_scalar("eval/pass@1", metrics["pass@1"], global_step=step)
            print(f"TensorBoard scalar written: step={step} ({clean})")

        writer.close()
        print(f"\nTensorBoard logs → {tb_dir}")
        print(f"Run:  tensorboard --logdir {tb_dir}")
    except ImportError:
        print("TensorBoard not installed — skipping TB write.")

    plt.close(fig)


if __name__ == "__main__":
    main()
