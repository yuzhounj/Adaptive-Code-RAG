#!/usr/bin/env python3
"""Evaluate pass@k on trained vs baseline retriever."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import random
import torch
from pathlib import Path

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
    all_rewards = []
    for problem in tqdm(problems, desc="Evaluating"):
        context = retriever.retrieve(problem)
        prompt = build_prompt(problem, context.snippets)
        generated_codes = llm_client.generate(prompt, n=n_samples, temperature=0.0)
        rewards = reward_fn.compute(problem, generated_codes)
        all_rewards.append(rewards)
    return all_rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeBERT retriever on HumanEval")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Evaluate no-retrieval baseline")
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
    print(f"Evaluating on {len(problems)} HumanEval problems")

    encoder = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(device)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, encoder)

    retriever = DifferentiableRetriever(config=config.retriever, encoder=encoder)

    encoder.eval()

    if not args.baseline:
        # Build HumanEval corpus index with the current encoder
        humaneval_snippets = load_humaneval_corpus(problems)
        retriever.build_index(humaneval_snippets)

    llm_client = LLMClient(config=config.generator)
    reward_fn = RewardFunction(config=config.reward)

    n_samples = config.generator.n_samples

    with torch.no_grad():
        if args.baseline:
            all_rewards = []
            for problem in tqdm(problems, desc="Baseline"):
                prompt = build_prompt(problem, snippets=[])
                generated_codes = llm_client.generate(prompt, n=n_samples, temperature=0.0)
                rewards = reward_fn.compute(problem, generated_codes)
                all_rewards.append(rewards)
        else:
            all_rewards = evaluate_model(problems, retriever, llm_client, reward_fn, n_samples)

    pass_at_1 = compute_pass_at_k(all_rewards, k=1)
    pass_at_n = compute_pass_at_k(all_rewards, k=n_samples)

    label = "Baseline" if args.baseline else ("Trained" if args.checkpoint else "Pretrained")
    print(f"\n=== {label} Results ===")
    print(f"pass@1:  {pass_at_1:.4f}")
    print(f"pass@{n_samples}: {pass_at_n:.4f}")


if __name__ == "__main__":
    main()
