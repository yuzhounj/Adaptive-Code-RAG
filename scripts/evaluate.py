#!/usr/bin/env python3
"""Evaluate pass@k on trained vs baseline retriever."""
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.humaneval_loader import load_humaneval, split_humaneval
from src.data.corpus_builder import load_corpus_metadata
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
        generated_codes = llm_client.generate(prompt, n=n_samples)
        rewards = reward_fn.compute(problem, generated_codes)
        all_rewards.append(rewards)
    return all_rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeBERT retriever")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Evaluate no-retrieval baseline")
    parser.add_argument("--split", default="eval", choices=["train", "eval", "all"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problems = load_humaneval(cache_dir=config.data.humaneval_dir)
    train_problems, eval_problems = split_humaneval(problems, config.data.train_split)

    if args.split == "train":
        eval_set = train_problems
    elif args.split == "eval":
        eval_set = eval_problems
    else:
        eval_set = problems

    print(f"Evaluating on {len(eval_set)} problems")

    encoder = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(device)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, encoder)

    retriever = DifferentiableRetriever(config=config.retriever, encoder=encoder)

    corpus_snippets = load_corpus_metadata(config.data.corpus_dir)
    index_path = str(Path(config.data.corpus_dir) / "faiss.index")

    if not args.baseline:
        if Path(index_path).exists():
            retriever.faiss_index.load(index_path)
            retriever.corpus_snippets = corpus_snippets
        else:
            retriever.build_index(corpus_snippets)

    llm_client = LLMClient(config=config.generator)
    reward_fn = RewardFunction(config=config.reward)

    n_samples = config.generator.n_samples

    if args.baseline:
        # No-retrieval baseline: empty snippets
        all_rewards = []
        for problem in tqdm(eval_set, desc="Baseline"):
            prompt = build_prompt(problem, snippets=[])
            generated_codes = llm_client.generate(prompt, n=n_samples)
            rewards = reward_fn.compute(problem, generated_codes)
            all_rewards.append(rewards)
    else:
        all_rewards = evaluate_model(eval_set, retriever, llm_client, reward_fn, n_samples)

    pass_at_1 = compute_pass_at_k(all_rewards, k=1)
    pass_at_n = compute_pass_at_k(all_rewards, k=n_samples)

    label = "Baseline" if args.baseline else ("Trained" if args.checkpoint else "Pretrained")
    print(f"\n=== {label} Results ===")
    print(f"pass@1:  {pass_at_1:.4f}")
    print(f"pass@{n_samples}: {pass_at_n:.4f}")


if __name__ == "__main__":
    main()
