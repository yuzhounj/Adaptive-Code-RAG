#!/usr/bin/env python3
"""RL training entry point."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import sys
import argparse
import random
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.humaneval_loader import load_humaneval
from src.data.codesearchnet_loader import load_codesearchnet
from src.data.corpus_builder import load_corpus_metadata, load_humaneval_corpus
from src.retriever.encoder import CodeBERTEncoder
from src.retriever.retriever import DifferentiableRetriever
from src.generator.llm_client import LLMClient
from src.reward.reward_fn import RewardFunction
from src.rl.trainer import RLTrainer


def parse_overrides(override_strs):
    """Parse 'key=value' override strings."""
    overrides = {}
    for s in override_strs:
        if "=" in s:
            k, v = s.split("=", 1)
            # Try to cast to number
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k] = v
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Train CodeBERT retriever with REINFORCE")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="Dot-notation overrides: rl.learning_rate=1e-5")
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    config = load_config(args.config, overrides=overrides)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data: CodeSearchNet for training, HumanEval for eval
    print(f"Loading CodeSearchNet as training set (max={config.data.codesearchnet_max_samples})...")
    train_problems = load_codesearchnet(
        max_samples=config.data.codesearchnet_max_samples,
        cache_dir=config.data.cache_dir,
    )
    print("Loading HumanEval as eval set...")
    eval_problems = load_humaneval(cache_dir=config.data.humaneval_dir)
    eval_size = config.data.humaneval_eval_size
    if eval_size < len(eval_problems):
        eval_problems = random.Random(42).sample(eval_problems, eval_size)
    humaneval_snippets = load_humaneval_corpus(eval_problems)
    print(f"Train: {len(train_problems)}, Eval: {len(eval_problems)}, HumanEval corpus: {len(humaneval_snippets)}")

    print("Loading corpus...")
    corpus_snippets = load_corpus_metadata(config.data.corpus_dir)
    print(f"Corpus size: {len(corpus_snippets)}")

    # Build components
    encoder = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(device)

    retriever = DifferentiableRetriever(config=config.retriever, encoder=encoder)

    # Load or build FAISS index
    index_path = str(Path(config.data.corpus_dir) / "faiss.index")
    if Path(index_path).exists():
        retriever.faiss_index.load(index_path)
        retriever.corpus_snippets = corpus_snippets
    else:
        print("Building FAISS index...")
        retriever.build_index(corpus_snippets)

    llm_client = LLMClient(config=config.generator)
    reward_fn = RewardFunction(config=config.reward)

    trainer = RLTrainer(
        config=config,
        retriever=retriever,
        encoder=encoder,
        llm_client=llm_client,
        reward_fn=reward_fn,
        device=device,
    )

    trainer.train(
        train_problems=train_problems,
        eval_problems=eval_problems,
        humaneval_snippets=humaneval_snippets,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
