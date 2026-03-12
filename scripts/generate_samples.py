#!/usr/bin/env python3
"""Inference demo: retrieve and generate for a given problem."""
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.humaneval_loader import load_humaneval
from src.data.corpus_builder import load_corpus_metadata
from src.retriever.encoder import CodeBERTEncoder
from src.retriever.retriever import DifferentiableRetriever
from src.generator.llm_client import LLMClient
from src.generator.prompt_builder import build_prompt
from src.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate code samples using RAG")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task-id", default=None, help="Specific HumanEval task ID")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    problems = load_humaneval(cache_dir=config.data.humaneval_dir)
    if args.task_id:
        problems = [p for p in problems if p.task_id == args.task_id]
        if not problems:
            print(f"Task {args.task_id} not found")
            return
    else:
        problems = problems[:3]  # Demo: first 3

    encoder = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(device)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, encoder)

    retriever = DifferentiableRetriever(config=config.retriever, encoder=encoder)
    corpus_snippets = load_corpus_metadata(config.data.corpus_dir)
    index_path = str(Path(config.data.corpus_dir) / "faiss.index")
    if Path(index_path).exists():
        retriever.faiss_index.load(index_path)
        retriever.corpus_snippets = corpus_snippets

    llm_client = LLMClient(config=config.generator)

    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Task: {problem.task_id}")
        print(f"Prompt:\n{problem.prompt}")

        context = retriever.retrieve(problem)
        print(f"\nRetrieved {len(context.snippets)} snippets")
        if context.snippets:
            print(f"Top snippet: {context.snippets[0].docstring[:100]}")

        prompt = build_prompt(problem, context.snippets)
        completions = llm_client.generate(prompt, n=args.n)

        for i, code in enumerate(completions, 1):
            print(f"\n--- Generation {i} ---")
            print(code)


if __name__ == "__main__":
    main()
