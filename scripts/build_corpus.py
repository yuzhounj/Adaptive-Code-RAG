#!/usr/bin/env python3
"""Build CodeSearchNet corpus and FAISS index for retriever training."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.codesearchnet_loader import load_codesearchnet
from src.data.humaneval_loader import load_humaneval
from src.data.corpus_builder import load_humaneval_corpus, save_corpus_metadata
from src.retriever.encoder import CodeBERTEncoder
from src.retriever.faiss_index import FaissIndex
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Build CodeSearchNet corpus and FAISS index")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    project_root = Path(__file__).parent.parent

    print("Caching HumanEval dataset...")
    load_humaneval(cache_dir=str(project_root / config.data.humaneval_dir))

    print(f"Loading CodeSearchNet (max={config.data.codesearchnet_max_samples})...")
    problems = load_codesearchnet(
        max_samples=config.data.codesearchnet_max_samples,
        cache_dir=str(project_root / config.data.cache_dir),
    )
    snippets = load_humaneval_corpus(problems)
    print(f"Loaded {len(snippets)} snippets from CodeSearchNet")

    # Save corpus metadata
    corpus_dir = str(project_root / config.data.corpus_dir)
    save_corpus_metadata(snippets, corpus_dir)

    # Encode corpus
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Encoding corpus on {device}...")
    encoder = CodeBERTEncoder(
        model_name=config.retriever.model_name,
        max_seq_len=config.retriever.max_seq_len,
    ).to(device)
    encoder.eval()

    batch_size = 64
    all_embeddings = []
    texts = [f"{s.docstring}\n{s.code}" for s in snippets]

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        embs = encoder.encode_corpus_batch(batch, device=device)
        all_embeddings.append(embs)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Build and save FAISS index
    index = FaissIndex(embedding_dim=config.retriever.embedding_dim)
    index.build(embeddings)

    index_path = str(project_root / config.data.corpus_dir / "faiss.index")
    index.save(index_path)

    # Quick test query
    print("\nTest query: 'sort a list of integers'")
    test_texts = ["sort a list of integers"]
    test_emb = encoder.encode_corpus_batch(test_texts, device=device)[0]
    scores, indices = index.search(test_emb, top_k=3)
    for score, idx in zip(scores, indices):
        print(f"  Score={score:.4f}: {snippets[idx].docstring[:80]}")

    print("\nDone! Corpus build complete.")


if __name__ == "__main__":
    main()
