import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.retriever.faiss_index import FaissIndex
from src.retriever.encoder import CodeBERTEncoder
from src.data.schema import HumanEvalProblem, CodeSnippet


def test_faiss_index_build_and_search():
    index = FaissIndex(embedding_dim=8)
    embeddings = np.random.randn(100, 8).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    index.build(embeddings)
    assert index.ntotal == 100

    query = embeddings[0]
    scores, indices = index.search(query, top_k=5)
    assert len(scores) == 5
    assert len(indices) == 5
    assert 0 in indices  # top result should be query itself


def test_faiss_index_get_embeddings_by_indices():
    index = FaissIndex(embedding_dim=8)
    embeddings = np.random.randn(10, 8).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.build(embeddings)

    result = index.get_embeddings_by_indices(np.array([0, 1, 2]))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 8)


def test_faiss_index_save_load(tmp_path):
    index = FaissIndex(embedding_dim=8)
    embeddings = np.random.randn(10, 8).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.build(embeddings)

    path = str(tmp_path / "test.index")
    index.save(path)

    index2 = FaissIndex(embedding_dim=8)
    index2.load(path)
    assert index2.ntotal == 10
