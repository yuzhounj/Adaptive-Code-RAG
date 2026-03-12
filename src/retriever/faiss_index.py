import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional


class FaissIndex:
    """FAISS IndexFlatIP for inner-product (cosine) search on normalized embeddings."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.corpus_embeddings: Optional[np.ndarray] = None  # [N, 768]

    def build(self, embeddings: np.ndarray) -> None:
        """Build index from corpus embeddings. embeddings: [N, dim] float32."""
        assert embeddings.dtype == np.float32, "FAISS requires float32"
        self.index.reset()
        self.index.add(embeddings)
        self.corpus_embeddings = embeddings.copy()
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def search(self, query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k neighbors. Returns (scores [top_k], indices [top_k])."""
        if query_emb.ndim == 1:
            query_emb = query_emb[None]  # [1, dim]
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)
        return scores[0], indices[0]  # [top_k], [top_k]

    def get_embeddings_by_indices(self, indices: np.ndarray) -> torch.Tensor:
        """Retrieve stored corpus embeddings for given indices as torch tensor."""
        assert self.corpus_embeddings is not None, "Index not built yet"
        embs = self.corpus_embeddings[indices]  # [k, dim]
        return torch.from_numpy(embs)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
        np.save(path + ".embeddings.npy", self.corpus_embeddings)
        print(f"Saved FAISS index to {path}")

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)
        emb_path = path + ".embeddings.npy"
        if Path(emb_path).exists():
            self.corpus_embeddings = np.load(emb_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")

    @property
    def ntotal(self) -> int:
        return self.index.ntotal
