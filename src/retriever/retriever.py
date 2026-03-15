import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.retriever.encoder import CodeBERTEncoder
from src.retriever.faiss_index import FaissIndex
from src.data.schema import CodeSnippet, RetrievedContext, HumanEvalProblem
from src.config import RetrieverConfig


class DifferentiableRetriever:
    """
    Retriever with gradient flow for REINFORCE training.

    FAISS retrieves top-k candidates (no_grad, periodic rebuild).
    Then re-scores them differentiably: query_emb @ corpus_embs_k.T
    log_prob = log_softmax(scores).sum() — gradient flows to query encoder.
    """

    def __init__(self, config: RetrieverConfig, encoder: CodeBERTEncoder):
        self.config = config
        self.encoder = encoder
        self.faiss_index = FaissIndex(embedding_dim=config.embedding_dim)
        self.corpus_snippets: List[CodeSnippet] = []
        self._step_count = 0

        if config.freeze_doc_encoder:
            self.doc_encoder = copy.deepcopy(encoder)
            for p in self.doc_encoder.parameters():
                p.requires_grad = False
            self.doc_encoder.eval()
        else:
            self.doc_encoder = None

    def build_index(self, snippets: List[CodeSnippet], batch_size: int = 64) -> None:
        """Encode corpus and build FAISS index. No gradient."""
        self.corpus_snippets = snippets
        device = next(self.encoder.model.parameters()).device

        all_embeddings = []
        texts = [f"{s.docstring} {s.code}"[:512] for s in snippets]
        enc = self.doc_encoder if self.doc_encoder is not None else self.encoder

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = enc.encode_corpus_batch(batch, device=device)
            all_embeddings.append(embs)

        corpus_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        self.faiss_index.build(corpus_embeddings)

    def refresh_index(self) -> None:
        """Rebuild FAISS index with updated encoder weights. Called periodically."""
        if self.config.freeze_doc_encoder:
            return  # corpus embedding 固定，无需重建
        if self.corpus_snippets:
            print(f"Refreshing FAISS index at step {self._step_count}...")
            self.build_index(self.corpus_snippets)

    def retrieve(
        self,
        problem: HumanEvalProblem,
        top_k: Optional[int] = None,
    ) -> RetrievedContext:
        """
        Retrieve top-k snippets for a problem.
        Returns RetrievedContext with gradient-attached log_probs.
        """
        k = top_k or self.config.top_k
        device = next(self.encoder.model.parameters()).device

        # Encode query WITH gradient
        query_emb = self.encoder.encode_query(problem.prompt, device=device)  # [768]

        # FAISS search (no grad)
        query_np = query_emb.detach().cpu().numpy()
        _, indices = self.faiss_index.search(query_np, k)

        # Filter invalid indices
        valid_mask = indices >= 0
        indices = indices[valid_mask]

        if len(indices) == 0:
            # Fallback: return empty context with zero log_prob
            dummy_log_prob = torch.zeros(1, device=device, requires_grad=True)
            return RetrievedContext(
                problem=problem,
                snippets=[],
                log_probs=dummy_log_prob,
                scores=dummy_log_prob,
            )

        # Re-score differentiably: gradient flows through query_emb
        corpus_embs_k = self.faiss_index.get_embeddings_by_indices(indices).to(device)  # [k, 768]
        scores_k = query_emb @ corpus_embs_k.T  # [k], gradient flows here

        log_probs_k = F.log_softmax(scores_k, dim=0)  # [k]

        # Get corresponding snippets
        snippets = [self.corpus_snippets[idx] for idx in indices]

        return RetrievedContext(
            problem=problem,
            snippets=snippets,
            log_probs=log_probs_k,
            scores=scores_k,
        )

    def compute_log_prob(self, context: RetrievedContext) -> torch.Tensor:
        """Scalar log probability for REINFORCE: sum of log probs of retrieved set."""
        if context.log_probs is None:
            raise ValueError("RetrievedContext has no log_probs")
        return context.log_probs.sum()
