import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from src.retriever.encoder import CodeBERTEncoder
from src.retriever.faiss_index import FaissIndex
from src.data.schema import CodeSnippet, RetrievedContext, HumanEvalProblem
from src.config import RetrieverConfig


class DifferentiableRetriever:
    """
    Retriever with gradient flow for PPO training.

    FAISS retrieves top-k candidates (no_grad).
    Then explicitly re-encodes the k documents text through the doc encoder
    to preserve full computational graph for Dual-Encoder training.
    """

    def __init__(self, config: RetrieverConfig, encoder: CodeBERTEncoder):
        self.config = config
        self.encoder = encoder
        self.faiss_index = FaissIndex(embedding_dim=config.embedding_dim)
        self.corpus_snippets: List[CodeSnippet] = []

        if config.freeze_doc_encoder:
            self.doc_encoder = copy.deepcopy(encoder)
            for p in self.doc_encoder.parameters():
                p.requires_grad = False
            self.doc_encoder.eval()
        else:
            self.doc_encoder = None

    def build_index(self, snippets: List[CodeSnippet], batch_size: int = 64) -> None:
        self.corpus_snippets = snippets
        device = next(self.encoder.model.parameters()).device

        all_embeddings = []
        texts = [f"{s.docstring}\n{s.code}"[:512] for s in snippets]
        enc = self.doc_encoder if self.doc_encoder is not None else self.encoder

        was_training = enc.training
        enc.eval()
        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embs = enc.encode_corpus_batch(batch, device=device)
                    all_embeddings.append(embs)
        finally:
            if was_training:
                enc.train()

        corpus_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        self.faiss_index.build(corpus_embeddings)

    def refresh_index(self, step: int = 0) -> None:
        if self.config.freeze_doc_encoder:
            return
        if self.corpus_snippets:
            print(f"Refreshing FAISS index at step {step}...")
            self.build_index(self.corpus_snippets)

    def rescore(self, problem: HumanEvalProblem, snippets: List[CodeSnippet]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Optimization 1] Dual-Encoder Gradients:
        Re-encode query AND specifically retrieved snippets with full gradient tracking.
        """
        device = next(self.encoder.model.parameters()).device
        if not snippets:
            dummy_tensor = torch.zeros(1, device=device, requires_grad=True)
            return dummy_tensor, dummy_tensor

        # Encode query WITH gradient
        query_emb = self.encoder.encode_query(problem.prompt, device=device)  # [768]

        # Re-encode documents WITH gradient
        doc_texts = [f"{s.docstring}\n{s.code}"[:512] for s in snippets]
        enc = self.doc_encoder if self.doc_encoder is not None else self.encoder
        corpus_embs_k = enc.encode(doc_texts, device=device, no_grad=False)  # [k, 768]

        # Re-score differentiably
        scores_k = query_emb @ corpus_embs_k.T  # [k]

        # 【关键修复：温度缩放 Temperature Scaling】
        temperature = 0.05
        scaled_scores = scores_k / temperature

        log_probs_k = F.log_softmax(scaled_scores, dim=0)  # [k]

        return log_probs_k, scores_k

    def retrieve(
            self,
            problem: HumanEvalProblem,
            top_k: Optional[int] = None,
    ) -> RetrievedContext:
        k = top_k or self.config.top_k
        device = next(self.encoder.model.parameters()).device

        # FAISS search strictly without grad for speed and safety
        with torch.no_grad():
            query_emb_no_grad = self.encoder.encode_query(problem.prompt, device=device)
            query_np = query_emb_no_grad.cpu().numpy()
            _, indices = self.faiss_index.search(query_np, k)

        valid_mask = indices >= 0
        indices = indices[valid_mask]
        snippets = [self.corpus_snippets[idx] for idx in indices]

        if not snippets:
            dummy = torch.zeros(1, device=device, requires_grad=True)
            return RetrievedContext(problem, [], dummy, dummy)

        # Perform differentiable rescoring
        log_probs_k, scores_k = self.rescore(problem, snippets)

        return RetrievedContext(
            problem=problem,
            snippets=snippets,
            log_probs=log_probs_k,
            scores=scores_k,
        )