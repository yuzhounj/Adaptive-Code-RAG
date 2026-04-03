import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from src.retriever.encoder import CodeBERTEncoder
from src.retriever.faiss_index import FaissIndex
from src.data.schema import CodeSnippet, RetrievedContext, HumanEvalProblem
from src.config import RetrieverConfig


class DifferentiableRetriever:
    """
    Retriever with gradient flow for RL training.
    Upgraded for GRPO: FAISS retrieves a larger pool, and we sample G items
    based on their similarity probabilities to enable exploration.
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
        texts = [f"{s.docstring} {s.code}"[:512] for s in snippets]
        enc = self.doc_encoder if self.doc_encoder is not None else self.encoder

        was_training = enc.training
        enc.eval()
        try:
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

    def retrieve(
            self,
            problem: HumanEvalProblem,
            top_k: Optional[int] = None,
    ) -> RetrievedContext:

        k = top_k or self.config.top_k
        device = next(self.encoder.model.parameters()).device

        # GRPO: 查出一个更大的候选池，以便从中采样 (4倍 k，保底32)
        pool_size = max(k * 4, 32)

        # Encode query WITH gradient
        query_emb = self.encoder.encode_query(problem.prompt, device=device)  # [768]

        # FAISS search (no grad)
        query_np = query_emb.detach().cpu().numpy()
        _, pool_indices = self.faiss_index.search(query_np, pool_size)

        valid_mask = pool_indices >= 0
        pool_indices = pool_indices[valid_mask]

        if len(pool_indices) == 0:
            dummy_log_prob = torch.zeros(1, device=device, requires_grad=True)
            return RetrievedContext(problem=problem, snippets=[], log_probs=dummy_log_prob, scores=dummy_log_prob)

        # 计算候选池中所有片段的相似度分数 (Gradient flows here)
        corpus_embs_pool = self.faiss_index.get_embeddings_by_indices(pool_indices).to(device)
        scores_pool = query_emb @ corpus_embs_pool.T  # [pool_size]

        # 在池子内部计算对数概率
        log_probs_pool = F.log_softmax(scores_pool, dim=0)

        # === GRPO 核心探索机制 ===
        num_to_take = min(k, len(pool_indices))

        if self.encoder.training:
            # 训练时：按照概率分布进行无放回采样，赋予模型探索不同 snippet 的机会
            probs_pool = torch.exp(log_probs_pool)
            if torch.isnan(probs_pool).any() or probs_pool.sum() == 0:
                probs_pool = torch.ones_like(probs_pool) / len(probs_pool)

            sampled_local_indices = torch.multinomial(probs_pool, num_samples=num_to_take, replacement=False)
        else:
            # 评估时：确定性取最高分的 Top-K
            _, sampled_local_indices = torch.topk(scores_pool, num_to_take)

        # 提取被选中的片段的对数概率、分数和原索引
        log_probs_k = log_probs_pool[sampled_local_indices]
        scores_k = scores_pool[sampled_local_indices]
        final_indices = pool_indices[sampled_local_indices.cpu().numpy()]

        snippets = [self.corpus_snippets[idx] for idx in final_indices]

        return RetrievedContext(
            problem=problem,
            snippets=snippets,
            log_probs=log_probs_k,
            scores=scores_k,
        )