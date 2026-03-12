import numpy as np
from typing import List
from math import comb


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. 2021 (HumanEval paper).
    n: total samples, c: number of correct samples, k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_at_k(all_rewards: List[List[float]], k: int = 1) -> float:
    """
    Compute average pass@k across all problems.
    all_rewards: list of reward lists per problem
    """
    scores = []
    for rewards in all_rewards:
        n = len(rewards)
        c = sum(1 for r in rewards if r >= 0.5)  # threshold for "pass"
        score = pass_at_k_unbiased(n, c, k)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def mean_reciprocal_rank(relevance_lists: List[List[int]]) -> float:
    """Compute MRR over a list of relevance lists (1=relevant, 0=not)."""
    mrr = 0.0
    for rel_list in relevance_lists:
        for rank, rel in enumerate(rel_list, 1):
            if rel > 0:
                mrr += 1.0 / rank
                break
    return mrr / len(relevance_lists) if relevance_lists else 0.0


def ndcg_at_k(relevance_list: List[float], k: int) -> float:
    """Compute NDCG@k for a single query."""
    def dcg(rels, k):
        return sum(
            (2**r - 1) / np.log2(i + 2)
            for i, r in enumerate(rels[:k])
        )

    actual_dcg = dcg(relevance_list, k)
    ideal_dcg = dcg(sorted(relevance_list, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_ndcg(relevance_lists: List[List[float]], k: int = 5) -> float:
    """Average NDCG@k over all queries."""
    scores = [ndcg_at_k(rl, k) for rl in relevance_lists]
    return sum(scores) / len(scores) if scores else 0.0
