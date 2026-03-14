from typing import List, Optional
from src.config import RewardConfig
from src.data.schema import CodeSnippet, HumanEvalProblem
from src.reward.executor import batch_execute
from src.reward.llm_judge import SnippetRelevanceJudge


class RewardFunction:
    """Reward interface: per-snippet rewards combining execution + LLM relevance."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self._judge: Optional[SnippetRelevanceJudge] = None

    def _get_judge(self) -> SnippetRelevanceJudge:
        if self._judge is None:
            self._judge = SnippetRelevanceJudge(self.config)
        return self._judge

    def compute_snippet_rewards(
        self,
        problem: HumanEvalProblem,
        generated_codes: List[str],
        snippets: List[CodeSnippet],
    ) -> List[float]:
        """Compute per-snippet rewards for REINFORCE training.

        Each snippet reward = w_exec * execution_reward + w_rel * relevance_score
        where execution_reward is the mean pass rate across n_samples generations.

        Returns List[float] of length k (one per snippet).
        """
        exec_rewards = batch_execute(
            problem, generated_codes,
            timeout=self.config.execution_timeout,
            snippets=snippets,
        )
        exec_reward = sum(exec_rewards) / len(exec_rewards) if exec_rewards else 0.0

        relevance_scores = self._get_judge().score_batch(problem, snippets)

        w_exec = self.config.execution_weight
        w_rel = self.config.relevance_weight
        return [w_exec * exec_reward + w_rel * rel for rel in relevance_scores]

    def compute(
        self,
        problem: HumanEvalProblem,
        generated_codes: List[str],
        snippets: Optional[List[CodeSnippet]] = None,
    ) -> List[float]:
        """Compute execution-only rewards for evaluation (pass@k).

        Returns List[float] of length n_samples (one per generated code).
        """
        return batch_execute(
            problem, generated_codes,
            timeout=self.config.execution_timeout,
            snippets=snippets,
        )
