from typing import List, Optional
from src.config import RewardConfig
from src.data.schema import CodeSnippet, HumanEvalProblem
from src.reward.executor import batch_execute  # used by compute() for eval
from src.reward.llm_judge import SnippetRelevanceJudge


class RewardFunction:
    """Reward interface: per-snippet relevance-based rewards for REINFORCE training.

    Training uses CodeSearchNet (no test cases), so reward is purely LLM relevance.
    Execution reward is preserved only for HumanEval evaluation via compute().
    """

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
        snippets: List[CodeSnippet],
    ) -> List[float]:
        """Compute per-snippet rewards for REINFORCE training.

        Returns LLM relevance scores — List[float] of length k (one per snippet).
        """
        return self._get_judge().score_batch(problem, snippets)

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
