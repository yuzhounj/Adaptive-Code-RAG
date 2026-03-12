from typing import List
from src.config import RewardConfig
from src.data.schema import HumanEvalProblem
from src.reward.executor import batch_execute
from src.reward.llm_judge import LLMJudge


class RewardFunction:
    """Unified reward interface supporting execution, LLM judge, and hybrid modes."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self._judge: LLMJudge = None

    def _get_judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(self.config)
        return self._judge

    def compute(self, problem: HumanEvalProblem, generated_codes: List[str]) -> List[float]:
        """Compute rewards for multiple generated code samples."""
        mode = self.config.mode

        if mode == "execution":
            return batch_execute(problem, generated_codes, timeout=self.config.execution_timeout)

        elif mode == "llm_judge":
            return self._get_judge().judge_batch(problem, generated_codes)

        elif mode == "hybrid":
            exec_rewards = batch_execute(problem, generated_codes, timeout=self.config.execution_timeout)
            judge_rewards = self._get_judge().judge_batch(problem, generated_codes)
            w = self.config.hybrid_execution_weight
            return [w * e + (1 - w) * j for e, j in zip(exec_rewards, judge_rewards)]

        else:
            raise ValueError(f"Unknown reward mode: {mode}")

    def compute_single(self, problem: HumanEvalProblem, generated_code: str) -> float:
        """Compute reward for a single generated code sample."""
        return self.compute(problem, [generated_code])[0]
