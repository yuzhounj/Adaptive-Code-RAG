import asyncio
from typing import List
import litellm
from src.config import RewardConfig
from src.data.schema import HumanEvalProblem


LLM_JUDGE_PROMPT = """You are a code quality judge. Evaluate the following Python function implementation.

Problem description:
{problem_prompt}

Generated implementation:
```python
{generated_code}
```

Rate the quality of this implementation on a scale from 0.0 to 1.0:
- 1.0: Correct, clean, handles edge cases
- 0.7-0.9: Mostly correct but minor issues
- 0.4-0.6: Partial implementation or logic errors
- 0.1-0.3: Significant problems
- 0.0: Completely wrong or empty

Respond with ONLY a single float number between 0.0 and 1.0."""


class LLMJudge:
    """LLM-as-judge for code quality evaluation (fallback when execution unavailable)."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self._api_base = config.llm_judge_api_base or None

    async def judge_async(self, problem: HumanEvalProblem, generated_code: str) -> float:
        """Get LLM quality score for generated code."""
        prompt = LLM_JUDGE_PROMPT.format(
            problem_prompt=problem.prompt,
            generated_code=generated_code,
        )
        try:
            response = await litellm.acompletion(
                model=self.config.llm_judge_model,
                api_base=self._api_base,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()
            score = float(text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"LLM judge error: {e}")
            return 0.0

    def judge(self, problem: HumanEvalProblem, generated_code: str) -> float:
        return asyncio.run(self.judge_async(problem, generated_code))

    async def judge_batch_async(
        self, problem: HumanEvalProblem, generated_codes: List[str]
    ) -> List[float]:
        tasks = [self.judge_async(problem, code) for code in generated_codes]
        return await asyncio.gather(*tasks)

    def judge_batch(self, problem: HumanEvalProblem, generated_codes: List[str]) -> List[float]:
        return asyncio.run(self.judge_batch_async(problem, generated_codes))
