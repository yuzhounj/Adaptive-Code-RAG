import asyncio
from typing import List
import litellm
from src.config import RewardConfig
from src.data.schema import CodeSnippet, HumanEvalProblem


SNIPPET_RELEVANCE_PROMPT = """You are evaluating the relevance of a code snippet for solving a programming problem.

Problem:
{problem_prompt}

Code Snippet:
```python
{snippet_code}
```

How helpful is this snippet for solving the above problem?
Rate from 0.0 to 1.0:
- 1.0: Directly solves or demonstrates the exact algorithm needed
- 0.7-0.9: Shows closely related patterns or techniques
- 0.4-0.6: Somewhat related, could provide partial guidance
- 0.1-0.3: Tangentially related
- 0.0: Completely unrelated

Respond with ONLY a single float between 0.0 and 1.0."""


class SnippetRelevanceJudge:
    """LLM-as-judge for snippet relevance scoring."""

    def __init__(self, config: RewardConfig):
        self.config = config

    async def score_async(self, problem: HumanEvalProblem, snippet: CodeSnippet) -> float:
        """Score relevance of a single snippet to a problem."""
        prompt = SNIPPET_RELEVANCE_PROMPT.format(
            problem_prompt=problem.prompt,
            snippet_code=snippet.code,
        )
        try:
            response = await litellm.acompletion(
                model=self.config.relevance_model,
                api_base=self.config.relevance_api_base,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()
            score = float(text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Relevance judge error: {e}")
            return 0.0

    async def score_batch_async(
        self, problem: HumanEvalProblem, snippets: List[CodeSnippet]
    ) -> List[float]:
        tasks = [self.score_async(problem, snippet) for snippet in snippets]
        return await asyncio.gather(*tasks)

    def score_batch(
        self, problem: HumanEvalProblem, snippets: List[CodeSnippet]
    ) -> List[float]:
        """Score relevance of multiple snippets concurrently."""
        if not snippets:
            return []
        return asyncio.run(self.score_batch_async(problem, snippets))
