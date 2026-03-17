import asyncio
import os
from typing import List
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import litellm
from src.config import RewardConfig
from src.data.schema import CodeSnippet, HumanEvalProblem


SNIPPET_RELEVANCE_PROMPT = """You are a strict code relevance evaluator. Your job is to judge whether a code snippet would genuinely help solve a specific programming problem.

Problem:
{problem_prompt}

Code Snippet:
```python
{snippet_code}
```

Choose EXACTLY one score from this fixed set: 0.0, 0.3, 0.7, 1.0

Scoring rules (be strict — when in doubt, score lower):
- 1.0: The snippet directly implements the required algorithm or data structure. A programmer could adapt it with minimal changes.
- 0.7: The snippet demonstrates a clearly relevant technique or pattern (e.g., same algorithmic idea, same data structure used similarly). Useful but not directly applicable.
- 0.3: The snippet is in the same general domain (e.g., both deal with strings, both use recursion) but the core logic is different. Provides little concrete guidance.
- 0.0: The snippet is unrelated or uses a completely different approach. Retrieving it wastes context.

Common mistakes to avoid:
- Do NOT give 0.7 just because both snippets use Python or share basic syntax.
- Do NOT give 0.7 for superficial keyword overlap (e.g., both mention "list").
- Only give 1.0 if the algorithm itself matches, not just the topic.

Respond with ONLY one of: 0.0, 0.3, 0.7, 1.0"""


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
        text = ""
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
            if score not in (0.0, 0.3, 0.7, 1.0):
                raise ValueError(f"Unexpected score value: {score!r}")
            return score
        except Exception as e:
            raise RuntimeError(f"Relevance judge failed (response: {text!r}): {e}") from e

    async def score_batch_async(
        self, problem: HumanEvalProblem, snippets: List[CodeSnippet]
    ) -> List[float]:
        """Score snippets in serial chunks of max_concurrency."""
        results = []
        chunk_size = self.config.max_concurrency
        for i in range(0, len(snippets), chunk_size):
            chunk = snippets[i:i + chunk_size]
            chunk_results = await asyncio.gather(
                *[self.score_async(problem, s) for s in chunk]
            )
            results.extend(chunk_results)
        return results

    def score_batch(
        self, problem: HumanEvalProblem, snippets: List[CodeSnippet]
    ) -> List[float]:
        """Score relevance of multiple snippets concurrently."""
        if not snippets:
            return []
        return asyncio.run(self.score_batch_async(problem, snippets))

    async def _score_pairs_async(
        self, pairs: List[tuple]
    ) -> List[float]:
        """Score all (problem, snippet) pairs with semaphore-controlled concurrency."""
        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def _one(p, s):
            async with sem:
                return await self.score_async(p, s)

        return list(await asyncio.gather(*[_one(p, s) for p, s in pairs]))

    def score_pairs_batch(
        self, pairs: List[tuple]
    ) -> List[float]:
        """Score a flat list of (problem, snippet) pairs in one event loop.

        More efficient than calling score_batch() per problem because all pairs
        share a single asyncio event loop and a semaphore limits concurrency to
        max_concurrency, preventing LLM overload.
        """
        if not pairs:
            return []
        return asyncio.run(self._score_pairs_async(pairs))
