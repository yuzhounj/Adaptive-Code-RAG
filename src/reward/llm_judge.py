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

Score the relevance on a continuous scale from 0.0 to 1.0 (one decimal place).

Scoring guidelines (be strict — when in doubt, score lower):
- 0.9–1.0: Directly implements the required algorithm or data structure. A programmer could adapt it with minimal changes.
- 0.6–0.8: Demonstrates a clearly relevant technique or pattern (same algorithmic idea, same data structure used similarly). Useful but not directly applicable.
- 0.3–0.5: Same general domain (e.g., both deal with strings, both use recursion) but core logic differs. Provides little concrete guidance.
- 0.0–0.2: Unrelated or uses a completely different approach. Retrieving it wastes context.

Common mistakes to avoid:
- Do NOT score above 0.5 just because both use Python or share basic syntax.
- Do NOT score above 0.5 for superficial keyword overlap (e.g., both mention "list").
- Only score above 0.8 if the algorithm itself matches, not just the topic.

Respond with ONLY a single float between 0.0 and 1.0, e.g. 0.7"""


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
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score out of range [0, 1]: {score!r}")
            return round(score, 1)
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
        from tqdm.asyncio import tqdm
        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def _one(p, s):
            async with sem:
                return await self.score_async(p, s)

        return list(await tqdm.gather(
            *[_one(p, s) for p, s in pairs],
            desc="Eval-judge",
            leave=False,
            total=len(pairs),
        ))

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
