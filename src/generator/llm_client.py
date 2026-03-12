import asyncio
import os
from typing import List
import litellm
from src.config import GeneratorConfig


class LLMClient:
    """Multi-provider LLM client via litellm."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "") or None

    async def generate_async(self, prompt: str, n: int = 1) -> List[str]:
        """Generate n completions via n parallel calls for universal provider support."""
        try:
            tasks = [
                litellm.acompletion(
                    model=self.config.model,
                    api_key=self._api_key,
                    messages=[
                        {"role": "system", "content": "You are an expert Python programmer. Complete the given function."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                for _ in range(n)
            ]
            responses = await asyncio.gather(*tasks)
            return [r.choices[0].message.content or "" for r in responses]
        except Exception as e:
            print(f"LLM generation error: {e}")
            return ["" for _ in range(n)]

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Synchronous wrapper around async generation."""
        return asyncio.run(self.generate_async(prompt, n=n))

    async def generate_batch_async(self, prompts: List[str], n: int = 1) -> List[List[str]]:
        """Generate completions for a batch of prompts concurrently."""
        tasks = [self.generate_async(prompt, n=n) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def generate_batch(self, prompts: List[str], n: int = 1) -> List[List[str]]:
        """Synchronous batch generation."""
        return asyncio.run(self.generate_batch_async(prompts, n=n))
