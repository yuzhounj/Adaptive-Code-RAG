import asyncio
import os
from typing import List, Optional
import openai
from src.config import GeneratorConfig


class LLMClient:
    """Async OpenAI client for code generation."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            timeout=config.timeout,
        )

    async def generate_async(self, prompt: str, n: int = 1) -> List[str]:
        """Generate n code completions asynchronously."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer. Complete the given function."},
                    {"role": "user", "content": prompt},
                ],
                n=n,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return [choice.message.content or "" for choice in response.choices]
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
