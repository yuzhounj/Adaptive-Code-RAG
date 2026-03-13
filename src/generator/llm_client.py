import asyncio
from typing import List
import litellm
from src.config import GeneratorConfig

_FENCE = "```"
SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given Python function.\n"
    "Output ONLY the complete Python function definition wrapped in a ```python code block.\n"
    "No comments, no docstrings in the body, no explanation, no extra text outside the code block.\n"
    "You must strictly adhere to the format of the example below.\n"
    "\n"
    "Example:\n"
    "User:\n"
    "def add(a: int, b: int) -> int:\n"
    '    """Return the sum of a and b."""\n'
    "\n"
    "Assistant:\n"
    f"{_FENCE}python\n"
    "def add(a: int, b: int) -> int:\n"
    "    return a + b\n"
    f"{_FENCE}\n"
)


def _strip_markdown_code_block(text: str) -> str:
    """Extract code from a markdown ```python ... ``` block, if present."""
    text = text.strip()
    if text.startswith(_FENCE):
        first_newline = text.find("\n")
        if first_newline == -1:
            return text
        text = text[first_newline + 1:]
        if text.endswith(_FENCE):
            text = text[: text.rfind(_FENCE)]
    return text.strip()


class LLMClient:
    """Multi-provider LLM client via litellm."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._api_base = config.api_base or None

    async def generate_async(self, prompt: str, n: int = 1) -> List[str]:
        """Generate n completions via n parallel calls for universal provider support."""
        try:
            tasks = [
                litellm.acompletion(
                    model=self.config.model,
                    api_base=self._api_base,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                for _ in range(n)
            ]
            responses = await asyncio.gather(*tasks)
            return [_strip_markdown_code_block(r.choices[0].message.content or "") for r in responses]
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
