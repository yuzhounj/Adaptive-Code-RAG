"""Load CodeSearchNet dataset from HuggingFace as HumanEvalProblem instances."""
from typing import List, Optional

from src.data.schema import HumanEvalProblem


def load_codesearchnet(
    language: str = "python",
    max_samples: Optional[int] = 10000,
    cache_dir: Optional[str] = None,
) -> List[HumanEvalProblem]:
    """Load CodeSearchNet and convert to HumanEvalProblem schema.

    Maps:
      func_documentation_string -> prompt
      func_code_string          -> canonical_solution
      func_name                 -> entry_point
      test=""                   (no test cases)
    """
    from datasets import load_dataset

    ds = load_dataset("code_search_net", language, split="train", cache_dir=cache_dir, trust_remote_code=True)

    # Filter empty docstrings / code
    ds = ds.filter(
        lambda ex: bool(ex["func_documentation_string"].strip()) and bool(ex["func_code_string"].strip())
    )

    # Shuffle with fixed seed before truncating
    ds = ds.shuffle(seed=42)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    problems = []
    for i, ex in enumerate(ds):
        problems.append(HumanEvalProblem(
            task_id=f"CodeSearchNet/{i}",
            prompt=ex["func_documentation_string"],
            canonical_solution=ex["func_code_string"],
            entry_point=ex["func_name"],
            test="",
        ))

    return problems
