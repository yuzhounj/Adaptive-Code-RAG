from datasets import load_dataset
from typing import List, Tuple
from src.data.schema import HumanEvalProblem


def load_humaneval(cache_dir: str = "data/humaneval") -> List[HumanEvalProblem]:
    """Load all 164 HumanEval problems."""
    ds = load_dataset("openai_humaneval", split="test", cache_dir=cache_dir)
    problems = []
    for item in ds:
        problems.append(HumanEvalProblem(
            task_id=item["task_id"],
            prompt=item["prompt"],
            canonical_solution=item["canonical_solution"],
            test=item["test"],
            entry_point=item["entry_point"],
        ))
    return problems


def split_humaneval(problems: List[HumanEvalProblem], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[HumanEvalProblem], List[HumanEvalProblem]]:
    """Split into train/eval sets with random shuffle for balanced difficulty distribution."""
    import random
    shuffled = problems.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]
