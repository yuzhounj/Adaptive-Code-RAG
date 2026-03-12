import pytest
from src.data.schema import HumanEvalProblem
from src.reward.executor import execute_solution, _build_test_script


def make_problem():
    return HumanEvalProblem(
        task_id="HumanEval/0",
        prompt="def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n",
        canonical_solution="    return a + b\n",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
        entry_point="add",
    )


def test_executor_correct_solution():
    problem = make_problem()
    reward = execute_solution(problem, "def add(a: int, b: int) -> int:\n    return a + b\n")
    assert reward == 1.0


def test_executor_wrong_solution():
    problem = make_problem()
    reward = execute_solution(problem, "def add(a: int, b: int) -> int:\n    return a - b\n")
    assert reward == 0.0


def test_executor_empty_solution():
    problem = make_problem()
    reward = execute_solution(problem, "")
    assert reward == 0.0


def test_executor_canonical_solution():
    problem = make_problem()
    # Canonical solution should always pass
    reward = execute_solution(
        problem,
        problem.prompt + problem.canonical_solution
    )
    assert reward == 1.0
