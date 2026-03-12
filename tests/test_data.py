import pytest
from unittest.mock import patch, MagicMock
from src.data.schema import HumanEvalProblem, CodeSnippet, RetrievedContext
import torch


def make_problem():
    return HumanEvalProblem(
        task_id="HumanEval/0",
        prompt="def add(a, b):\n    ",
        canonical_solution="    return a + b\n",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add",
    )


def make_snippet():
    return CodeSnippet(
        snippet_id="test_0",
        code="def add(a, b):\n    return a + b",
        docstring="Add two numbers",
        language="python",
        corpus_idx=0,
    )


def test_humaneval_problem_creation():
    p = make_problem()
    assert p.task_id == "HumanEval/0"
    assert p.entry_point == "add"


def test_code_snippet_creation():
    s = make_snippet()
    assert s.language == "python"
    assert s.corpus_idx == 0


def test_retrieved_context_with_log_probs():
    problem = make_problem()
    snippets = [make_snippet()]
    log_probs = torch.tensor([-0.5, -1.2, -0.8])

    context = RetrievedContext(
        problem=problem,
        snippets=snippets,
        log_probs=log_probs,
    )
    assert context.log_probs is not None
    assert context.log_probs.shape == (3,)
