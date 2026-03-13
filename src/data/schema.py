from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


@dataclass
class CodeSnippet:
    snippet_id: str
    code: str
    docstring: str
    language: str = "python"
    corpus_idx: int = -1  # index in the FAISS corpus


@dataclass
class RetrievedContext:
    problem: HumanEvalProblem
    snippets: List[CodeSnippet]
    log_probs: Optional[torch.Tensor] = None  # shape [k], gradient-attached
    scores: Optional[torch.Tensor] = None     # raw similarity scores
