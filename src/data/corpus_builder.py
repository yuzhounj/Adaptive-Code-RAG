import json
from pathlib import Path
from typing import List

from src.data.schema import CodeSnippet, HumanEvalProblem


def load_humaneval_corpus(
    problems: List[HumanEvalProblem],
) -> List[CodeSnippet]:
    """Build retrieval corpus from HumanEval canonical solutions."""
    snippets = []
    for i, problem in enumerate(problems):
        snippets.append(CodeSnippet(
            snippet_id=problem.task_id,
            code=problem.canonical_solution,
            docstring=problem.prompt,
            language="python",
            corpus_idx=i,
        ))
    return snippets


def save_corpus_metadata(snippets: List[CodeSnippet], output_dir: str) -> None:
    """Save corpus snippets metadata to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta_path = Path(output_dir) / "corpus_meta.json"
    data = [
        {
            "snippet_id": s.snippet_id,
            "code": s.code,
            "docstring": s.docstring,
            "language": s.language,
            "corpus_idx": s.corpus_idx,
        }
        for s in snippets
    ]
    with open(meta_path, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} snippets to {meta_path}")


def load_corpus_metadata(corpus_dir: str) -> List[CodeSnippet]:
    """Load corpus snippets from saved JSON."""
    meta_path = Path(corpus_dir) / "corpus_meta.json"
    with open(meta_path) as f:
        data = json.load(f)
    return [
        CodeSnippet(
            snippet_id=d["snippet_id"],
            code=d["code"],
            docstring=d["docstring"],
            language=d["language"],
            corpus_idx=d["corpus_idx"],
        )
        for d in data
    ]
