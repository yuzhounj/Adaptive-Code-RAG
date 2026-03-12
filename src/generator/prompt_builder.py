from typing import List
from src.data.schema import HumanEvalProblem, CodeSnippet


def build_prompt(
    problem: HumanEvalProblem,
    snippets: List[CodeSnippet],
    max_snippet_chars: int = 300,
) -> str:
    """Build LLM prompt from problem and retrieved code snippets."""
    lines = []

    if snippets:
        lines.append("# Relevant code examples:")
        for i, snippet in enumerate(snippets, 1):
            code_preview = snippet.code[:max_snippet_chars]
            if len(snippet.code) > max_snippet_chars:
                code_preview += "..."
            doc_preview = snippet.docstring[:100] if snippet.docstring else ""
            lines.append(f"\n## Example {i}:")
            if doc_preview:
                lines.append(f"# {doc_preview}")
            lines.append("```python")
            lines.append(code_preview)
            lines.append("```")
        lines.append("")

    lines.append("# Complete the following Python function:")
    lines.append("```python")
    lines.append(problem.prompt)
    lines.append("```")
    lines.append("")
    lines.append("Provide only the function body (the implementation), without repeating the function signature or docstring. Output raw Python code only.")

    return "\n".join(lines)
