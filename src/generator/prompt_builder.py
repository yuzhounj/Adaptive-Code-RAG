from typing import List
from src.data.schema import HumanEvalProblem, CodeSnippet


def build_prompt(
    problem: HumanEvalProblem,
    snippets: List[CodeSnippet],
    max_snippet_chars: int = 1000,
) -> str:
    """Build LLM prompt from problem and retrieved code snippets."""
    lines = []

    if snippets:
        lines.append("# IMPORTANT: The following code examples are highly relevant reference implementations.")
        lines.append("# You MUST base your solution closely on these examples — reuse their logic, structure,")
        lines.append("# and edge-case handling (including boundary conditions). Do NOT ignore them.")
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
        lines.append("# Complete the following Python function by strictly following the reference examples above:")
    else:
        lines.append("# Complete the following Python function:")

    lines.append("```python")
    lines.append(problem.prompt)
    lines.append("```")
    lines.append("")
    lines.append("Output the complete Python function definition starting with `def`, strictly following the logic shown in the reference examples. Do NOT include any explanation, comments outside the function, or markdown formatting.")

    return "\n".join(lines)
