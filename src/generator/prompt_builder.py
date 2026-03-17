from typing import List
from src.data.schema import HumanEvalProblem, CodeSnippet


def build_prompt(
    problem: HumanEvalProblem,
    snippets: List[CodeSnippet],
    max_snippet_chars: int = 1000,
) -> str:
    """Build LLM prompt from problem and a single retrieved code snippet."""
    lines = []

    if snippets:
        snippet = snippets[0]
        # Show only the code part as the reference — docstring often repeats the problem
        # and distracts from the actual implementation to borrow.
        ref_code = snippet.code.strip()
        if len(ref_code) > max_snippet_chars:
            ref_code = ref_code[:max_snippet_chars] + "..."

        lines.append("## Reference Implementation")
        lines.append("The following is a retrieved implementation that solves a closely related problem.")
        lines.append("You MUST study its logic carefully and adapt it directly to solve the problem below.")
        lines.append("Do NOT invent a different approach — reuse the algorithm, structure, and edge-case handling shown here.")
        lines.append("```python")
        lines.append(ref_code)
        lines.append("```")
        lines.append("")
        lines.append("## Your Task")
        lines.append("Adapt the reference implementation above to complete the following function:")
    else:
        lines.append("## Your Task")
        lines.append("Complete the following Python function:")

    lines.append("```python")
    lines.append(problem.prompt)
    lines.append("```")
    lines.append("")
    lines.append("Output ONLY the complete Python function definition starting with `def`. No explanation, no markdown, no extra text.")

    return "\n".join(lines)
