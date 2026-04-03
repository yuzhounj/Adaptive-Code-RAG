from typing import List
from src.data.schema import HumanEvalProblem, CodeSnippet

def build_prompt(
        problem: HumanEvalProblem,
        snippets: List[CodeSnippet],
        max_snippet_chars: int = 1000,
) -> str:
    """Build LLM prompt from problem and multiple retrieved code snippets."""
    lines = []

    if snippets:
        lines.append("## Reference Implementations")
        lines.append("The following are retrieved implementations that solve closely related problems.")
        lines.append("You MUST study their logic carefully and adapt relevant parts to solve the problem below.")
        lines.append(
            "Do NOT invent a totally different approach — reuse the algorithms, structures, or edge-case handling shown here where appropriate.")
        lines.append("")

        for i, snippet in enumerate(snippets, 1):
            ref_code = snippet.code.strip()
            if len(ref_code) > max_snippet_chars:
                ref_code = ref_code[:max_snippet_chars] + "..."

            lines.append(f"### Reference {i}")
            lines.append("```python")
            lines.append(ref_code)
            lines.append("```")
            lines.append("")

        lines.append("## Your Task")
        lines.append("Synthesize and adapt the reference implementations above to complete the following function:")
    else:
        lines.append("## Your Task")
        lines.append("Complete the following Python function:")

    lines.append("```python")
    lines.append(problem.prompt)
    lines.append("```")
    lines.append("")
    lines.append(
        "Output ONLY the complete Python function definition starting with `def`. No explanation, no markdown, no extra text.")

    return "\n".join(lines)