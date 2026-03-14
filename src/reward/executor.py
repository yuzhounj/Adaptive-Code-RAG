import logging
import os
import re
import subprocess
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from src.data.schema import CodeSnippet, HumanEvalProblem

logger = logging.getLogger(__name__)

_FAIL_LOG_DIR = "outputs/fail_cases"


def _write_fail_log(
    problem: HumanEvalProblem,
    generated_code: str,
    stderr: str,
    tag: str,
    snippets: Optional[List[CodeSnippet]] = None,
) -> None:
    os.makedirs(_FAIL_LOG_DIR, exist_ok=True)
    task_id_safe = problem.task_id.replace("/", "_")
    filename = f"{tag}_{task_id_safe}_{int(time.time() * 1000)}.md"
    filepath = os.path.join(_FAIL_LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# [{tag}] {problem.task_id}\n\n")
        f.write(f"## Problem Prompt\n\n```python\n{problem.prompt}\n```\n\n")
        f.write(f"## Canonical Solution\n\n```python\n{problem.canonical_solution}\n```\n\n")
        if snippets:
            f.write(f"## RAG Retrieved Snippets ({len(snippets)} total)\n\n")
            for i, snippet in enumerate(snippets, 1):
                f.write(f"### Snippet {i}: `{snippet.snippet_id}`\n\n")
                if snippet.docstring:
                    f.write(f"{snippet.docstring}\n\n")
                f.write(f"```python\n{snippet.code}\n```\n\n")
        f.write(f"## Generated Code\n\n```python\n{generated_code}\n```\n\n")
        f.write(f"## Test\n\n```python\n{problem.test}\n```\n\n")
        if stderr:
            f.write(f"## Stderr\n\n```\n{stderr}\n```\n\n")
        f.write("---\n\n")


def execute_solution(
    problem: HumanEvalProblem,
    generated_code: str,
    timeout: int = 10,
    snippets: Optional[List[CodeSnippet]] = None,
) -> float:
    """
    Execute generated code against HumanEval test cases.
    Returns 1.0 on pass, 0.0 on failure/timeout.
    Uses subprocess to avoid exec() security issues.
    """
    # Build the complete test script
    # The generated_code should be the function implementation
    # We prepend the function signature from the prompt
    full_code = _build_test_script(problem, generated_code)

    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return 1.0
        _write_fail_log(problem, generated_code, result.stderr.strip(), "FAIL", snippets)
        return 0.0
    except subprocess.TimeoutExpired:
        _write_fail_log(problem, generated_code, "", "TIMEOUT", snippets)
        return 0.0
    except Exception:
        return 0.0


def _build_test_script(problem: HumanEvalProblem, generated_code: str) -> str:
    """Build complete executable Python script for testing."""
    stripped = generated_code.strip()

    # Detect whether the generated code contains a top-level def for the target function.
    # This handles both "def foo():" and "from x import y\ndef foo():" patterns.
    has_toplevel_def = bool(re.search(
        rf'^def {re.escape(problem.entry_point)}\b', stripped, re.MULTILINE
    ))

    if has_toplevel_def:
        # Complete implementation — prepend prompt to preserve helper functions
        function_code = problem.prompt.rstrip() + "\n\n" + stripped
    else:
        # Only the function body was generated — indent and attach to prompt signature
        indented_body = textwrap.indent(textwrap.dedent(generated_code), "    ")
        function_code = problem.prompt.rstrip() + "\n" + indented_body
    # Prepend any imports from the prompt that the body may need
    import_lines = "\n".join(
        line.strip() for line in problem.prompt.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    )
    if import_lines:
        function_code = import_lines + "\n\n" + function_code

    test_call = f"\ncheck({problem.entry_point})"
    full_code = function_code + "\n\n" + problem.test + test_call
    return full_code


def batch_execute(
    problem: HumanEvalProblem,
    generated_codes: list,
    timeout: int = 10,
    snippets: Optional[List[CodeSnippet]] = None,
) -> list:
    """Execute multiple generated solutions concurrently, return list of rewards."""
    with ThreadPoolExecutor(max_workers=len(generated_codes)) as executor:
        futures = [
            executor.submit(execute_solution, problem, code, timeout, snippets)
            for code in generated_codes
        ]
        return [f.result() for f in futures]
