import subprocess
import textwrap
from typing import Optional
from src.data.schema import HumanEvalProblem


def execute_solution(
    problem: HumanEvalProblem,
    generated_code: str,
    timeout: int = 10,
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
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


def _build_test_script(problem: HumanEvalProblem, generated_code: str) -> str:
    """Build complete executable Python script for testing."""
    # The prompt contains the function signature + docstring
    # generated_code should be the implementation body

    # Try to assemble: if generated_code starts with "def", use it directly
    # Otherwise, combine prompt (signature) + generated_code (body)
    if generated_code.strip().startswith("def "):
        function_code = generated_code
    else:
        # generated_code is just the body, need to indent it
        indented_body = textwrap.indent(generated_code, "    ")
        # Remove trailing whitespace/newlines from prompt before adding body
        function_code = problem.prompt.rstrip() + "\n" + indented_body

    test_call = f"\ncheck({problem.entry_point})"
    full_code = function_code + "\n\n" + problem.test + test_call
    return full_code


def batch_execute(
    problem: HumanEvalProblem,
    generated_codes: list,
    timeout: int = 10,
) -> list:
    """Execute multiple generated solutions, return list of rewards."""
    return [execute_solution(problem, code, timeout=timeout) for code in generated_codes]
