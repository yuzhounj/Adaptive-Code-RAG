"""Validation script for SnippetRelevanceJudge.

Constructs 30 contrast cases (relevant vs irrelevant snippet) and prints
scores. Marks [PASS] if relevant score > irrelevant score, [FAIL] otherwise.

Cases are grouped into three categories so that the irrelevant snippet is
always from a completely different domain:
  - Math/number problems  → irrelevant = string-manipulation snippet
  - String problems       → irrelevant = math/number snippet
  - List/array problems   → irrelevant = string or geometry snippet

Usage:
    ollama serve  # ensure Ollama is running
    python scripts/check_relevance_judge.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.humaneval_loader import load_humaneval
from src.data.corpus_builder import load_corpus_metadata
from src.reward.llm_judge import SnippetRelevanceJudge

# (problem_id, relevant_snippet_id, irrelevant_snippet_id)
# Irrelevant snippet is always from a different algorithmic domain.
CASES = [
    # ── Math / number-theory problems → irrelevant = string snippet ──────────
    ("HumanEval/13",  "HumanEval/13",  "HumanEval/27"),   # gcd           vs flip_case
    ("HumanEval/25",  "HumanEval/25",  "HumanEval/86"),   # factorize     vs anti_shuffle
    ("HumanEval/31",  "HumanEval/31",  "HumanEval/140"),  # is_prime      vs fix_spaces
    ("HumanEval/44",  "HumanEval/44",  "HumanEval/51"),   # change_base   vs remove_vowels
    ("HumanEval/55",  "HumanEval/55",  "HumanEval/64"),   # fib           vs vowels_count
    ("HumanEval/60",  "HumanEval/60",  "HumanEval/124"),  # sum_to_n      vs valid_date
    ("HumanEval/62",  "HumanEval/62",  "HumanEval/80"),   # derivative    vs is_happy
    ("HumanEval/99",  "HumanEval/99",  "HumanEval/91"),   # closest_int   vs is_bored
    ("HumanEval/139", "HumanEval/139", "HumanEval/141"),  # special_fact  vs file_name_check
    ("HumanEval/157", "HumanEval/157", "HumanEval/17"),   # right_angle   vs parse_music

    # ── String problems → irrelevant = math / number snippet ─────────────────
    ("HumanEval/10",  "HumanEval/10",  "HumanEval/25"),   # make_palindrome  vs factorize
    ("HumanEval/27",  "HumanEval/27",  "HumanEval/60"),   # flip_case        vs sum_to_n
    ("HumanEval/38",  "HumanEval/38",  "HumanEval/31"),   # decode_cyclic    vs is_prime
    ("HumanEval/48",  "HumanEval/48",  "HumanEval/76"),   # is_palindrome    vs is_simple_power
    ("HumanEval/51",  "HumanEval/51",  "HumanEval/44"),   # remove_vowels    vs change_base
    ("HumanEval/65",  "HumanEval/65",  "HumanEval/63"),   # circular_shift   vs fibfib
    ("HumanEval/80",  "HumanEval/80",  "HumanEval/139"),  # is_happy         vs special_factorial
    ("HumanEval/89",  "HumanEval/89",  "HumanEval/97"),   # encrypt          vs multiply
    ("HumanEval/124", "HumanEval/124", "HumanEval/77"),   # valid_date       vs iscube
    ("HumanEval/162", "HumanEval/162", "HumanEval/157"),  # string_to_md5    vs right_angle_triangle

    # ── List / array problems → irrelevant = string or geometry snippet ───────
    ("HumanEval/0",   "HumanEval/0",   "HumanEval/148"),  # has_close_elements vs bf (planets)
    ("HumanEval/3",   "HumanEval/3",   "HumanEval/86"),   # below_zero         vs anti_shuffle
    ("HumanEval/5",   "HumanEval/5",   "HumanEval/36"),   # intersperse        vs fizz_buzz
    ("HumanEval/9",   "HumanEval/9",   "HumanEval/156"),  # rolling_max        vs roman_encode
    ("HumanEval/20",  "HumanEval/20",  "HumanEval/143"),  # find_closest_elems vs words_in_sentence
    ("HumanEval/57",  "HumanEval/57",  "HumanEval/162"),  # monotonic          vs string_to_md5
    ("HumanEval/109", "HumanEval/109", "HumanEval/45"),   # move_one_ball      vs triangle_area
    ("HumanEval/114", "HumanEval/114", "HumanEval/27"),   # minSubArraySum     vs flip_case
    ("HumanEval/126", "HumanEval/126", "HumanEval/48"),   # is_sorted          vs is_palindrome
    ("HumanEval/129", "HumanEval/129", "HumanEval/51"),   # minPath (2D grid)  vs remove_vowels
]


def main():
    config = load_config(str(ROOT / "configs" / "default.yaml"))

    print("Loading HumanEval problems...")
    problems = load_humaneval(cache_dir=str(ROOT / config.data.cache_dir))
    problem_map = {p.task_id: p for p in problems}

    print("Loading corpus snippets...")
    snippets = load_corpus_metadata(str(ROOT / config.data.corpus_dir))
    snippet_map = {s.snippet_id: s for s in snippets}

    judge = SnippetRelevanceJudge(config.reward)

    group_size = 10  # cases per category group
    group_labels = [
        "Math/number  → irrelevant = string",
        "String       → irrelevant = math",
        "List/array   → irrelevant = string/geometry",
    ]

    print("\n" + "=" * 65)
    passed = 0
    group_passed = [0] * len(group_labels)

    for idx, (problem_id, relevant_id, irrelevant_id) in enumerate(CASES):
        group = idx // group_size
        if idx % group_size == 0:
            print(f"\n── Group {group + 1}: {group_labels[group]} ──")

        problem = problem_map.get(problem_id)
        relevant = snippet_map.get(relevant_id)
        irrelevant = snippet_map.get(irrelevant_id)

        if problem is None or relevant is None or irrelevant is None:
            missing = problem_id if problem is None else (relevant_id if relevant is None else irrelevant_id)
            print(f"  [SKIP] {problem_id}  — {missing} not found in corpus")
            continue

        scores = judge.score_batch(problem, [relevant, irrelevant])
        rel_score, irrel_score = scores[0], scores[1]
        ok = rel_score > irrel_score
        tag = "[PASS]" if ok else "[FAIL]"
        if ok:
            passed += 1
            group_passed[group] += 1

        print(
            f"  {tag}  {problem_id:<18s}"
            f"  rel={rel_score:.2f}  irrel({irrelevant_id})={irrel_score:.2f}"
            f"  diff={rel_score - irrel_score:+.2f}"
        )

    print("\n" + "=" * 65)
    for i, label in enumerate(group_labels):
        print(f"  Group {i + 1} ({label}): {group_passed[i]}/{group_size}")
    print(f"\nTotal: {passed}/{len(CASES)} cases passed")


if __name__ == "__main__":
    main()
