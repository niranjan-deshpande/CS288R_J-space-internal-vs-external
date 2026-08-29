"""Answer extraction and grading.

GSM8K: numeric exact match after normalization.
MATH / AIME: math-verify (sympy-based equivalence), falling back to
normalized string match.
"""
from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces)."""
    i = text.rfind("\\boxed")
    if i == -1:
        return None
    j = text.find("{", i)
    if j == -1:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _norm_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def grade_gsm8k(answer_text: str, gold: str) -> bool:
    pred = extract_boxed(answer_text)
    if pred is None:
        return False
    g = _norm_number(gold)
    p = _norm_number(pred)
    if g is None or p is None:
        return False
    return abs(p - g) < 1e-6 * max(1.0, abs(g))


_MV = None


def _math_verify():
    global _MV
    if _MV is None:
        from math_verify import parse, verify
        _MV = (parse, verify)
    return _MV


def grade_math(answer_text: str, gold: str) -> bool:
    pred = extract_boxed(answer_text)
    if pred is None:
        return False
    try:
        parse, verify = _math_verify()
        gold_p = parse(f"\\boxed{{{gold}}}")
        pred_p = parse(f"\\boxed{{{pred}}}")
        if verify(gold_p, pred_p):
            return True
    except Exception:
        pass
    # fallback: normalized string equality
    norm = lambda s: re.sub(r"\s+|\\left|\\right|\\!|\\,", "", s).strip("$ ")
    return norm(pred) == norm(gold)


def grade(dataset: str, answer_text: str, gold: str) -> bool:
    if dataset == "gsm8k":
        return grade_gsm8k(answer_text, gold)
    return grade_math(answer_text, gold)
