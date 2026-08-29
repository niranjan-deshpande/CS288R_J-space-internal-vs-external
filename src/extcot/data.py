"""Dataset loading and fixed-seed sampling.

Sources (stated per spec / report):
- GSM8K: openai/gsm8k, test split, n=150, seed 0.
- MATH: HuggingFaceH4/MATH-500, test split, n=250 stratified by level
  (50 per level, seed 0) — compute-bound fallback approved by user.
- AIME 2024: HuggingFaceH4/aime_2024 (all 30).
- MMLU (selectivity control): cais/mmlu 'all' test, n=500, seed 0.
"""
from __future__ import annotations

import random

from datasets import load_dataset

MMLU_INSTR = ("Answer the multiple-choice question. Put only the letter of "
              "the correct option (A, B, C, or D) within \\boxed{}.")
SST2_INSTR = ("Classify the sentiment of the following movie review sentence. "
              "Put only the word positive or negative within \\boxed{}.")


def _gsm8k_gold(ans: str) -> str:
    return ans.split("####")[-1].strip()


def load_problems() -> list[dict]:
    """Each problem: {id, dataset, question, gold, level (MATH only)}."""
    problems = []

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(0).sample(range(len(gsm)), 150)
    for i in sorted(idx):
        problems.append({
            "id": f"gsm8k-{i}", "dataset": "gsm8k",
            "question": gsm[i]["question"], "gold": _gsm8k_gold(gsm[i]["answer"]),
        })

    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_level: dict[int, list[int]] = {}
    for i in range(len(math)):
        by_level.setdefault(math[i]["level"], []).append(i)
    rng = random.Random(0)
    for level in sorted(by_level):
        pool = by_level[level]
        take = pool if len(pool) <= 50 else rng.sample(pool, 50)
        for i in sorted(take):
            problems.append({
                "id": f"math-{i}", "dataset": "math",
                "question": math[i]["problem"], "gold": str(math[i]["answer"]),
                "level": level,
            })

    aime = load_dataset("HuggingFaceH4/aime_2024", split="train")
    for i in range(len(aime)):
        problems.append({
            "id": f"aime-{i}", "dataset": "aime",
            "question": aime[i]["problem"], "gold": str(aime[i]["answer"]),
        })

    return problems


def load_sst2(n: int = 300) -> list[dict]:
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    idx = random.Random(0).sample(range(len(sst)), n)
    return [{"id": f"sst2-{i}", "dataset": "sst2",
             "question": sst[i]["sentence"].strip(),
             "gold": "positive" if sst[i]["label"] == 1 else "negative"}
            for i in sorted(idx)]


def load_mmlu(n: int = 500) -> list[dict]:
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(0).sample(range(len(mmlu)), n)
    out = []
    for i in sorted(idx):
        row = mmlu[i]
        opts = "\n".join(f"{letter}. {choice}" for letter, choice in
                         zip("ABCD", row["choices"]))
        out.append({
            "id": f"mmlu-{i}", "dataset": "mmlu",
            "question": f"{row['question']}\n\n{opts}",
            "gold": "ABCD"[row["answer"]],
        })
    return out
