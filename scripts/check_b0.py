"""Go/no-go: does J-space ablation impair direct answering (B=0)?
Clean vs jspace vs random at two band widths, GSM8K + MATH slices."""
import time

import torch
from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, NOTHINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems


def run_b0(model, tok, probs, controller, label):
    reqs = [GenRequest(prompt=build_prompt(tok, p["question"], thinking=False),
                       budget=0, seed=hash((p["id"], "b0")) % 2**31, thinking=False,
                       max_answer_tokens=512,
                       meta={"id": p["id"], "gold": p["gold"], "ds": p["dataset"]})
            for p in probs]
    t0 = time.time()
    res, _ = generate_batch(model, tok, reqs, controller=controller,
                         sampling=NOTHINK_SAMPLING)
    dt = time.time() - t0
    solved = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in res)
    toks = sum(r.answer_tokens for r in res)
    print(f"[{label}] {solved}/{len(res)} solved | {toks} tok, {toks/dt:.0f} tok/s",
          flush=True)
    return res


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    problems = load_problems()
    gsm = [p for p in problems if p["dataset"] == "gsm8k"][:48]
    math = [p for p in problems if p["dataset"] == "math"][:48]

    bands = {"medium(19-28)": list(range(19, 29)), "heavy(14-33)": list(range(14, 34))}
    for name, probs in [("gsm8k", gsm), ("math", math)]:
        run_b0(model, tok, probs, None, f"{name} clean")
        for bname, band in bands.items():
            ctrl = AblationController(model, lens, band, k=10, mode="jspace")
            run_b0(model, tok, probs, ctrl, f"{name} jspace {bname}")
        ctrl = AblationController(model, lens, bands["medium(19-28)"], k=10,
                                  mode="random", rand_seed=0)
        run_b0(model, tok, probs, ctrl, f"{name} random medium")


if __name__ == "__main__":
    main()
