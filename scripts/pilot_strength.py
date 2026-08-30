"""Strength escalation: does raising k create dynamic range on math+CoT?
Also NOEX probe (exemption's role in CoT mode) and norm-matched random at
high k. Same 60-problem slice as pilot_bands."""
import time

from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, THINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems

HEAVY = range(22, 35)
MEDIUM = range(24, 33)


def run(model, tok, probs, budget, controller, label, batch=64):
    reqs = [GenRequest(prompt=build_prompt(tok, p["question"], thinking=True),
                       budget=budget, seed=hash((p["id"], "pilot")) % 2**31,
                       meta={"id": p["id"], "gold": p["gold"], "ds": p["dataset"]})
            for p in probs]
    t0 = time.time()
    res = []
    todo = reqs
    b = batch
    while todo:
        nxt = []
        for s in range(0, len(todo), b):
            r_, conts = generate_batch(model, tok, todo[s:s + b],
                                       controller=controller,
                                       sampling=THINK_SAMPLING, kv_budget_gb=55.0)
            res += [r for r in r_ if r is not None]
            nxt += conts
        todo = nxt
        b = max(2, b // 2)
    dt = time.time() - t0
    solved = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in res)
    rem = (controller.stat_sum / controller.stat_n) if controller and controller.stat_n else 0
    print(f"[{label} B={budget}] {solved}/{len(res)} | rem_frac {rem:.3f} | "
          f"{dt:.0f}s", flush=True)


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    problems = load_problems()
    hard = ([p for p in problems if p["dataset"] == "math" and p.get("level", 0) >= 4][:40]
            + [p for p in problems if p["dataset"] == "gsm8k"][:20])

    conds = [
        ("heavy-k25", dict(band=HEAVY, k=25, mode="jspace")),
        ("heavy-k50", dict(band=HEAVY, k=50, mode="jspace")),
        ("heavy-k100", dict(band=HEAVY, k=100, mode="jspace")),
        ("heavy-k10-NOEX", dict(band=HEAVY, k=10, mode="jspace", use_exemption=False)),
        ("random-heavy-k50", dict(band=HEAVY, k=50, mode="random", rand_seed=0)),
        ("medium-k50", dict(band=MEDIUM, k=50, mode="jspace")),
    ]
    for budget in (2048, 256):
        for label, kw in conds:
            ctrl = AblationController(model, lens, **kw)
            run(model, tok, hard, budget, ctrl, label)


if __name__ == "__main__":
    main()
