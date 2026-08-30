"""Rerun of the band pilot's B=2048 ablated rows (k=10) with the KV-safe engine."""
import time

from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, THINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems

BANDS = {"light": range(26, 31), "medium": range(24, 33), "heavy": range(22, 35)}


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
    solved = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in res)
    rem = (controller.stat_sum / controller.stat_n) if controller and controller.stat_n else 0
    print(f"[{label} B={budget}] {solved}/{len(res)} | rem_frac {rem:.3f} | "
          f"{time.time()-t0:.0f}s", flush=True)


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    problems = load_problems()
    hard = ([p for p in problems if p["dataset"] == "math" and p.get("level", 0) >= 4][:40]
            + [p for p in problems if p["dataset"] == "gsm8k"][:20])
    for name, band in BANDS.items():
        ctrl = AblationController(model, lens, band, k=10, mode="jspace")
        run(model, tok, hard, 2048, ctrl, f"jspace-{name}-k10")
    ctrl = AblationController(model, lens, BANDS["medium"], k=10, mode="random",
                              rand_seed=0)
    run(model, tok, hard, 2048, ctrl, "random-medium-k10")


if __name__ == "__main__":
    main()
