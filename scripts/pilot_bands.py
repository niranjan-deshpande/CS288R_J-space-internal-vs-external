"""Band-strength pilot in thinking mode: clean vs light/medium/heavy J-space
ablation vs random, at budgets 256 and 2048, on a hard-ish 60-problem slice.
Also logs think-token usage (verbosity confound telemetry)."""
import time

from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, THINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems

BANDS = {
    "light": range(26, 31),    # 5 layers
    "medium": range(24, 33),   # 9 layers
    "heavy": range(22, 35),    # 13 layers = full workspace band
}


def run(model, tok, probs, budget, controller, label, batch=32):
    reqs = [GenRequest(prompt=build_prompt(tok, p["question"], thinking=True),
                       budget=budget, seed=hash((p["id"], "pilot")) % 2**31,
                       meta={"id": p["id"], "gold": p["gold"], "ds": p["dataset"]})
            for p in probs]
    t0 = time.time()
    res = []
    for s in range(0, len(reqs), batch):
        res += generate_batch(model, tok, reqs[s:s + batch], controller=controller,
                              sampling=THINK_SAMPLING)
    dt = time.time() - t0
    solved = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in res)
    toks = sum(r.think_tokens for r in res)
    trunc = sum(r.think_truncated for r in res)
    print(f"[{label} B={budget}] {solved}/{len(res)} | think tok mean "
          f"{toks/len(res):.0f} | trunc {trunc} | {dt:.0f}s {toks/dt:.0f} tok/s",
          flush=True)


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    problems = load_problems()
    hard = ([p for p in problems if p["dataset"] == "math" and p.get("level", 0) >= 4][:40]
            + [p for p in problems if p["dataset"] == "gsm8k"][:20])

    for budget in (256, 2048):
        run(model, tok, hard, budget, None, "clean")
        for name, band in BANDS.items():
            ctrl = AblationController(model, lens, band, k=10, mode="jspace")
            run(model, tok, hard, budget, ctrl, f"jspace-{name}")
        ctrl = AblationController(model, lens, BANDS["medium"], k=10, mode="random",
                                  rand_seed=0)
        run(model, tok, hard, budget, ctrl, "random-medium")


if __name__ == "__main__":
    main()
