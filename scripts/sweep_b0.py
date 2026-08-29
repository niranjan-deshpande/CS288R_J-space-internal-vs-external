"""Band x k x exemption mini-sweep at B=0: where does J-space ablation bite?"""
import time

from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, NOTHINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems


def run_b0(model, tok, probs, controller, label):
    reqs = [GenRequest(prompt=build_prompt(tok, p["question"], thinking=False),
                       budget=0, seed=hash((p["id"], "b0")) % 2**31, thinking=False,
                       meta={"id": p["id"], "gold": p["gold"], "ds": p["dataset"]})
            for p in probs]
    t0 = time.time()
    res, _ = generate_batch(model, tok, reqs, controller=controller,
                         sampling=NOTHINK_SAMPLING)
    solved = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in res)
    print(f"[{label}] {solved}/{len(res)} ({time.time()-t0:.0f}s)", flush=True)


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    problems = load_problems()
    gsm = [p for p in problems if p["dataset"] == "gsm8k"][:48]
    math = [p for p in problems if p["dataset"] == "math"][:48]

    conds = [
        ("jspace 22-31 k10", dict(band=range(22, 32), k=10, mode="jspace")),
        ("jspace 25-34 k10", dict(band=range(25, 35), k=10, mode="jspace")),
        ("jspace 19-28 k25", dict(band=range(19, 29), k=25, mode="jspace")),
        ("jspace 22-31 k25", dict(band=range(22, 32), k=25, mode="jspace")),
        ("jspace 25-34 k25", dict(band=range(25, 35), k=25, mode="jspace")),
        ("jspace 19-28 k10 NOEX", dict(band=range(19, 29), k=10, mode="jspace", use_exemption=False)),
        ("jspace 25-34 k10 NOEX", dict(band=range(25, 35), k=10, mode="jspace", use_exemption=False)),
        ("random 25-34 k25", dict(band=range(25, 35), k=25, mode="random", rand_seed=0)),
    ]
    for name, probs in [("gsm8k", gsm), ("math", math)]:
        for label, kw in conds:
            ctrl = AblationController(model, lens, **kw)
            run_b0(model, tok, probs, ctrl, f"{name} {label}")


if __name__ == "__main__":
    main()
