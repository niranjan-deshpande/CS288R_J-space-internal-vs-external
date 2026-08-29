"""Engine smoke test + throughput benchmark: clean and ablated batched generation."""
import sys, time

import torch
from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, THINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems


def run(model, tok, probs, budget, controller=None, batch_label=""):
    reqs = [GenRequest(prompt=build_prompt(tok, p["question"], thinking=True),
                       budget=budget, seed=hash((p["id"], 0)) % 2**31,
                       max_answer_tokens=512, meta={"id": p["id"], "gold": p["gold"], "ds": p["dataset"]})
            for p in probs]
    torch.cuda.synchronize(); t0 = time.time()
    res, _ = generate_batch(model, tok, reqs, controller=controller, verbose=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    total = sum(r.think_tokens + r.answer_tokens for r in res)
    solved = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in res)
    trunc = sum(r.think_truncated for r in res)
    print(f"[{batch_label}] n={len(res)} B={budget}: {total} gen tokens in {dt:.1f}s "
          f"= {total/dt:.0f} tok/s | solved {solved}/{len(res)} | truncated {trunc} "
          f"| peak mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    return res


def main():
    model, tok = load_model()
    problems = load_problems()
    gsm = [p for p in problems if p["dataset"] == "gsm8k"]

    # 1. correctness smoke: small batch, tight budget -> forcing must kick in
    res = run(model, tok, gsm[:8], budget=128, batch_label="clean-B128-n8")
    r = res[0]
    print("--- sample completion (B=128):")
    print(r.completion[:400].replace("\n", "\\n"))
    print(f"think={r.think_tokens} ans={r.answer_tokens} trunc={r.think_truncated} fin={r.finished}")
    assert all(r.think_tokens <= 128 for r in res), "budget violated!"

    # 2. clean throughput at scale
    run(model, tok, gsm[:32], budget=1024, batch_label="clean-B1024-n32")

    # 3. ablated: provisional medium band (finalized after diagnostics)
    lens = JacobianLens.load(LENS_PATH)
    band = list(range(19, 29))
    ctrl = AblationController(model, lens, band, k=10, mode="jspace")
    res = run(model, tok, gsm[:16], budget=1024, controller=ctrl, batch_label="jspace-B1024-n16")
    print("--- sample ablated completion:")
    print(res[0].completion[:400].replace("\n", "\\n"))

    # 4. random control mode
    ctrl_r = AblationController(model, lens, band, k=10, mode="random", rand_seed=0)
    run(model, tok, gsm[:16], budget=1024, controller=ctrl_r, batch_label="random-B1024-n16")


if __name__ == "__main__":
    main()
