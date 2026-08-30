"""A100 batch-scaling probe: tok/s and peak memory vs batch size, clean and
ablated (dual KV cache), to size the grid's per-cell batches."""
import time

import torch
from jlens.lens import JacobianLens

from extcot.model import load_model, build_prompt, LENS_PATH, THINK_SAMPLING
from extcot.engine import GenRequest, generate_batch
from extcot.ablation import AblationController
from extcot.data import load_problems


def run(model, tok, probs, budget, controller, label, kv_budget):
    reqs = [GenRequest(prompt=build_prompt(tok, p["question"], thinking=True),
                       budget=budget, seed=1234 + i, meta={})
            for i, p in enumerate(probs)]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res, conts = generate_batch(model, tok, reqs, controller=controller,
                                sampling=THINK_SAMPLING, kv_budget_gb=kv_budget)
    torch.cuda.synchronize(); dt = time.time() - t0
    done = [r for r in res if r is not None]
    total = sum(r.think_tokens + r.answer_tokens for r in done)
    print(f"[{label}] n={len(reqs)} B={budget}: {total} tok in {dt:.0f}s = "
          f"{total/dt:.0f} tok/s | evicted {len(conts)} | "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)


def main():
    model, tok = load_model()
    lens = JacobianLens.load(LENS_PATH)
    problems = load_problems()
    gsm = [p for p in problems if p["dataset"] == "gsm8k"]
    math = [p for p in problems if p["dataset"] == "math"]
    mix = (gsm + math)

    run(model, tok, mix[:64], 1024, None, "clean-n64", 55)
    run(model, tok, mix[:128], 1024, None, "clean-n128", 55)
    for n in (32, 48, 64):
        ctrl = AblationController(model, lens, range(22, 35), k=25, mode="jspace")
        run(model, tok, mix[:n], 1024, ctrl, f"jspace-hk25-n{n}", 55)


if __name__ == "__main__":
    main()
