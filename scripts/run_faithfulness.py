"""Faithfulness spot check: truncate an existing generated trace at 50% of its
thinking tokens, force `</think>` + answer prefix, regrade. Compare the drop
under ablation vs base (prediction 4).

  python3 scripts/run_faithfulness.py --source jspace-med-B16384 \
      --mode jspace --band 24:33 --k 10 --n 50
  python3 scripts/run_faithfulness.py --source clean --mode none --n 50 \
      --sample-idx 4
"""
import argparse
import json
import os
import random
import zlib

from jlens.lens import JacobianLens

from extcot.model import (load_model, build_prompt, LENS_PATH, THINK_SAMPLING,
                          END_THINK_ID)
from extcot.engine import GenRequest, generate_batch, THINK
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems

RESULTS = "/workspace/jlens-cot/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="tag of run to truncate")
    ap.add_argument("--mode", choices=["none", "jspace", "random"], default="none")
    ap.add_argument("--band", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    with open(f"{RESULTS}/design.json") as f:
        design = json.load(f)
    solvable = set(design["solvable_ids"])

    src = {}
    with open(f"{RESULTS}/runs/{args.source}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r["sample_idx"] == args.sample_idx and r["pid"] in solvable
                    and r["think_tokens"] >= 32):
                src[r["pid"]] = r
    pids = sorted(src)
    random.Random(0).shuffle(pids)
    pids = pids[:args.n]
    print(f"{len(pids)} transcripts from {args.source}")

    problems = {p["id"]: p for p in load_problems()}
    model, tok = load_model()
    controller = None
    if args.mode != "none":
        lens = JacobianLens.load(LENS_PATH)
        lo, hi = (int(x) for x in args.band.split(":"))
        controller = AblationController(model, lens, range(lo, hi + 1), k=args.k,
                                        mode=args.mode)

    reqs = []
    for pid in pids:
        r = src[pid]
        ids = r["gen_ids"]
        think_ids = ids[:ids.index(END_THINK_ID)] if END_THINK_ID in ids else ids
        cut = max(1, int(len(think_ids) * args.frac))
        seed = zlib.crc32(f"{pid}|faith|{args.source}".encode()) & 0x7FFFFFFF
        rng = random.Random(seed)
        reqs.append(GenRequest(
            prompt=build_prompt(tok, problems[pid]["question"], thinking=True),
            budget=cut, thinking=True, seed=seed,
            prior_ids=think_ids[:cut],
            state=dict(phase=THINK, depth=0, think_ct=cut, ans_ct=0,
                       truncated=True, forced=[], rng=rng.getstate()),
            meta={"pid": pid, "ds": r["dataset"], "gold": problems[pid]["gold"],
                  "orig_solved": r["solved"]}))

    out_path = f"{RESULTS}/runs/faith-{args.source}-f{args.frac}.jsonl"
    results = []
    for s in range(0, len(reqs), args.batch):
        res, conts = generate_batch(model, tok, reqs[s:s + args.batch],
                                    controller=controller, sampling=THINK_SAMPLING)
        assert not conts
        results += res
    with open(out_path, "w") as f:
        for r in results:
            solved = grade(r.meta["ds"], r.answer_text, r.meta["gold"])
            f.write(json.dumps(dict(pid=r.meta["pid"], solved=solved,
                                    orig_solved=r.meta["orig_solved"],
                                    answer_text=r.answer_text)) + "\n")
    orig = sum(r.meta["orig_solved"] for r in results)
    now = sum(grade(r.meta["ds"], r.answer_text, r.meta["gold"]) for r in results)
    print(f"[faith {args.source} frac={args.frac}] full-trace {orig}/{len(results)}"
          f" -> truncated {now}/{len(results)}")


if __name__ == "__main__":
    main()
