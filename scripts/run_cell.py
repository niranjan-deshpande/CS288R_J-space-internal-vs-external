"""Run one condition cell over problems, checkpointed to JSONL.

Example:
  python3 scripts/run_cell.py --tag clean --mode none --budget 16384 \
      --samples 8 --problems all --batch 20
  python3 scripts/run_cell.py --tag jspace-med-B2048 --mode jspace --band 24:33 \
      --budget 2048 --samples 4 --problems solvable --batch 16
"""
import argparse
import json
import os
import time
import zlib

from jlens.lens import JacobianLens

from extcot.model import (load_model, build_prompt, LENS_PATH,
                          THINK_SAMPLING, NOTHINK_SAMPLING)
from extcot.engine import GenRequest, generate_batch, KV_BYTES_PER_TOKEN
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems, load_mmlu, load_sst2, MMLU_INSTR, SST2_INSTR
from extcot.model import MATH_INSTR

RESULTS = "/workspace/jlens-cot/results"


def seed_for(pid: str, sidx: int, tag: str) -> int:
    return zlib.crc32(f"{pid}|{sidx}|{tag}".encode()) & 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["none", "jspace", "random"], default="none")
    ap.add_argument("--band", default=None, help="e.g. 24:33 (inclusive)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rand-seed", type=int, default=0)
    ap.add_argument("--no-exemption", action="store_true")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--datasets", default="gsm8k,math,aime")
    ap.add_argument("--problems", default="all", help="all | solvable | path.json")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--kv-budget", type=float, default=9.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"{RESULTS}/runs/{args.tag}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    datasets = set(args.datasets.split(","))
    problems = [p for p in load_problems() if p["dataset"] in datasets]
    if "mmlu" in datasets:
        problems += load_mmlu()
    if "sst2" in datasets:
        problems += load_sst2()
    if args.problems == "solvable":
        with open(f"{RESULTS}/design.json") as f:
            solvable = set(json.load(f)["solvable_ids"])
        problems = [p for p in problems if p["id"] in solvable]
    elif args.problems != "all":
        with open(args.problems) as f:
            keep = set(json.load(f))
        problems = [p for p in problems if p["id"] in keep]

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["pid"], r["sample_idx"]))
                except json.JSONDecodeError:
                    pass
    print(f"[{args.tag}] {len(problems)} problems x {args.samples} samples; "
          f"{len(done)} already done", flush=True)

    model, tok = load_model()
    controller = None
    if args.mode != "none":
        lens = JacobianLens.load(LENS_PATH)
        lo, hi = (int(x) for x in args.band.split(":"))
        controller = AblationController(
            model, lens, range(lo, hi + 1), k=args.k, mode=args.mode,
            rand_seed=args.rand_seed, use_exemption=not args.no_exemption)

    thinking = args.budget > 0
    sampling = THINK_SAMPLING if thinking else NOTHINK_SAMPLING

    todo = []
    for p in problems:
        for sidx in range(args.sample_offset, args.sample_offset + args.samples):
            if (p["id"], sidx) in done:
                continue
            instr = {"mmlu": MMLU_INSTR, "sst2": SST2_INSTR}.get(p["dataset"], MATH_INSTR)
            todo.append(GenRequest(
                prompt=build_prompt(tok, p["question"], thinking=thinking, instr=instr),
                budget=args.budget, thinking=thinking,
                seed=seed_for(p["id"], sidx, args.tag),
                meta={"pid": p["id"], "ds": p["dataset"], "gold": p["gold"],
                      "sample_idx": sidx}))

    fout = open(out_path, "a")
    t0 = time.time()
    n_written = 0
    tok_total = 0

    def write(res):
        nonlocal n_written, tok_total
        for r in res:
            if r is None:
                continue
            rec = dict(pid=r.meta["pid"], dataset=r.meta["ds"],
                       sample_idx=r.meta["sample_idx"],
                       solved=grade(r.meta["ds"], r.answer_text, r.meta["gold"]),
                       think_tokens=r.think_tokens, answer_tokens=r.answer_tokens,
                       truncated=r.think_truncated, finished=r.finished,
                       answer_text=r.answer_text, completion=r.completion,
                       gen_ids=r.ids,
                       tag=args.tag, mode=args.mode, band=args.band, k=args.k,
                       budget=args.budget, rand_seed=args.rand_seed)
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
            tok_total += r.think_tokens + r.answer_tokens
        fout.flush()

    # Length-bucketed continuous scheduling. Fresh requests run in large
    # batches with a bounded step window; whatever is still alive comes back
    # as a continuation and is re-batched with peers of similar prior length
    # (bucket b holds priors in [1024*2^(b-1), 1024*2^b)). Windows double
    # with the bucket, so a long sequence's total re-prefill work stays
    # ~its own length (amortized doubling); batch sizes shrink with the
    # bucket to respect the KV budget.
    if args.mode == "none":
        n_caches = 1.0
    else:  # fork-at-band shares layers below the band between the two passes
        n_caches = 1 + (36 - int(args.band.split(":")[0])) / 36
    n_total = len(todo)

    def bucket_of(r):
        n = len(r.prior_ids)
        return 0 if n < 1024 else min(4, n.bit_length() - 10)

    def batch_for(b):
        # rows such that a full window at this length fits the KV budget
        max_len = (2 ** (b + 1)) * 1024 + 512   # prior max + window + slack
        cap = int(args.kv_budget * 1e9 / (KV_BYTES_PER_TOKEN * n_caches * max_len))
        return max(6, min(args.batch, cap))

    # sort fresh requests by expected think length so each call's sequences
    # finish together (homogeneous batches feed the straggler tail less)
    rank = {p["id"]: {"sst2": 0, "mmlu": 0, "gsm8k": 1, "aime": 9}
            .get(p["dataset"], 2 + p.get("level", 3) / 10) for p in problems}
    todo.sort(key=lambda r: rank[r.meta["pid"]])

    buckets = {b: [] for b in range(5)}
    buckets[0] = todo
    calls = 0
    while any(buckets.values()):
        b = min(k for k, v in buckets.items() if v)
        n = batch_for(b)
        chunk, buckets[b] = buckets[b][:n], buckets[b][n:]
        window = 1024 * (2 ** b)
        res, conts = generate_batch(model, tok, chunk, controller=controller,
                                    sampling=sampling, max_steps=window,
                                    kv_budget_gb=args.kv_budget, verbose=True)
        write(res)
        for c in conts:
            buckets[bucket_of(c)].append(c)
        el = time.time() - t0
        left = sum(len(v) for v in buckets.values())
        print(f"  [{args.tag}] {n_written}/{n_total} done, {left} queued "
              f"(bucket {b}, n={len(chunk)}) | {tok_total} tok | "
              f"{tok_total/max(el,1):.0f} tok/s | {el/60:.0f}m", flush=True)
        calls += 1
        if calls > 2000:
            print("  scheduler runaway, aborting remaining", flush=True)
            break

    if controller is not None and controller.stat_n:
        print(f"  removal_frac mean: {controller.stat_sum/controller.stat_n:.4f}",
              flush=True)
    fout.close()
    print(f"[{args.tag}] DONE {n_written} records in {(time.time()-t0)/60:.0f}m",
          flush=True)


if __name__ == "__main__":
    main()
