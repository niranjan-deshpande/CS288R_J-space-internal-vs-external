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
from extcot.engine import GenRequest, generate_batch
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
    ap.add_argument("--kv-budget", type=float, default=11.0)
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

    wave = 0
    batch = args.batch
    while todo:
        pending = []
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            res, conts = generate_batch(model, tok, chunk, controller=controller,
                                        sampling=sampling,
                                        kv_budget_gb=args.kv_budget, verbose=True)
            write(res)
            pending += conts
            el = time.time() - t0
            print(f"  [{args.tag}] wave{wave} {n_written}/{len(todo) + n_written} "
                  f"done | {tok_total} tok | {tok_total/max(el,1):.0f} tok/s | "
                  f"{el/60:.0f}m", flush=True)
        todo = pending
        wave += 1
        batch = max(2, batch // 2)
        if wave > 8:
            print("  too many waves, aborting remaining", flush=True)
            break

    if controller is not None and controller.stat_n:
        print(f"  removal_frac mean: {controller.stat_sum/controller.stat_n:.4f}",
              flush=True)
    fout.close()
    print(f"[{args.tag}] DONE {n_written} records in {(time.time()-t0)/60:.0f}m",
          flush=True)


if __name__ == "__main__":
    main()
