"""Derive a budget-B cell from its B=16384 source cell by prefix reuse.

The spec defines budget B as truncation of the think block at B tokens,
with "</think>" + the answer prefix forced. The engine is budget-oblivious
until the forcing check (think_ct >= budget), so the first B sampled think
tokens of a B=16384 run are distributed exactly as a fresh budget-B run's.
This runner therefore truncates each source transcript at B and regenerates
only the forced end + answer (under the same ablation condition, prefilled
with exemptions like any continuation), with a fresh per-target-cell seed.
Records whose natural think length < B are copied verbatim: the budget
never binds, so the full record is already a budget-B outcome.

Note for the report: budget points derived from one source share think
prefixes (common-random-numbers pairing across the budget axis). Marginal
per-budget solve rates are unbiased; the budget-response curve is estimated
with lower variance; bootstrap-over-problems CIs remain valid.

  python3 scripts/run_reuse.py --source-tag jspace-med-B16384 \
      --tag jspace-med-B512 --budget 512 --mode jspace --band 22:34 --k 10 \
      --datasets gsm8k,math --batch 64

--staged-source (opt-in): use when the source cell ran run_cell.py --staged
(early-stop on determined majority), so samples 2/3 are missing for problems
whose 16K majority was already determined. The coverage gate then requires
samples {0,1} only (always present under staging). Missing source samples are
NOT silently dropped: after deriving everything available, this script writes
results/backfill-{tag}-s{2,3}.json listing problems whose majority AT THIS
BUDGET is still undecided and which lack that sample, and prints the exact
run_cell.py commands to generate them FRESH at this budget (distributionally
identical to a derived sample: the spec defines budget B as truncation, and
the engine is budget-oblivious until the forcing check, so a fresh budget-B
sample and a truncated 16K sample are the same distribution; only the
common-random-numbers pairing across budgets is lost for those samples —
variance, not bias). Workflow (idempotent, resumable):
  run_reuse --staged-source -> run_cell backfill s2 (if listed)
  -> run_reuse --staged-source (recomputes; regens nothing) -> run_cell
  backfill s3 (if listed). Empty lists = cell complete.
"""
import argparse
import json
import os
import random
import time

from extcot.model import load_model, build_prompt, LENS_PATH
from extcot.engine import GenRequest, generate_batch, THINK
from extcot.model import THINK_SAMPLING
from extcot.ablation import AblationController
from extcot.grading import grade
from extcot.data import load_problems

RESULTS = "/workspace/jlens-cot/results"


def seed_for(pid: str, sidx: int, tag: str) -> int:
    import zlib
    return zlib.crc32(f"{pid}|{sidx}|{tag}".encode()) & 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--mode", choices=["none", "jspace", "random"], default="none")
    ap.add_argument("--band", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rand-seed", type=int, default=0)
    ap.add_argument("--datasets", default="gsm8k,math")
    ap.add_argument("--samples", default=None,
                    help="comma list of sample_idx to keep (default: all)")
    ap.add_argument("--problems", default="solvable")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--kv-budget", type=float, default=55.0)
    ap.add_argument("--source-path", default=None)
    ap.add_argument("--staged-source", action="store_true",
                    help="source cell ran with run_cell --staged; gate on "
                         "samples {0,1} and emit backfill lists for the rest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    B = args.budget
    datasets = set(args.datasets.split(","))
    keep_sidx = (set(int(x) for x in args.samples.split(","))
                 if args.samples else None)
    out_path = args.out or f"{RESULTS}/runs/{args.tag}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    target = {}   # (pid, sidx) -> solved, over everything in the TARGET cell
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["pid"], r["sample_idx"]))
                    target[(r["pid"], r["sample_idx"])] = bool(r["solved"])
                except json.JSONDecodeError:
                    pass

    if args.problems == "solvable":
        with open(f"{RESULTS}/design.json") as f:
            allowed = set(json.load(f)["solvable_ids"])
    elif args.problems == "all":
        allowed = None
    else:
        with open(args.problems) as f:
            allowed = set(json.load(f))

    src_path = args.source_path or f"{RESULTS}/runs/{args.source_tag}.jsonl"
    src = []
    src_keys = set()
    with open(src_path) as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] not in datasets:
                continue
            if allowed is not None and r["pid"] not in allowed:
                continue
            if keep_sidx is not None and r["sample_idx"] not in keep_sidx:
                continue
            assert B < r["budget"], f"target budget {B} >= source {r['budget']}"
            src_keys.add((r["pid"], r["sample_idx"]))
            if (r["pid"], r["sample_idx"]) in done:
                continue
            src.append(r)

    # coverage gate: a hole in the source silently vanishes from the derived
    # cell (and the loss is length-correlated), so fail loudly instead
    from extcot.data import load_problems as _lp
    exp_pids = [p["id"] for p in _lp()
                if p["dataset"] in datasets and (allowed is None or p["id"] in allowed)]
    if args.staged_source:
        # staged sources always carry samples {0,1}; 2/3 exist only where the
        # source majority was undecided — those go to the backfill path below
        want = keep_sidx if keep_sidx is not None else {0, 1, 2, 3}
        sidxs = sorted(want & {0, 1})
        staged_missing = {(pid, s) for pid in exp_pids
                          for s in sorted(want & {2, 3})} - src_keys
    else:
        sidxs = sorted(keep_sidx) if keep_sidx else sorted({s for _, s in src_keys})
        staged_missing = set()
    missing = {(pid, s) for pid in exp_pids for s in sidxs} - src_keys
    if missing:
        raise SystemExit(f"FATAL [{args.tag}]: source {args.source_tag} missing "
                         f"{len(missing)} (pid, sample) pairs, e.g. "
                         f"{sorted(missing)[:5]}")

    problems = {p["id"]: p for p in load_problems()}
    fout = open(out_path, "a")

    def emit(rec):
        target[(rec["pid"], rec["sample_idx"])] = bool(rec["solved"])
        fout.write(json.dumps(rec) + "\n")

    def write_backfill():
        """Staged-source only: list (pid, sample) pairs this budget still
        needs (majority undecided) but the staged source never generated;
        they must be generated fresh at this budget via run_cell.py."""
        if not args.staged_source:
            return
        def det(pid):
            got = [target[(pid, i)] for i in range(4) if (pid, i) in target]
            return sum(got) >= 2 or (len(got) - sum(got)) >= 3
        need = {2: [], 3: []}
        for pid, s in sorted(staged_missing):
            if (pid, s) in target or det(pid):
                continue
            if s == 3 and any((pid, i) not in target for i in (0, 1, 2)):
                continue   # re-run this script after backfilling sample 2
            need[s].append(pid)
        for s, pids in need.items():
            path = os.path.join(os.path.dirname(out_path) or ".",
                                f"backfill-{args.tag}-s{s}.json")
            with open(path, "w") as f:
                json.dump(pids, f)
            if pids:
                cmd = (f"python3 scripts/run_cell.py --tag {args.tag} "
                       f"--mode {args.mode} "
                       + (f"--band {args.band} --k {args.k} "
                          if args.mode != "none" else "")
                       + (f"--rand-seed {args.rand_seed} "
                          if args.mode == "random" else "")
                       + f"--budget {B} --samples 1 --sample-offset {s} "
                       f"--problems {path} --datasets {args.datasets} "
                       f"--batch {args.batch} --kv-budget {args.kv_budget} "
                       f"--out {out_path}")
                print(f"[{args.tag}] BACKFILL: {len(pids)} problems undecided "
                      f"at B={B} lack sample {s}; run:\n  {cmd}\n  then re-run "
                      f"this run_reuse command to refresh the backfill lists.",
                      flush=True)
        if not need[2] and not need[3]:
            print(f"[{args.tag}] staged-source: no backfill needed; "
                  f"cell majority outcomes complete.", flush=True)

    # 1) verbatim copies: budget never binds (natural think < B)
    n_copy = 0
    todo_src = []
    for r in src:
        if r["think_tokens"] < B:
            rec = dict(r)
            rec.update(tag=args.tag, budget=B, mode=args.mode, band=args.band,
                       k=args.k, rand_seed=args.rand_seed,
                       reused_from=args.source_tag, reuse="verbatim")
            emit(rec)
            n_copy += 1
        else:
            todo_src.append(r)
    fout.flush()
    print(f"[{args.tag}] source {len(src)}: {n_copy} verbatim, "
          f"{len(todo_src)} to truncate+regen at B={B}", flush=True)

    if not todo_src:
        write_backfill()
        fout.close()
        return

    model, tok = load_model()
    controller = None
    if args.mode != "none":
        from jlens.lens import JacobianLens
        lens = JacobianLens.load(LENS_PATH)
        lo, hi = (int(x) for x in args.band.split(":"))
        controller = AblationController(
            model, lens, range(lo, hi + 1), k=args.k, mode=args.mode,
            rand_seed=args.rand_seed)

    reqs = []
    for r in todo_src:
        pid, sidx = r["pid"], r["sample_idx"]
        seed = seed_for(pid, sidx, args.tag)
        rng = random.Random(seed)
        reqs.append(GenRequest(
            prompt=build_prompt(tok, problems[pid]["question"], thinking=True),
            budget=B, thinking=True, seed=seed,
            prior_ids=list(r["gen_ids"][:B]),
            state=dict(phase=THINK, depth=0, think_ct=B, ans_ct=0,
                       truncated=True, forced=[], rng=rng.getstate()),
            meta={"pid": pid, "ds": r["dataset"], "gold": problems[pid]["gold"],
                  "sample_idx": sidx}))

    t0 = time.time()
    n_done = 0
    todo = reqs
    while todo:
        pending = []
        for s in range(0, len(todo), args.batch):
            res, conts = generate_batch(model, tok, todo[s:s + args.batch],
                                        controller=controller,
                                        sampling=THINK_SAMPLING,
                                        kv_budget_gb=args.kv_budget)
            pending += conts
            for g in res:
                if g is None:
                    continue
                emit(dict(pid=g.meta["pid"], dataset=g.meta["ds"],
                          sample_idx=g.meta["sample_idx"],
                          solved=grade(g.meta["ds"], g.answer_text, g.meta["gold"]),
                          think_tokens=g.think_tokens, answer_tokens=g.answer_tokens,
                          truncated=g.think_truncated, finished=g.finished,
                          answer_text=g.answer_text, completion=g.completion,
                          gen_ids=g.ids, tag=args.tag, mode=args.mode,
                          band=args.band, k=args.k, budget=B,
                          rand_seed=args.rand_seed,
                          reused_from=args.source_tag, reuse="truncated"))
                n_done += 1
            fout.flush()
            print(f"  [{args.tag}] {n_done}/{len(reqs)} regen | "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
        todo = pending
    write_backfill()
    fout.close()
    print(f"[{args.tag}] DONE {n_copy} verbatim + {n_done} regen in "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
