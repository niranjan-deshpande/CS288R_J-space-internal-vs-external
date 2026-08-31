"""Mirror results/runs/*.jsonl into results/runs-slim/ without the generated
token arrays/text (gen_ids, completion, answer_text, prior_ids), which the
analysis never reads. The slim files are small enough to commit, so every
figure and table can be regenerated from the repository alone:
    cp results/runs-slim/*.jsonl results/runs/
    python3 scripts/make_figures.py --results-dir results
    python3 scripts/make_tables.py  --results-dir results
Full transcripts (215MB) stay on the experiment volume / available on request.
"""
import glob
import json
import os
import sys

DROP = {"gen_ids", "completion", "answer_text", "prior_ids", "think_text",
        "prompt", "trace"}

src = sys.argv[1] if len(sys.argv) > 1 else "results/runs"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/runs-slim"
os.makedirs(dst, exist_ok=True)
total_in = total_out = 0
for path in sorted(glob.glob(f"{src}/*.jsonl")):
    name = os.path.basename(path)
    if name.startswith("_archived"):
        continue  # killed partial cells, excluded from analysis
    total_in += os.path.getsize(path)
    with open(path) as f, open(f"{dst}/{name}", "w") as out:
        for line in f:
            r = json.loads(line)
            out.write(json.dumps({k: v for k, v in r.items()
                                  if k not in DROP}) + "\n")
    total_out += os.path.getsize(f"{dst}/{name}")
print(f"{total_in/1e6:.0f}MB -> {total_out/1e6:.1f}MB in {dst}")
