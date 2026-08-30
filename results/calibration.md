# Ablation-level calibration (A100, 2026-08-30)

Pilot slice: MATH lvl>=4 x40 + GSM8K x20 ("hard-60"), 1 sample/problem,
THINK sampling. Clean references (4090 pilots, same slice/protocol):
B=256 -> 16/60, B=2048 -> 52/60. All ablated rows exemption ON (clean top-10)
unless marked NOEX. rem_frac = mean ||J-space removal|| / ||h||.

## Band pilot (k=10, B=2048) — scripts/pilot_bands2048.py

| band (layers)   | solved | rem_frac |
|-----------------|--------|----------|
| light 26-30     | 51/60  | 0.067    |
| medium 24-32    | 49/60  | 0.051    |
| heavy 22-34     | 45/60  | 0.047    |
| random 24-32    | 48/60  | 0.140    |

Band width DOES give a gradient at k=10 and B=2048 (the 4090 "no gradient"
finding was from B<=256 rows, where the clean floor hides everything; its
B=2048 band rows were lost to OOM before completion).

## Strength pilot (band 22-34) — scripts/pilot_strength.py

| condition        | B=2048 | B=256 | rem_frac |
|------------------|--------|-------|----------|
| k=25             | 49/60  | 14/60 | 0.056    |
| k=50             | 50/60  | 15/60 | 0.071    |
| k=100            | 39/60  | 13/60 | 0.092    |
| k=10 NOEX        |  6/60  |  3/60 | 0.059    |
| random k=50      | 49/60  | 15/60 | 0.168    |
| medium-band k=50 | 51/60  | 18/60 | 0.076    |

- k=25/50 are indistinguishable from k=10 (and near clean) at B=2048;
  only k=100 clearly bites (39/60, retention ~0.75).
- At B=256 all levels sit at the clean floor (~16/60): the model remains
  functional under ablation at low budget; the deficit is expressible as
  needing more external tokens, which is the quantity we measure.
- NOEX collapses the model (6/60 at B=2048) — the exemption is load-bearing;
  the NOEX variant stays cut (approved amendment 2).
- Note (4090, same protocol, different seeds): heavy k=25 B=2048 was 46/60,
  vs 49/60 here — consistent with single-sample binomial noise (sd ~3).

## Selectivity gate — scripts/check_extraction.py

Closed-book recall (paper: impaired) vs extraction with the answer in
context (paper: intact), 20 facts each:

| condition        | recall | extract |
|------------------|--------|---------|
| clean            | 20/20  | 20/20   |
| k=10 (22-31)     | 10/20  | 20/20   |
| k=25 (22-34)     | 10/20  | 20/20   |
| k=50 (22-34)     | 12/20  | 19/20   |
| k=100 (22-34)    |  7/20  | 18/20   |

(k=100 row from the rerun in pilot_extraction_k100.log; its k=10/25/50 rows
replicate the first run within noise: recall 12/11/13, extract 20/20/19.)
GATE PASSED at all grid strengths: extraction >= 90% of clean everywhere,
while recall degrades monotonically-ish to 35% at k=100 — the dissociation
widens with strength, the signature of selective workspace removal.

Gate criterion (agreed with user): extraction must stay within ~10% of clean
at any k used in the grid; recall breaking is predicted by the paper
(generation-from-latent) and is not treated as a confound.

## Locked levels (grid)

| level  | band  | k   | pilot effect @ B=2048 |
|--------|-------|-----|-----------------------|
| light  | 26:30 | 10  | 51/60 (-1)            |
| medium | 22:34 | 10  | 45/60 (-7)            |
| heavy  | 22:34 | 100 | 39/60 (-13)           |

Rationale: monotone measured ladder; light/medium realize the original
spec's band-width levels at the paper's own k=10 protocol (medium = paper
protocol on the full measured workspace band, hence the natural anchor for
the random / out-of-band / selectivity / faithfulness controls); heavy uses
the amendment's k-escalation, the only strength that visibly bites at
B=2048. k=25/50 were rejected as indistinguishable from k=10 at pilot n.
Heavy (k=100) passed the extraction gate (18/20).

Out-of-band control: band 4:16 (13 layers, width-matched to medium), k=10.
