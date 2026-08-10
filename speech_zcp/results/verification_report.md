# Verification report (P0.2) — 2026-08-10

Trust-but-verify pass over the Aug 4–5 experimental campaign, per FEEDBACK.md P0.2.
All three checks PASS. No stop-the-line events.

## 1. Holdout access audit

Rule (PROTOCOL.md hard rule 3): Space-B / A-test *correlations* are computed once, by
`evaluate.py`, after evolution configs froze. Full inventory of code touching
`results/gt_A` / `results/gt_B` (grep over `speech_zcp/*.py`, commit 595d729 lineage):

| Code path | What it reads | Status |
|---|---|---|
| `train_gt.py` | writes ground truth | writer, sanctioned |
| `evolve.py:91` (`nested_chain`) | `gt_A` *_s0 files, filtered `if r["arch_id"] in flops` where keys = the 200-arch pool — FLOPs metadata only, pool only | sanctioned; **A-test never enters fitting** (split: `random.Random('Atest-2027').sample(range(250), 50)`, saved in `splits/`) |
| `evaluate.py:135` | gt_A + gt_B accuracies | THE sanctioned one-time evaluation; ran once Aug 5 (commit 7378d37 → artifact in 595d729) |
| `search_demo.py:45` | gt_B accuracies | sanctioned post-evaluation demo (PROTOCOL.md §7 specifies GT lookups); computes no correlations |
| `analysis_terciles.py` | gt accuracies via `evaluate.gt_map` | post-evaluation re-analysis (FEEDBACK.md P0.1); no model/config selection |
| `transfer_matrix.py` | `evaluation.json` only (frozen artifact) | re-presentation, no raw GT access |

Interpretation: the holdout-correlation rule is intact. Readers beyond `evaluate.py`
are all post-evaluation consumers of ground truth for purposes the plan itself
specifies (search demo, tercile analysis) — none influenced evolution, model
selection, or configs, all of which were frozen and committed before `evaluate.py`
first ran (git order: evolution artifacts c54714e → evaluate.py 7378d37 → results 595d729).

## 2. End-to-end reproduction of evaluation.json

- Committed artifact backed up to `logs/evaluation_committed_backup.json`.
- `evaluate.py` re-run end-to-end on 2026-08-10 (same machine, same venv;
  standard-proxy scores reloaded from their committed caches
  `proxy_scores_{B,Atest}.json`; evolved scores and all 10,000-resample
  bootstraps recomputed from the frozen statistics caches).
- Comparison: **2,320 numeric/structural values compared; 0 differences**
  (tolerance 1e-9 relative; `timestamp` and `git_hash` fields excluded as
  expected-different). The rerun log is `logs/evaluation_rerun.log`.

## 3. Hand verification of an evolved elite

Formula: `asum(rl1(W))` (elite of A_N200_s1; the modal N=200 formula, 6/10 seeds).
Recomputed from raw cached tensors (`cache/B/<arch>.npz`) by explicit arithmetic —
per layer sum of |W|, then sum over layers — and compared to the pipeline's
`gp_engine.evaluate` on the first 5 Space-B architectures (by benchmark index):

| arch_id | layers | manual asum(rl1(W)) | pipeline | rel. err |
|---|---|---|---|---|
| c85cfac3f4 | 11 | 747.194679 | 747.194679 | 0.0 |
| c2f3107f59 | 14 | 2223.165630 | 2223.165630 | 0.0 |
| c59a4a8c7f | 17 | 13676.293793 | 13676.293793 | 0.0 |
| 5b2a0b4f44 | 14 | 1345.743210 | 1345.743210 | 0.0 |
| accaa36acf | 14 | 2758.347191 | 2758.347191 | 0.0 |

Exact match on all 5 (raw numbers in `results/hand_verification.json`).

## Verdict

The evaluation pipeline is reproducible, the holdout protocol held, and the GP
engine's arithmetic matches hand computation. Drafting may proceed (pending P1).
