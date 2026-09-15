# Cost of the rankers vs. zero-cost proxies (2026-09-14)

Measured on the M3 Pro laptop. Proxies and statistics: CPU, 4 threads, 20 architectures per space, median
seconds per architecture (exploratory/cost/ranker_cost.json). Training: wall-clock recorded in every
ground-truth file (speech_zcp/results/gt_*/*.json), frozen recipe pilot-v2 on the laptop GPU (MPS).

| ranker family | what it needs per architecture | Space A (DS-CNN) | Space B (TC-ResNet) |
|---|---|---|---|
| #params | build the model, count | 0.7 ms | 0.9 ms |
| FLOPs | build + one hooked forward, batch 1 | 3.0 ms | 1.8 ms |
| fixed ZC proxies (synflow, nwot, zen, snip, grad_norm, fisher, plain, l2) | one forward/backward on the frozen 8-clip minibatch | 1–26 ms | 1–5 ms |
| grasp (slowest fixed proxy) | forward/backward + Hessian-vector product | 71 ms | 15 ms |
| evolved proxy, inference | statistics cache: two forward/backward passes (real + matched-noise batch) | 50 ms | 9 ms |
| evolved proxy, formula evaluation on the cache | numpy reductions | < 0.1 ms | < 0.3 ms |
| training one model (ground truth) | frozen recipe, ≤15 epochs, 10k clips | median 32 s, mean 56 s | median 14 s, mean 14 s |

Seed-0 sweep, all 1,000 models: median 17.7 s, mean 35.7 s, 95th percentile 139 s, worst 610 s; 9.9 h total.
All 2,600 runs (three seeds on the 800 evaluation architectures): 21.3 h total on one laptop.

Up-front cost that only the evolved proxies pay (fixed proxies and trivial statistics have none):
- N specialization models: at the seed-0 median, N=50 ≈ 15 min, N=200 ≈ 1 h (means: 30 min, 2 h).
- GP search per seed, from the artifact timestamps (includes cache loading): N=25 7 s, N=50 11 s,
  N=100 15 s, N=200 26 s; N=0 on 300 NB201 architectures 290 s. All 90 learning-curve runs: 1 h 24 min.

Ratios: a fixed proxy costs about 1/1,000 of training one model; the trivial statistics about 1/10,000;
the evolved proxy's inference cost is the same order as a fixed proxy (two passes instead of one).

## Draft paragraph for the manuscript (Sec. 3, after "Implementation validation"; 5–6 lines)

**Cost.** All rankers are cheap next to training. #params and FLOPs come from the architecture alone
(≤3 ms per model on a laptop CPU); a fixed ZC proxy needs one forward/backward pass on the 8-clip
minibatch (1–26 ms; grasp 71 ms); an evolved proxy needs the same pass plus a matched-noise pass to
fill its statistics cache (9–50 ms) and microseconds to evaluate the formula. Training one model under
the frozen recipe takes a median of 18 s (mean 36 s, worst 610 s) on the same laptop's GPU, three orders
of magnitude more. What distinguishes the evolved proxies is an up-front cost that the fixed proxies
never pay: N trained specialization models (≈15 min at N=50, ≈1 h at N=200) plus a GP search of 7–26 s
per seed.

If these numbers go into the paper, commit exploratory/cost/ranker_cost.json first (rule 1: every
number traces to a committed artifact); the training-time numbers already trace to the gt files.
