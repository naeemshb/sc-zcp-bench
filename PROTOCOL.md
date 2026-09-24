# SC-ZCP-Bench: protocol and pre-registration log

This file records the frozen protocol behind the benchmark and every pre-registered decision, each with
the commit in which it was fixed before the corresponding run. It is a condensed version: the full
working notes are preserved in this file's own history (`git log -p -- PROTOCOL.md`), and
`COMMIT_HASH_MAP.txt` links the git hashes recorded inside the result artifacts to current commits.
Paper section numbers refer to the ICASSP 2027 submission.

## 1. Rules that governed every result

1. **Every number traces to an artifact** in `speech_zcp/results/`, written by a script in this
   repository with the config, git hash and timestamp embedded.
2. **Holdout sanctity.** No model or proxy was selected using Space B or the Space-A test set. Their
   sealed evaluations (every proxy and elite correlation in the paper) are computed by `evaluate.py`,
   once per pre-registered configuration, after that configuration froze; the descriptive checks that
   also read their ground truth (`search_demo`, `analysis_terciles`, `recipe_check`, `striding_check`)
   select nothing. Evolved proxies are fitted on the Space-A specialization pool and model-selected on
   validation architectures only.
3. **CI-gated language.** "X outperforms Y" requires separated bootstrap CIs; overlapping intervals
   are ties.
4. **Fixed recipes.** One frozen training recipe (`pilot-v2`) for all ground truth and one frozen
   statistics-extraction protocol (`cache-v1`) for all caches.
5. **Invalid evolved formulas** (NaN/Inf, constant output, fewer than 90% valid architectures) receive
   the worst fitness.
6. **Pre-registration.** For each experiment the protocol and its reading rule were committed before
   the run; the outcome is reported as it lands, and any wording it invalidates is changed
   (honesty clause).

## 2. Benchmark (frozen 2026-08-04, commit 588c516)

- **Task.** Google Speech Commands v2, 12 classes (10 commands + silence + unknown), official
  validation/testing lists, 40-mel log-spectrograms (40 × 101), cached once.
- **Search spaces** (`speech_zcp/spaces.py`, grammar v2.1, seeded and deterministic, 60M-MAC
  rejection cap). Space A: DS-CNN-style 2D depthwise-separable stacks. Space B: TC-ResNet-style
  temporal-convolution residual stacks. Both grammars include deliberately quality-degrading choices
  (1×1 kernels, strided stems, free strides, dilation 4, widths down to 8 channels / 0.25×) so that
  accuracy spans weak to strong models.
- **Sets.** Space A: 250 architectures = 200 specialization pool + 50 sealed A-test
  (`splits/a_pool_test.json`). Space B: 200 architectures, evaluation only, never fitted.
  Replication sets sampled later under pre-registration (Sec. 7): A_rep 150, B_rep 200, A_rep2 100,
  B_rep2 100. Totals: 1,000 architectures, 2,600 training runs, three seeds on all 800 evaluation
  architectures.
- **Recipe `pilot-v2`.** Adam 3e-3 with cosine schedule, batch 128, at most 15 epochs, early-stopping
  patience 3 on the fixed validation split, fixed stratified 10k-clip training subsample, seed 0.
  Seeds 1–2 vary initialization and minibatch order only. Every record stores validation and test
  accuracy, #params, FLOPs and wall-clock.
- **Pilot gate.** Pass criteria set in advance: wall-clock at most 5 min per model, accuracy spread of
  at least 15 points across the pilot models, no divergence. Outcome on 40 models (20 per space):
  spread 65.5 points, maximum wall-clock 197 s in the pilot, within-space #params–accuracy ρ 0.684 (A)
  / 0.727 (B). Recipe v1 and grammars v1/v2 had failed earlier iterations (56 min per model;
  8.1-point spread; a 313 s outlier at 112M MACs) and were replaced before any benchmark training.
- **Noise ceiling.** Mean of the three pairwise Spearman correlations of seed-wise test accuracies over
  a set's architectures; 10,000-resample percentile bootstrap over architectures with a
  content-defined architecture order. It is a test–retest reliability under the frozen recipe, an
  approximate upper bound on single-seed proxy–accuracy rank correlation. Ground truth for every
  correlation remains the seed-0 accuracy.

## 3. Proxy suite and validation gate (2026-08-04, commit ee86fa7)

Eleven rankers: #params, FLOPs, synflow, snip, grasp, fisher, grad_norm, nwot, l2_norm, plain, zen
(conventions of zero-cost-nas / NASLib; zen follows ZenNAS). Gate: Spearman against
NAS-Bench-Suite-Zero's precomputed scores on 500 NAS-Bench-201/CIFAR-10 architectures, pass at
ρ ≥ 0.99 for #params, FLOPs and synflow and ρ ≥ 0.90 for the other eight (`results/nb201_gate.json`;
l2_norm and zen are data-free but depend on the initialization seed and random inputs, hence 0.90).
Outcome: 9/11 pass (params 1.000, synflow 0.9992, zen 0.9978 after a real bug fix, nwot 0.9959,
flops 0.9960, grad_norm 0.9906, snip 0.9887, l2_norm 0.9468, fisher 0.9303). plain (0.167) and grasp
(0.683) miss for a measured reason: they self-correlate at only 0.295 / 0.770 across two minibatches
(`results/batch_sensitivity.json`), so the threshold is unattainable across batches; both stay in the
suite as batch-unstable baselines on the frozen minibatch.

## 4. Statistics caches (`cache-v1`)

For every architecture, one forward and one backward pass (cross-entropy loss) on a fixed minibatch
(batch 8, seed 0, CPU), caching per-layer weights, gradients and activations. Speech-specific terminals:
per-layer activation standard deviation across the time axis, and the same statistics on a matched
Gaussian-noise minibatch (terminals `AstdT`, `Gn`, `An`, `AnstdT`).

## 5. Evolution and model selection (frozen 2026-08-09, commit 9494106, before the first run)

- **Engine.** Typed genetic programming: per-layer tensor terminals → elementwise operators
  (+, −, ×, safe ÷, abs, log(|·|+ε), sign, relu, square, neg) → reductions (sum, mean, std, L1, L2,
  fraction positive) → cross-layer aggregation (sum, mean, min, max). Ill-typed offspring are rejected.
- **Search settings.** Population 30, 15 generations, tournament size 3, crossover probability 0.5,
  elitism 2, depth cap 6, complexity penalty 0.002 per node. Ten seeds per budget, cold- and
  warm-started (warm start injects the seed's vision-evolved N=0 elite).
- **Budgets and chains.** N ∈ {0, 25, 50, 100, 200, 300}. N=0 evolves on the NAS-Bench-201 cache
  against its accuracies. For N ≥ 25 each seed draws one nested chain (25 ⊂ 50 ⊂ 100 ⊂ 200) from the
  Space-A pool: the pool is sorted by FLOPs, split into five interleaved groups, each group shuffled by
  the seed and read round-robin, so every budget is a prefix of the same seeded random chain. N=300
  appends 100 A_rep2 architectures drawn by the same rule (Sec. 7, 9e).
- **Model selection.** At N ≥ 100 fitness is the penalized Spearman on the first 80% of the chain and
  the elite is selected on the held-back 20%; at N ≤ 50, where an 80/20 split would leave at most 10
  validation architectures, fitness and selection share the full chain and the complexity penalty is
  the only regularizer. Reported metrics are always computed on never-fitted sets.
- **Pre-set sanity gate and its outcome (2026-08-09, commit c08b1f8).** Criterion fixed in advance: at
  N=200 the evolved proxies must exceed #params on Space B, overall and within FLOPs terciles, with CI
  separation; otherwise the paper re-centers on the benchmark and the transfer result. Outcome:
  evolved N=200 0.739 ± 0.017 vs. #params 0.752 (one-sided Wilcoxon p = 0.99, wrong direction) and
  FLOPs 0.879 [0.837, 0.910]; 6/10 seeds converged to the weight-norm formula `asum(rl1(W))`. What
  survived with CI separation: vision-evolved 0.458 ± 0.192 vs. N=50 speech-evolved 0.702 ± 0.060.
  Warm start gave no benefit. The pivot was taken as pre-set.
- **Consensus formula (2026-08-10, commit 7720f34).** `evolved_consensus` = the modal N=200 elite
  `asum(rl1(W))`, reached independently by 6/10 cold seeds; Space-B Spearman 0.741 [0.663, 0.803].
  The search demonstration is proxy-rank-ordered querying (architectures queried in descending proxy
  order; random baseline = 1,000 shuffles), since mutation-based search is not meaningful on a
  200-architecture tabular pool.

## 6. Evaluation protocol

`evaluate.py` scores every standard proxy and every committed elite on Space B (primary, n=200) and
A-test (n=50): Spearman with 10,000-resample percentile bootstrap CIs over architectures, Kendall τ,
Precision@10% (reported, never headlined), within-FLOPs-tercile Spearman, and one-sided Wilcoxon
signed-rank tests across the 10 evolution seeds for evolved vs. #params at each budget. Evolved rows
report mean ± s.d. over seeds; separation claims use t-based 95% seed intervals. Each sealed evaluation
runs once; before every later evaluation path was used, a self-check reproduced the already committed
numbers exactly (`--selfcheck`, `--replication-selfcheck`, `--ceiling-selfcheck`, `--n300-selfcheck`).

## 7. Pre-registered experiments and outcomes

Each entry: what was pinned before the run, the reading rule, and what happened.

**9b — replication sets.** Declared 2026-08-10 (b6d2191), amended to run before submission (95734d1),
outcome 2026-09-01 (91fc81a). Pinned: 150 new Space-A and 200 new Space-B architectures from the seed
streams `Atest-expansion-2027` / `B-replication-2027`, arch_id-disjoint from the release, same grammar
and recipe, scored once with the frozen proxies and the already committed elites, reported beside the
originals and never merged. Honesty clause: if tighter precision moved FLOPs off the ceiling, the
wording changes. Outcome: every finding replicated — B_rep curve N=0 0.524 ± 0.149 → N=50 0.714 ± 0.049
→ N=200 0.745 ± 0.013; FLOPs 0.834 [0.772, 0.878] > nwot 0.810 > #params 0.755 ≈ evolved 0.745;
A_rep #params 0.823 [0.76, 0.87] vs. FLOPs 0.445 [0.31, 0.56]. The honesty clause fired: FLOPs' CI no
longer contained the n=50 ceiling (0.909), and at-vs-below was left to 9b-seeds.

**9b-seeds — noise ceilings on every evaluation set.** Declared 2026-09-02 (7ef216d), outcome
2026-09-02 (5c17b41). Pinned: seeds 1 and 2 for all 200 Space-B architectures, the 50 A-test
architectures, B_rep and A_rep; ground truth stays seed 0; ceilings computed once by
`evaluate.py --ceiling` after a self-check reproducing the original 0.9092 on the first 50 Space-B
architectures. Reading rule: ceiling CI entirely above FLOPs' CI → "below the ceiling"; overlapping →
"near"; FLOPs' CI containing the ceiling point → "at". A paired-bootstrap gap on identical resamples
is reported but does not decide. Outcome: ceilings B 0.953 [0.922, 0.972], B_rep 0.931 [0.885, 0.960],
A-test 0.970 [0.940, 0.980], A_rep 0.979 [0.967, 0.984] (1,100 seed runs, none non-finite). Rule
applied → **below the ceiling** on both the original (paired gap +0.073 [+0.036, +0.114]) and the
replication (+0.096 [+0.038, +0.159]).

**9b-rep2 — second replication.** Declared 2026-09-04 (ef3cc80), outcomes 2026-09-05 (c551b64).
Pinned: 100 new architectures per space from the `-r2` seed streams, seeds 0–2, scored once
(`evaluate.py --replication2`) alone and pooled with the first replication (pooling declared before
the data existed); ceilings appended with `--ceiling --extend` without touching existing blocks.
Outcome, pooled Space B (n=300): FLOPs 0.838 [0.793, 0.874] > nwot 0.819 [0.770, 0.857] > #params
0.744 ≈ evolved N=200 0.732 ± 0.016; curve 0.462 → 0.695 → 0.732; pooled Space A (n=250): #params
0.821 [0.768, 0.862] vs. FLOPs 0.538 [0.445, 0.623]. Ceilings: B_rep2 0.951 [0.911, 0.973], A_rep2
0.979 [0.958, 0.987], pooled B 0.939 [0.908, 0.960], pooled A 0.980 [0.972, 0.985]. Rule on pooled B →
below the ceiling (gap +0.101 [+0.060, +0.146]); all three verdicts agree.

**9c-recipe — recipe robustness.** Declared 2026-09-05 (b11fa5a), outcome the same day (fa199aa).
Pinned: the 50-architecture Space-B noise subset retrained with a heavier recipe (all training clips,
up to 30 epochs, patience 5), never used as ground truth; statistic = Spearman between heavy-recipe
and frozen-recipe accuracy against the 50-architecture test–retest scale. Reading rule: overlapping
CIs → "recipe-stable within test–retest noise". Outcome: 0.907 [0.809, 0.958] vs. test–retest
0.909 [0.800, 0.965] (mean accuracy +4.7 points) → recipe-stable. Fixed rankers correlate point-lower
with heavy-recipe accuracy on these 50 (FLOPs 0.819 vs. 0.913), CIs overlapping.

**9c-partial — signal beyond compute.** Declared 2026-09-05 (b11fa5a), outcome the same day
(2184ace). Pinned: partial Spearman of each ranker with accuracy controlling for log10 FLOPs, on
B, pooled B, A-test and pooled A; descriptive, no decision rule. Outcome on Space B: nwot
+0.18 [+0.03, +0.33] and grasp +0.20 [+0.05, +0.32] are the only fixed rankers with positive residual
signal; every weight-magnitude statistic is negative (#params −0.28 [−0.40, −0.13], synflow −0.32,
zen −0.33, l2 −0.28, evolved N=200 −0.26 ± 0.02). On Space A the signs flip (#params +0.77
[+0.60, +0.88], synflow +0.68, evolved N=200 +0.76 ± 0.04, nwot −0.29).

**Post-hoc descriptive check (2026-09-23, not pre-registered).** Behind the Sec. 4.4 mechanism sentence.
Controlling for log10 FLOPs on Space B, parameter count and total temporal stride are strongly associated
(+0.89 [+0.84, +0.91]; +0.92 [+0.89, +0.93] on the pooled replication), and striding is associated with
lower accuracy at fixed compute on the original set (−0.20 [−0.32, −0.06]) but not resolvably on the pooled
replication (−0.09 [−0.21, +0.03]); `results/striding_check.json`. Nothing was selected; ground truth
seed 0 only.

**9d — search scale.** Declared 2026-09-10 (ab7308d), outcome the same day (3216d35). Pinned: the
N=200 evolution re-run at population 100 × 30 generations with everything else identical, fitting side
only, no sealed set scored. Reading rule: at least 6/10 weight-only elites → the size attractor is
search-scale-robust. Outcome: weight-only elites 7/10 vs. 7/10 at the committed scale, modal formula
`asum(rl1(W))` ×5 vs. ×6, median discovery generation 3 vs. 3, fit-ρ 0.832 ± 0.015 vs. 0.808 ± 0.059 →
search-scale-robust.

**9e — N=300.** Declared 2026-09-10 (c4960f4), outcome the same day (71d363d). Pinned: each seed's
N=200 chain extended with 100 A_rep2 architectures (fitting data, so A_rep2 and the pooled Space-A
replication are not holdouts for these elites); same search settings, cold start, 240/60
fit/selection; scored once on Space B, A-test, pooled B replication and A_rep after a 40/40
self-check. Reading rule: mean within N=200's t-interval → "flat beyond the pool"; CI-separated above
→ a rise; below → reported as is. Outcome: Space B 0.721 ± 0.016 (t95 [0.709, 0.733]) vs. N=200
0.739 ± 0.017 ([0.726, 0.751]), intervals overlap → flat beyond the pool; pooled B 0.715 vs. 0.732;
A-test 0.829 vs. 0.824; A_rep 0.850 vs. 0.825; Wilcoxon vs. #params p = 1.000; 6/10 weight-only elites.

**9f — post-hoc N=100 variant (not part of the benchmark).** A search-setting variant at N=100
(population 50, 20 generations, depth cap 3, 25 seeds) was pre-registered as exploratory on
2026-09-12 (4d68373) and scored once; it did not improve on the frozen N=100 result, is not reported
in the paper's tables, and was removed from this log (85e8621). Its artifact is kept in
`speech_zcp/results/exploratory_n100/` so that the paper's mention of one further sealed scoring is traceable.

## 8. NAS-Bench-ASR availability (verified observations)

Repository front page unavailable via API and HTML on 2026-08-10 and 2026-08-11; release page
unavailable; all eight release assets of tag v1.1.0 (nb-asr-e40-{1234,1235,1236}, e10-1234, e5-1234,
info, bench-{gtx-1080ti,jetson-nano}-fp32 pickles) unavailable directly and via the Wayback Machine,
whose index for the download path is empty; no copy found on Hugging Face mirrors (commit b578ffa).
The paper states the observations only.

## 9. Literature checks behind paper claims (2026-08-04, commit 6882b8b)

NAS-Bench-Suite-Zero has no audio task; its NinaPro (sEMG) task on NAS-Bench-201 provides an
independent non-vision reference with precomputed scores. EZNAS, GreenMachine and GreenFactory are
vision-only. Zero-cost proxies were evaluated on NAS-Bench-ASR by Abdelfattah et al. (ICLR 2021) and
by ProxyBO (AAAI 2023), so claims are phrased as "to our knowledge" and scoped to the specialization
curve and the noise ceiling. zero-cost-nas and NASLib are Apache-2.0.
