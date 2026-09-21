# SC-ZCP-Bench: zero-cost proxy evaluation for speech architectures

Code, benchmark and artifacts for **"How much ground truth does a zero-cost proxy need? A benchmark
and specialization study for speech architectures"** (submitted to ICASSP 2027).

SC-ZCP-Bench is a keyword-spotting (KWS) proxy-evaluation benchmark on Google Speech Commands v2
(12-class): **1,000 trained architectures** from two families — Space A, DS-CNN-style 2D
depthwise-separable stacks (500), and Space B, TC-ResNet-style temporal-convolution residual stacks
(500) — **2,600 training runs**, with **three seeds on every evaluation architecture**. The extra seeds
give a test–retest noise ceiling, the agreement a retrained replicate reaches against the seed-0 ground
truth: **ρ = 0.953 [0.922, 0.972]** on Space B (n = 200). Every proxy correlation is read against it.

## Findings

Brackets are 10,000-resample bootstrap 95% CIs over architectures; ± is the standard deviation over
10 evolution seeds. Every number is in `speech_zcp/results/` (see the artifact index below).

1. **Specialization is real and cheap for evolved proxies.** Proxies evolved by genetic programming
   on vision alone (NAS-Bench-201) transfer weakly to the TC-ResNet family: ρ = 0.458 ± 0.192.
   Fifty in-domain models raise this to 0.702 ± 0.060; no later budget is CI-separated from N = 50
   (0.739 ± 0.017 at N = 200, 0.721 ± 0.016 at N = 300).
2. **Nothing beats trivial compute on Space B.** FLOPs scores 0.879 [0.837, 0.910], statistically
   tied with `nwot` (0.864 [0.820, 0.895]); no proxy, evolved or fixed, demonstrates an improvement
   over it. FLOPs is nevertheless below the ceiling: paired-bootstrap gap 0.07 [0.04, 0.11]
   (0.10 [0.06, 0.15] on the pooled replication). Evolved > #params fails at every budget
   (one-sided Wilcoxon across seeds, p ≥ 0.986).
3. **Which trivial statistic wins is family-dependent.** On the DS-CNN family the ordering
   reverses: #params 0.833 [0.674, 0.918] vs. FLOPs 0.548 [0.323, 0.708] on the sealed A-test
   (n = 50), 0.821 [0.768, 0.862] vs. 0.538 [0.445, 0.623] on the pooled Space-A replication (n = 250).
4. **Beyond compute, weight magnitude turns negative.** Controlling for log FLOPs (partial Spearman),
   #params is −0.28 [−0.40, −0.13] on Space B while `nwot` keeps a small positive residual
   (+0.18 [+0.03, +0.33]); on Space A the signs flip (#params +0.77 [+0.60, +0.88], `nwot` −0.29).

Protocol and success criteria were pre-registered as dated commits (`PROTOCOL.md`), and every
finding was scored once on 550 replication architectures (250 A, 300 B) sampled and trained after
the claims were fixed. The pre-registered success criterion for evolved proxies — outranking
trivial baselines with CI separation — was **not** met; that branch is reported as is.

## The benchmark

| Set | Architectures | Seeds | Role | Records |
|---|---|---|---|---|
| Space A pool | 200 | 0 | specialization (GP fitting) | `results/gt_A/` |
| Space A test | 50 | 0, 1, 2 | sealed within-family evaluation | `results/gt_A/` (ids in `splits/a_pool_test.json`) |
| Space B | 200 | 0, 1, 2 | sealed cross-family evaluation, never fitted | `results/gt_B/` |
| A_rep, A_rep2 | 150, 100 | 0, 1, 2 | replication, sampled after the claims froze | `results/gt_A_rep/`, `results/gt_A_rep2/` |
| B_rep, B_rep2 | 200, 100 | 0, 1, 2 | replication | `results/gt_B_rep/`, `results/gt_B_rep2/` |
| Heavy-recipe check | 50 (Space B) | 0 | recipe-robustness only, never ground truth | `results/gt_B_fullrecipe/` |

Ground truth for every correlation is the seed-0 test accuracy; seeds 1–2 estimate the ceilings only.
Each record `<arch_id>_s<seed>.json` carries the architecture config, recipe, validation and test
accuracy, #params, FLOPs, wall-clock, git hash and timestamp.

- Task: Speech Commands v2, 12 classes (10 commands + silence + unknown), official validation and
  testing lists, 40 × 101 log-mel input.
- Frozen recipe `pilot-v2`: Adam 3e-3 with cosine schedule, batch 128, ≤ 15 epochs, early-stopping
  patience 3, fixed stratified 10k-clip training subsample. Median 18 s per model on an M3 Pro (MPS).
- Grammars (`speech_zcp/spaces.py`) are seeded and deterministic with a 60M-MAC cap, so the
  architecture lists regenerate identically anywhere. Deliberately quality-degrading choices give
  accuracy spreads of 27.9–94.6% (A) and 58.0–94.7% (B).
- Noise ceilings (`results/noise_ceiling.json`): Space B 0.953 [0.922, 0.972]; A-test 0.970
  [0.940, 0.980]; pooled B replication 0.939 [0.908, 0.960]; pooled A replication 0.980 [0.972, 0.985].
- Recipe robustness: the 50-architecture noise subset retrained on all clips for up to 30 epochs
  reproduces the frozen-recipe ordering at ρ = 0.907 [0.809, 0.958], equal to the same models'
  seed-to-seed agreement (0.909).

## Proxy suite and validation

Eleven rankers: #params, FLOPs, `synflow`, `snip`, `grasp`, `fisher`, `grad_norm`, `nwot`,
`l2_norm`, `plain`, `zen`. Implementations were gated against NAS-Bench-Suite-Zero's precomputed
scores on 500 NB201/CIFAR-10 architectures (`results/nb201_gate.json`): 9/11 pass (ρ ≥ 0.99
data-free, ≥ 0.90 data-dependent). `plain` and `grasp` miss for a measured reason: they
self-correlate at only 0.30 / 0.77 across two minibatches (`results/batch_sensitivity.json`), so
the threshold is unattainable cross-batch; they are kept as batch-unstable baselines on the frozen
minibatch. Per-architecture cost of every ranker family is in `results/ranker_cost.md`.

## Repository layout

```
speech_zcp/
  spaces.py            Space A/B sampling grammars (seeded, 60M-MAC cap)
  data_sc.py           Speech Commands v2 12-class protocol + feature cache
  train_gt.py          frozen ground-truth recipe, sweep runner, pilot gate
  replication.py       replication sets: pinned seed streams, splits, training
  recipe_check.py      heavy-recipe robustness check (50 Space-B archs)
  proxies.py           the 11-proxy suite (zero-cost-nas / NASLib conventions)
  bench_201.py         NB201 models + NAS-Bench-Suite-Zero validation gate
  cache_stats.py       frozen statistics extraction (cache-v1; speech terminals)
  gp_engine.py         typed GP over cached statistics (+ unit tests)
  evolve.py            budget-curve evolution: N in {0,25,50,100,200,300} x 10 seeds
  evaluate.py          the only script that touches the sealed sets (each evaluation runs once)
  search_scale.py      population-100 x 30-generation search check
  transfer_matrix.py   cross-modality transfer incl. the NinaPro reference column
  search_demo.py       ranked-querying demonstration
  analysis_terciles.py within-FLOPs-tercile analysis
  ranker_cost.py       per-architecture cost of every ranker family
  make_fig1_curve.py   Fig. 1(a);  make_fig1.py --scatter  Fig. 1(b);  make_fig2.py;  make_table1.py
  splits/              sealed A-test ids and the replication splits
  results/             ground truth, elites, and every evaluation artifact
PROTOCOL.md            pre-registration log: every protocol decision and outcome, dated
COMMIT_HASH_MAP.txt    maps the git hashes recorded inside artifacts to this repository's commits
```

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m speech_zcp.data_sc --root data                    # Speech Commands v2 + features
.venv/bin/python -m speech_zcp.smoke_test                             # proxies finite + deterministic
.venv/bin/python -m speech_zcp.bench_201 --n 500                      # proxy validation gate

# ground truth (resume-safe; training on Apple MPS)
.venv/bin/python -m speech_zcp.train_gt --pilot --device mps
.venv/bin/python -m speech_zcp.train_gt --space A --count 250 --device mps
.venv/bin/python -m speech_zcp.train_gt --space B --count 200 --device mps
.venv/bin/python -m speech_zcp.train_gt --space B --count 200 --seed 1 --device mps   # and --seed 2
.venv/bin/python -m speech_zcp.replication --sample                   # replication splits (pinned streams)
.venv/bin/python -m speech_zcp.replication --train A_rep --device mps # also B_rep, A_rep2, B_rep2;
                                                                      # --seed 1 / 2 for the ceilings; --train Atest --seed 1 / 2
.venv/bin/python -m speech_zcp.recipe_check --train --device mps      # heavy-recipe check

# statistics caches (one forward/backward pass per architecture, frozen minibatch)
.venv/bin/python -m speech_zcp.cache_stats --target A                 # then B, nb201, A_rep, B_rep, A_rep2, B_rep2

# evolution grid: --budget {0,25,50,100,200,300} x --seed 1..10 (--warm-start <N=0 elite> for the ablation)
.venv/bin/python -m speech_zcp.evolve --budget 50 --seed 1
.venv/bin/python -m speech_zcp.evolve --budget 200 --seed 1 --population 100 --generations 30 \
    --out-dir speech_zcp/results/evolved_bigsearch --tag-suffix _big    # search-scale check

# sealed evaluations, each run once (the matching --*selfcheck flags reproduce the committed numbers first)
.venv/bin/python -m speech_zcp.evaluate                               # -> results/evaluation.json
.venv/bin/python -m speech_zcp.evaluate --replication                 # -> replication.json
.venv/bin/python -m speech_zcp.evaluate --replication2                # -> replication2.json (rep2 + pooled)
.venv/bin/python -m speech_zcp.evaluate --ceiling                     # -> noise_ceiling.json (then --ceiling --extend)
.venv/bin/python -m speech_zcp.evaluate --partial                     # -> partial_flops.json
.venv/bin/python -m speech_zcp.evaluate --n300                        # -> n300.json
.venv/bin/python -m speech_zcp.recipe_check --analyze                 # -> recipe_check.json
.venv/bin/python -m speech_zcp.search_scale                           # -> search_scale.json
.venv/bin/python -m speech_zcp.transfer_matrix                        # -> transfer_matrix.json
.venv/bin/python -m speech_zcp.search_demo                            # -> search_demo.json
.venv/bin/python -m speech_zcp.analysis_terciles                      # -> terciles.json
.venv/bin/python -m speech_zcp.ranker_cost                            # -> ranker_cost.json

# figures and table
.venv/bin/python -m speech_zcp.make_fig1_curve                        # Fig. 1(a) -> fig1_curve.pdf
.venv/bin/python -m speech_zcp.make_fig1 --scatter                    # Fig. 1(b) -> fig1_scatter.pdf
.venv/bin/python -m speech_zcp.make_fig2                              # -> fig2_partial_flops.pdf
.venv/bin/python -m speech_zcp.make_table1                            # -> table1.tex
```

## Artifact index

| `speech_zcp/results/` | Contents | Paper |
|---|---|---|
| `evaluation.json` | all rankers on sealed Space B and A-test: Spearman with bootstrap CIs, Kendall τ, P@10%, tercile Spearman, Wilcoxon vs. #params | Table 1, Table 2, Fig. 1, Secs. 4.1–4.2 |
| `replication.json/.md`, `replication2.json/.md` | the same rankers on the replication sets, alone and pooled | Secs. 1, 4.2, 4.3 |
| `noise_ceiling.json` | test–retest ceilings and paired ceiling-minus-FLOPs gaps | Secs. 2, 4.2 |
| `partial_flops.json/.md` | partial Spearman controlling for log10 FLOPs | Sec. 4.4 |
| `terciles.json/.md` | within-FLOPs-tercile Spearman | Table 2 |
| `n300.json/.md`, `evolved_n300/` | the N = 300 elites and their sealed scores | Table 1, Fig. 1, Sec. 4.1 |
| `search_scale.json/.md`, `evolved_bigsearch/` | population 100 × 30 generations check | Sec. 4.4 |
| `recipe_check.json`, `gt_B_fullrecipe/` | heavy-recipe robustness | Sec. 2 |
| `transfer_matrix.json` | cross-modality transfer incl. NinaPro | Sec. 4.5 |
| `search_demo.json` | ranked querying | Sec. 4.2 |
| `nb201_gate.json`, `batch_sensitivity.json` | proxy validation gate | Sec. 3 |
| `ranker_cost.json/.md` | per-architecture cost of every ranker family | Sec. 3 |
| `evolved/` | 90 elite records (cold and warm-started, budgets 25–200; N = 0 vision elites) | Table 1 |
| `pilot/` | the 40-model pilot gate | Sec. 2 |
| `proxy_scores_*.json` | cached standard-proxy scores per set (frozen minibatch) | — |
| `envcheck_seed1_retrain.json`, `hand_verification.json`, `space_heterogeneity.json`, `terminal_usage.json`, `verification_report.md` | supporting checks | — |

The `git_hash` values embedded in the artifacts, and the hashes cited in `PROTOCOL.md`, refer to the
repository's pre-release history; `COMMIT_HASH_MAP.txt` maps each to the corresponding current commit.

## Attribution and licenses

Proxy implementations follow the conventions of
[zero-cost-nas](https://github.com/mohsaied/zero-cost-nas) (Abdelfattah et al., ICLR 2021) and
[NASLib](https://github.com/automl/NASLib) (both Apache-2.0); the Zen-score follows ZenNAS.
Search-space families: DS-CNN (Zhang et al., "Hello Edge") and TC-ResNet (Choi et al., Interspeech
2019). Dataset: Google Speech Commands v2 (Warden, CC-BY-4.0).

The benchmark (ground truth, grammars, recipe and per-model records) will be archived with a DOI on
acceptance.
