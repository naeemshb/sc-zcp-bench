# SC-ZCP-Bench

Benchmark, code and artifacts for **"How much ground truth does a zero-cost proxy need? A benchmark
and specialization study for speech architectures"** (submitted to ICASSP 2027).

## What it is

- **1,000 trained keyword-spotting architectures** on Google Speech Commands v2 (12 classes) from two
  families: Space A, DS-CNN-style 2D depthwise-separable stacks (500), and Space B, TC-ResNet-style
  temporal-convolution residual stacks (500).
- **2,600 training runs** under one frozen recipe, with **three seeds on every evaluation
  architecture**. The seeds give a test–retest noise ceiling of **ρ = 0.953 [0.922, 0.972]** on Space B
  (n = 200), the agreement a retrained replicate reaches against the seed-0 ground truth.
- An 11-proxy suite validated against NAS-Bench-Suite-Zero, genetic-programming evolution of proxies
  under in-domain budgets N ∈ {0, 25, 50, 100, 200, 300}, and every evaluation artifact behind the paper.

## Main results (sealed Space B, n = 200)

| Ranker | Spearman ρ |
|---|---|
| Noise ceiling (test–retest) | 0.953 [0.922, 0.972] |
| FLOPs | 0.879 [0.837, 0.910] |
| `nwot` | 0.864 [0.820, 0.895] |
| #params | 0.752 [0.677, 0.813] |
| Evolved, N = 300 | 0.721 ± 0.016 |
| Evolved, N = 200 | 0.739 ± 0.017 |
| Evolved, N = 100 | 0.669 ± 0.177 |
| Evolved, N = 50 | 0.702 ± 0.060 |
| Evolved, N = 25 | 0.645 ± 0.154 |
| Evolved, N = 0 (vision only) | 0.458 ± 0.192 |

Brackets: 10,000-resample bootstrap 95% CIs over architectures; ±: standard deviation over 10 evolution
seeds; overlapping intervals are ties. In words: fifty in-domain models recover most of the
cross-modality gap of vision-evolved proxies; no proxy, evolved or fixed, beats FLOPs, which itself
stays 0.07 [0.04, 0.11] below the ceiling; which trivial statistic wins is family-dependent (#params
leads on the DS-CNN family: 0.833 vs. FLOPs 0.548 on the sealed A-test); and beyond compute,
weight-magnitude statistics carry negative signal on Space B while `nwot` keeps a small positive
residual. All findings replicate on 550 architectures sampled and trained after the claims were
fixed. Protocol and pre-registrations: `PROTOCOL.md`.

## Contents

```
speech_zcp/              code: grammars, training, proxies, caches, evolution, evaluation, figures
speech_zcp/results/      ground truth (gt_*/), evolved elites (evolved*/), every evaluation artifact
speech_zcp/splits/       sealed A-test ids and the replication splits
PROTOCOL.md              frozen protocol and pre-registration log, with the commit behind each decision
COMMIT_HASH_MAP.txt      maps the git hashes recorded inside artifacts to this repository's commits
fig1_curve.pdf, fig1_scatter.pdf   the paper's Fig. 1(a) and 1(b)
```

Each ground-truth record `results/gt_*/<arch_id>_s<seed>.json` holds the architecture config, recipe,
validation and test accuracy, #params, FLOPs, wall-clock, git hash and timestamp. The sealed evaluation
is `results/evaluation.json`; the replications `replication.json` and `replication2.json`; the noise
ceilings `noise_ceiling.json`; the partial correlations `partial_flops.json`; the N = 300 elites
`n300.json`. The other artifacts are named after their experiment (`recipe_check`, `search_scale`,
`terciles`, `transfer_matrix`, `search_demo`, `nb201_gate`, `batch_sensitivity`, `ranker_cost`,
`striding_check`). `exploratory_n100/` holds a post-hoc N = 100 search-setting variant that was scored
once and is not part of the paper's tables (its README explains).

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m speech_zcp.data_sc --root data                          # dataset + features
.venv/bin/python -m speech_zcp.train_gt --space A --count 250 --device mps  # ground truth; also --space B,
                                                                            # --seed 1/2, and replication.py
.venv/bin/python -m speech_zcp.cache_stats --target A                       # caches; also B, nb201, *_rep, *_rep2
.venv/bin/python -m speech_zcp.evolve --budget 50 --seed 1                  # budgets 0..300 x seeds 1..10
.venv/bin/python -m speech_zcp.evaluate                                     # sealed evaluation; flags for the
                                                                            # replications, ceilings, partial, N=300
.venv/bin/python -m speech_zcp.make_fig1_curve                              # Fig. 1(a); make_fig1 --scatter for (b)
```

Each script's docstring lists its options; `PROTOCOL.md` gives the order in which the runs were made
and the self-checks that preceded every sealed evaluation.

## Attribution and licenses

Proxy implementations follow [zero-cost-nas](https://github.com/mohsaied/zero-cost-nas) (Abdelfattah
et al., ICLR 2021) and [NASLib](https://github.com/automl/NASLib), both Apache-2.0; the Zen-score
follows ZenNAS. Search-space families: DS-CNN (Zhang et al., "Hello Edge") and TC-ResNet (Choi et
al., Interspeech 2019). Dataset: Google Speech Commands v2 (Warden, CC-BY-4.0). The benchmark
(ground truth, grammars, recipe and per-model records) will be archived with a DOI on acceptance.
