# SC-ZCP-Bench: Zero-Cost Proxy Evaluation for Speech Architectures

Code and benchmark for **"Sample-Efficient Specialization of Zero-Cost Proxies
for Speech Architectures"** (submitted to ICASSP 2027).

Motivated by the documented loss of NAS-Bench-ASR — the field's only speech NAS
benchmark — this repo provides a small, permanently archived zero-cost-proxy
evaluation benchmark for keyword spotting (two search spaces, 450 trained
architectures with multi-seed noise estimates), plus the full pipeline that
uses it: a validated 11-proxy suite, GP evolution of proxies under speech
ground-truth budgets, and a cross-modality transfer analysis.

## Headline findings

1. **Vision-evolved proxies transfer poorly to speech** (Spearman 0.46 ± 0.19
   on held-out Space B), and a budget of only **N ≤ 50 trained speech models
   closes most of the gap** (0.70 ± 0.06).
2. **FLOPs ranks at the noise ceiling on TC-ResNet-style KWS** (0.879
   [CI 0.837–0.910] vs. a test–retest ceiling of 0.909): on small-footprint
   KWS spaces, trivial compute is a near-sufficient ranking statistic. No
   proxy — standard or evolved — beats it. We state this plainly rather than
   stretching claims: specialization did not beat trivial baselines at these
   budgets.
3. The best evolved proxies select **speech-specific terminals** (gradients
   under a matched-noise input), evidence that domain signals are chosen when
   they help.

## The benchmark

| Component | Contents |
|---|---|
| Space A (DS-CNN-style) | 250 archs: 200 specialization pool + 50 fixed A-test |
| Space B (TC-ResNet-style) | 200 archs, evaluation-only (never fitted) |
| Noise subset | 50 Space-B archs × 3 seeds → test–retest Spearman **0.909** |

- Task: Speech Commands v2, 12-class (10 commands + silence + unknown),
  official validation/testing split, 40-mel log-spectrograms.
- Frozen training recipe `pilot-v2`: Adam 3e-3 cosine, batch 128, ≤ 15 epochs,
  early-stop patience 3, stratified 10k train subsample, seed 0.
- Every trained model: `speech_zcp/results/gt_{A,B}/<arch_id>_s<seed>.json`
  with config, recipe, metrics, git hash, timestamp, machine.
- Sampling grammars are seeded and deterministic (`spaces.py`, MASTER_SEED
  2027, 60M-MAC cap) — architecture lists regenerate identically anywhere.

## Repo layout

```
speech_zcp/
  spaces.py        # Space A/B sampling grammars (seeded, released)
  train_gt.py      # frozen ground-truth recipe + sweep runner + pilot gate
  proxies.py       # 11-proxy suite (NASLib/zero-cost-nas conventions)
  bench_201.py     # NB201 models + NB-Suite-Zero validation gate
  data_sc.py       # SC v2 12-class protocol + feature cache
  cache_stats.py   # frozen statistics extraction (cache-v1)
  gp_engine.py     # typed GP over cached statistics + unit tests
  evolve.py        # budget-curve evolution (N ∈ {0,25,50,100,200} × 10 seeds)
  evaluate.py      # the ONLY script that touches Space B / A-test (run once)
  search_demo.py   # aging-evolution search demo
  results/         # all artifacts: ground truth, gate outcomes, evaluation.json
  figures/         # paper figures
```

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m speech_zcp.data_sc --root data          # SC v2 + features
.venv/bin/python -m speech_zcp.smoke_test                   # sanity: proxies finite + deterministic
.venv/bin/python -m speech_zcp.train_gt --pilot --device mps  # pilot gate
# full sweep (resume-safe; ~5 h on an M3 Pro via MPS):
.venv/bin/python -m speech_zcp.train_gt --space A --count 250 --device mps
.venv/bin/python -m speech_zcp.train_gt --space B --count 200 --device mps
.venv/bin/python -m speech_zcp.train_gt --space B --count 200 --range 0:50 --seed 1 --device mps
.venv/bin/python -m speech_zcp.train_gt --space B --count 200 --range 0:50 --seed 2 --device mps
# proxy validation gate vs NAS-Bench-Suite-Zero (download per bench_201.py):
.venv/bin/python -m speech_zcp.bench_201 --n 500
# statistics caches, evolution grid, one-time evaluation:
.venv/bin/python -m speech_zcp.cache_stats --target A   # then B, nb201
.venv/bin/python -m speech_zcp.evolve
.venv/bin/python -m speech_zcp.evaluate
```

Every number in the paper traces to a JSON/CSV artifact in `speech_zcp/results/`
produced by these scripts (config + git hash + timestamp embedded).
The `git_hash` values embedded in the artifacts refer to the repository's pre-release history;
`COMMIT_HASH_MAP.txt` maps each of them to the corresponding current commit. The pre-registration
log is `PROTOCOL.md`: every protocol decision and its outcome, with the date it was committed.

## Proxy-suite validation

Our implementations are gated against NAS-Bench-Suite-Zero precomputed scores
on 500 NB201/CIFAR-10 architectures (`results/nb201_gate.json`): 9/11 pass
(ρ ≥ 0.99 data-free, ≥ 0.90 data-dependent). `plain` and `grasp` miss the
threshold for a measured reason: their scores self-correlate at only
0.30/0.77 across two different minibatches (`results/batch_sensitivity.json`),
so the threshold is unattainable cross-batch regardless of implementation.
Within this benchmark all data-dependent proxies use one frozen seed-0 batch.

## Attribution and licenses

Proxy implementations follow the conventions of
[zero-cost-nas](https://github.com/mohsaied/zero-cost-nas) (Abdelfattah et al.,
ICLR 2021) and [NASLib](https://github.com/automl/NASLib) (both Apache-2.0);
the Zen-score follows ZenNAS (Alibaba). Search-space families: DS-CNN
(Zhang et al., "Hello Edge") and TC-ResNet (Choi et al., Interspeech 2019).
Dataset: Google Speech Commands v2 (Warden, CC-BY-4.0).

The benchmark (ground truth + grammars + recipe + logs) will be archived with
a DOI (Zenodo/HF) at submission.
