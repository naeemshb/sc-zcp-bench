# Project: Sample-Efficient Specialization of Zero-Cost Proxies for Speech Architectures (ICASSP 2027)

**Deadline:** September 16, 2026 (AoE). Internal freeze: **September 9** (numbers frozen, writing only).
**Format:** ICASSP — 4 pages + 1 reference page. Budget: 2 figures, 1 main table, 1 compact ablation/search block.
**Track:** Speech & Language Processing (primary); Machine Learning and Generative AI (fallback).
**Today's baseline assumption:** the NAS-Bench-ASR ground-truth pickles are permanently lost (repo deleted; release assets not in forks; Wayback has the page, not the binaries; the marsggbo/automl-benchmarks HF mirror has an empty nasbench-asr folder asking for copies). Recovery emails to the original authors are out; see Contingencies.

---

## 1. Research question and claims

**Primary question:** How much target-domain ground truth does it take to specialize a zero-cost proxy to a domain that has no benchmark?

**Design principle — finding-robustness:** the headline experiment (specialization learning curve) is publishable on either branch of its hypothesis:
- If vision-evolved proxies transfer poorly to speech and small budgets (N ≤ 200 trained models) close the gap → "specialization is necessary and cheap."
- If vision-evolved proxies transfer well → "evolved proxies generalize across modalities," itself noteworthy against the transfer-skeptical literature.
Do not let any experiment or writing choice re-anchor the paper on "evolved beats baselines on speech" as the only success condition.

**Contributions (paper order):**
1. A small, permanently archived zero-cost-proxy evaluation benchmark for speech (two KWS search spaces with trained ground truth), motivated by the documented loss of NAS-Bench-ASR — the field's only speech NAS benchmark.
2. The specialization learning curve: held-out speech rank correlation of GP-evolved proxies as a function of speech ground-truth budget N ∈ {0, 25, 50, 100, 200}, with standard proxies and #params/FLOPs as budget-0 reference lines.
3. A cross-space and cross-modality transfer analysis (speech space A ↔ speech space B ↔ NAS-Bench-201 vision).
4. A compact training-free search demonstration: proxy-guided evolutionary search discovering small-footprint KWS models at CPU-minutes cost.

**Positioning (verify before writing; see §8 to-dos):**
- Abdelfattah et al., ICLR 2021: introduced the standard proxy suite and evaluated it on NAS-Bench-ASR; their published speech correlations are the citable motivation that proxies weaken on speech. We cite their numbers; we cannot re-measure them (data lost).
- EZNAS (NeurIPS 2022), GreenMachine (2024), GreenFactory (2025): automatic proxy design/combination — all believed vision-only. Our delta is NOT "we evolve proxies"; it is the domain-budget question + speech + benchmark release. Never claim method-level novelty for GP-evolved proxies.
- ICASSP precedent for the task: SANAS (ICASSP 2019), Zhang et al. differentiable NAS for KWS (ICASSP 2021, ~97.2% @ ~100K params on SC v1), Peter/Roth/Pernkopf end-to-end KWS NAS + quantization (ICASSP 2022). All training-based; all efficiency-framed. Our search demo speaks to this audience.
- Training-free NAS for RNNs/Transformers (language) and the TPAMI zero-shot-NAS survey: nearest non-vision neighbors; no zero-cost NAS work for speech found as of Aug 2026. Phrase as "to our knowledge" only after the §8 literature re-check.

---

## 2. Hard rules (project-wide, non-negotiable)

1. **No claim without a reproducing artifact.** Every number in the paper traces to a CSV/JSON produced by a script in this repo, with config JSON + git hash + timestamp saved alongside.
2. **Acceptance tests before features.** No proxy implementation is "done" until it passes the NB201 validation gate (§4).
3. **Holdout sanctity.** Evolved proxies are fitted on designated specialization pools and model-selected on validation architectures only. Space-B and A-test correlations are computed once, by `evaluate.py`, after evolution configs are frozen. No peeking, no post-hoc pool re-draws, no early stopping on test metrics.
4. **CI-gated language.** Never write "X outperforms Y" unless their bootstrap CIs separate. Ties are reported as ties. With n≈200 evaluation architectures, Spearman CIs are ~±0.08–0.10 wide: we can distinguish 0.70 from 0.50, not 0.72 from 0.68. Precision@10% at n=200 involves 20 architectures and is noisy — report it, never headline it.
5. **NaN/Inf discipline.** Invalid evolved formulas (NaN, Inf, constant output, <90% valid architectures) receive worst fitness; never crash, never silently propagate.
6. **Fixed recipes.** One frozen training recipe for all ground-truth models (§3); one frozen statistics-extraction protocol for all caches (§5). Any change after the pilot invalidates and restarts the affected sweep.
7. **No cross-version comparisons presented as equivalent.** Prior ICASSP KWS numbers on SC v1 are context, not baselines; label protocol differences explicitly.

---

## 3. Phase B — the speech benchmark (name TBD; placeholder "SC-ZCP-Bench")

**Dataset:** Google Speech Commands v2. Default task: 12-class (10 commands + silence + unknown). Switch to 35-class if the pilot shows accuracy saturation (spread check below). Features: 40-mel log-spectrograms, fixed frontend, cached to disk once.

**Two search spaces, grounded in citable KWS families:**
- **Space A (DS-CNN-style):** 2D depthwise-separable conv stacks — sampled dims: #blocks 2–6, channels per block from {16…172}, kernel {3×3, 5×5, 7×7 subset}, stride pattern, pooling type. (Zhang et al., "Hello Edge.")
- **Space B (TC-ResNet-style):** temporal-conv residual stacks — sampled dims: #blocks, width multiplier, temporal kernel {3…15}, dilation on/off. (Choi et al., Interspeech 2019.)
Sampling grammars live in `spaces.py`, are seeded, and are released with the benchmark. Ranges must be tuned in the pilot to produce **deliberate accuracy spread** (weak to strong models); a benchmark where everything scores 93–95% cannot rank proxies.

**Counts (primary plan; trim per pilot wall-clock):**
- Space A: 250 architectures = 200 specialization pool + 50 fixed A-test.
- Space B: 200 architectures, evaluation-only (never used for fitting anything).
- Noise subset: 50 Space-B architectures × 2 extra seeds (3 total) → **test-retest Spearman = the noise ceiling**, reported in the paper as the upper bound no proxy can exceed.
- Total trainings ≈ 550. At 2–5 min/model split across both machines ≈ 1.5–3 days wall-clock.

**Training recipe (frozen after pilot):** Adam, fixed lr schedule, ≤20 epochs with early stopping on a fixed validation split, fixed batch size, one seed (seed 0) except the noise subset. Record test accuracy, val accuracy, params, FLOPs, wall-clock per model.

**Pilot gate (before the full sweep):** train 10 architectures per space on each machine. PASS requires: (a) wall-clock ≤ 5 min/model on the slower machine, (b) accuracy spread ≥ 15 percentage points across the 20 pilot models (else widen sampling ranges or move to 35-class and re-pilot), (c) no recipe instability (divergence/NaN).

**PILOT OUTCOME (2026-08-04): PASSED — recipe and grammars FROZEN.** Three iterations: v1 recipe failed wall-clock (56 min/model, full-data CPU); grammar v1 failed spread (8.1 pts); grammar v2 uncapped failed wall-clock marginally (313 s outlier at 112M MACs) with pooled params-rho 0.908. Frozen state: recipe `pilot-v2` (Adam 3e-3 cosine, batch 128, ≤15 epochs, patience 3, stratified 10k train subsample, seed 0) + grammar v2.1 (quality-degrading knobs: kernel-1, strided stems, free strides, dilation 4, width down to 8ch/0.25×; 60M-MAC rejection cap). Final gate on n=40 (20/space, MPS): max wall-clock 197 s, spread 65.5 pts (30.8 functional-only), params-rho within-A 0.684 / within-B 0.727 (pooled 0.852 is a cross-space size confound — within-space is the operative headroom number). 12-class retained; 35-class not needed. ALL ground-truth training runs on the Mac (MPS, 12–59× faster than CPU; depthwise convs pathological on CPU — CPU-only PC cannot train Space A). PC role reassigned: statistics caches, evolution seeds, gate compute.

**Release:** at submission, archive ground truth + grammars + recipe + per-model logs on Zenodo or HF with a DOI. The lost-NB-ASR story is the opening motivation; the DOI is its resolution.

---

## 4. Phase V — proxy suite and the NB201 validation gate

**Proxy suite:** synflow, snip, grasp, fisher, grad_norm, jacob_cov (nwot-style), l2_norm, plain, zen (or one AZ-NAS component) + trivial baselines #params, FLOPs. Implementations adapted from the reference zero-cost-nas / NASLib code where licenses allow.

**Validation gate (replaces the lost NB-ASR reproduction test, and is tighter):** NAS-Bench-Suite-Zero releases precomputed per-architecture proxy scores on NAS-Bench-201. On a ≥500-architecture NB201/CIFAR-10 sample:
- Compute each proxy with our implementation and Spearman-correlate against the precomputed scores **per proxy, per architecture**.
- PASS thresholds: ρ ≥ 0.99 for data-free proxies (synflow, params, flops; fixed init seed), ρ ≥ 0.90 for data-dependent proxies (minibatch differences legitimately perturb scores). Any miss = implementation bug until proven otherwise; investigate, do not rationalize.

**GATE OUTCOME (2026-08-04, 500 archs, artifact results/nb201_gate.json): 9/11 PASS** — params 1.000, synflow 0.9992, zen 0.9978 (after fixing a real bug: our zen now mirrors ZenNAS/NASLib exactly — gaussian re-init, L1 pre-GAP delta, running_var BN term; was 0.371 before), nwot 0.9959, flops 0.9960, grad_norm 0.9906, snip 0.9887, l2_norm 0.9468 (init-seed-dependent; explained), fisher 0.9303. **plain (0.167) and grasp (0.683) miss the threshold with cause established by experiment** (results/batch_sensitivity.json): our own implementations self-correlate at only 0.295 / 0.770 across two different minibatches (n=100), so 0.90 is unattainable across batches for these two proxies regardless of implementation; both are line-for-line convention matches to NASLib. They stay in the suite (weak baselines are informative); paper reports them with this caveat. Within-benchmark consistency is preserved by the frozen seed-0 minibatch (§5).
- Note precisely: NB-Suite-Zero provides **scalar scores and ground-truth accuracies** (via NASLib), NOT raw statistics tensors. It serves (a) this gate and (b) baseline columns in vision cells of the transfer matrix. Vision-side *evolution* requires our own NB201 statistics cache (§5).

---

## 5. Phase C — statistics caches

For every architecture in {Space A, Space B, NB201 sample (~500)}: one forward + one backward pass (CE loss) on a fixed minibatch (batch 8, fixed seed), caching per-layer: weights W_l, gradients G_l, activations A_l, plus params/FLOPs/depth metadata. One compressed file per architecture under `cache/`; memory-mapped loading; working set must fit 16 GB RAM.

**Speech-specific terminals (the interpretability bet):** additionally cache (i) per-layer activation std **across the time axis**, (ii) the same statistics on a matched noise minibatch, enabling real-vs-noise contrast terminals. Vision-evolved proxies cannot exploit temporal structure; if speech-specialized proxies preferentially select these terminals, that is the paper's interpretable mechanism. Report terminal-usage frequencies across evolved elites.

Parallelization: 4–6 worker processes, `torch.set_num_threads(2)` per worker. PC: High Performance power plan, sleep off. Mac: `caffeinate`; benchmark MPS vs CPU on 10 architectures first and use whichever wins.

---

## 6. Phase E — GP evolution and the learning curve

**Engine:** typed GP adapted from the 3C-EA `core.py` tree machinery (node builders, mutation, crossover, validate, infix printer). Types: {per-layer tensor} → elementwise ops (+, −, ×, safe-÷, abs, log(|·|+ε), sign, relu, square) → reductions to scalar (sum, mean, std, L1, L2, frac-positive) → cross-layer aggregation (sum, mean, min, max). Reject ill-typed offspring. Complexity penalty on node count. Depth cap from pilot timing.

**Budgets and splits.** For each evolution seed s ∈ {1..10} and each budget N ∈ {25, 50, 100, 200}: draw a **nested** chain (25 ⊂ 50 ⊂ 100 ⊂ 200) from the Space-A specialization pool, stratified by FLOPs, seeded by s. N=0 = evolved on the NB201 cache against NB201 accuracies (no speech data touched).

**Model-selection protocol (FROZEN 2026-08-05, before any evolution run):** at N ≥ 100, fitness = Spearman on the first 80% of the chain and elite selection on the remaining 20%, as originally planned. At N ≤ 50 an 80/20 split leaves ≤ 10 validation architectures — a Spearman there is near-noise (at N=25 it is 5 points) — so for N ≤ 50 fitness AND selection both use the full chain with the complexity penalty as the sole regularizer (small-sample standard practice; the honest check is that reported metrics are always computed on never-fitted Space B / A-test). The complexity penalty λ = 0.002/node applies at all budgets.
**Warm-start ablation:** at each N, compare cold-start populations vs populations seeded with N=0 (vision-evolved) elites.

**Reporting.** For every (seed, N): Spearman, Kendall τ, Precision@10% on (a) Space B (primary, never fitted), (b) A-test (within-space). Headline **Figure 1**: x = N (log scale incl. 0), y = Space-B Spearman, mean ± 95% CI over 10 seeds; flat reference lines = each standard proxy, #params, FLOPs (with their own bootstrap CIs); horizontal dashed line = test-retest noise ceiling.

**Early sanity gate (end of week 3):** at N=200, evolved proxies must exceed #params on Space B overall AND within FLOPs terciles, with CI separation. If not → §9 pivots.

**GATE OUTCOME (2026-08-05, artifact results/evaluation.json): FAILS → §9 pivot taken.** Evolved N=200 on Space B: 0.739 ± 0.017 vs #params 0.752 (Wilcoxon one-sided p=0.99, wrong direction) and FLOPs 0.879 [CI 0.837–0.910], which sits AT the noise ceiling (0.909) — on this space no proxy, standard or evolved, beats compute. 6/10 N=200 seeds rediscovered `asum(rl1(W))` (a size proxy); the two best seeds (0.771, 0.750) use the noise-gradient terminal Gn — speech-specific terminals ARE selected when they help (interpretability finding intact). What survives with CI separation: the TRANSFER result — vision-evolved 0.458 ± 0.192 vs N=50-speech-evolved 0.702 ± 0.060; small speech budgets close most of the transfer gap. A-test (within-space) N=200 = 0.824. Warm-start ≈ cold everywhere (ablation: no benefit). RE-CENTERED PAPER: (1) benchmark release, (2) vision-evolved proxies transfer poorly + N ≤ 50 closes most of the gap, (3) sobering headline for the ICASSP audience: FLOPs is at the noise ceiling on TC-ResNet KWS — trivial compute is a near-sufficient ranking statistic; state plainly that specialization did not beat trivial baselines at these budgets. Do not stretch claims.

Evolution runs over cached tensors cost seconds per candidate; pop 30 × 15 generations × 4 budgets × 10 seeds is hours-to-a-day per machine. Split seeds: Mac 6, PC 4.

---

## 7. Phase X — transfer matrix, search demo, statistics

**Figure 2 — transfer matrix (heatmap):** rows = where evolved (NB201, Space A @ N=200; optionally Space B for symmetry), columns = where evaluated (NB201 held-out, Space A-test, Space B). Include the best standard proxy as a reference row. If the §8 check confirms a non-vision NB-Suite-Zero task (e.g., NinaPro-on-NB201) with precomputed scores + accuracies, add it as a free established non-vision column — it blunts "everything non-vision here is self-made."

**Table 1 — correlations with CIs:** all proxies + evolved (per budget) on Space B: Spearman [95% CI], Kendall τ, P@10%; plus within-FLOPs-tercile Spearman (size-bias control). Wilcoxon signed-rank across the 10 evolution seeds for evolved-vs-best-baseline at each N (one-sided; p<.05 / p<.01 two-symbol convention; cite Derrac et al. 2011).

**Search demo (compact block):** aging evolution over Space B using (i) evolved proxy, (ii) best standard proxy, (iii) #params, (iv) random — best-found ground-truth accuracy vs #queries, mean ± CI over ≥20 repetitions (pure lookups). Punchline sentence: best discovered model's accuracy@params, with search cost in CPU-minutes on commodity hardware, contextualized against ICASSP 2021/2022 training-based KWS-NAS (explicitly labeled different protocol/version — context, not baseline).

**AMENDMENT (2026-08-10, per FEEDBACK.md P1.2) — protocol actually run:** aging evolution is not meaningful on a 200-arch tabular pool (mutations leave the trained set), so the demo is **proxy-rank-ordered querying**: architectures queried in descending proxy order, best-found ground-truth accuracy vs #queries; the random baseline is 1,000 shuffles with a 95% band (deterministic guides need no repetitions). Figure/caption language must match this amendment, not the original spec.
**AMENDMENT (2026-08-10, per FEEDBACK.md P1.1) — `evolved_consensus` defined:** the modal N=200 elite formula `asum(rl1(W))` (independently reached by 6/10 cold seeds; artifact `results/evolved/A_N200_s1.json`). Selected by evolution's own convergence — no holdout data touched. Space-B Spearman 0.741 [bootstrap CI 0.663–0.803] (= the shared value of the six identical-formula seeds in evaluation.json). Any figure using it cites this definition; if Table 1 retains it, this is its row.

**Bootstrap protocol:** percentile bootstrap over architectures, 10,000 resamples, for every correlation and P@k. Seeds and resample indices logged.

---

## 8. Verification to-dos (cheap, do before the corresponding paper sentences)

- [x] NB-Suite-Zero task list: 28 tasks confirmed; NinaPro (sEMG, non-vision, NOT audio) on NB201 with precomputed scores + accuracies — free transfer-matrix column. Suite has zero audio tasks (motivation). *(2026-08-04, see RELATED_WORK_NOTES.md)*
- [x] GreenMachine + GreenFactory: vision-only confirmed (NATS-Bench, CIFAR/ImageNet16); both arXiv-only, cite as preprints. *(2026-08-04)*
- [x] EZNAS: NB201 + NDS-DARTS evolution, NB201/NDS/NATS eval — vision-only confirmed. *(2026-08-04)*
- [x] Literature pass done: blanket "no ZC-NAS for speech" is FALSE — Abdelfattah ICLR'21 and **ProxyBO (AAAI 2023)** both evaluated on NB-ASR. Use the qualified phrasing in RELATED_WORK_NOTES.md; add ProxyBO to references. *(2026-08-04)*
- [x] Licenses: zero-cost-nas (via mohsaied mirror; original 404) and NASLib both Apache-2.0; attribute in code headers, keep notices. *(2026-08-04)*
- [ ] SC v2 12-class vs 35-class conventions; report exact split protocol in the paper. *(frozen protocol implemented in data_sc.py; write-up pending)*
- [x] NB-ASR recovery re-check: still unavailable as of 2026-08-04 (repo deleted, mirrors code-only/empty, nothing on HF).
- [x] NB-ASR download attempt (2026-08-10, exhaustive): all 8 release assets of tag v1.1.0 tried directly (404) and via Wayback (404); Wayback CDX index for releases/download/* is EMPTY — the binaries were never archived; recovery via archive is impossible. Gained: exact lost-asset inventory from the archived release page (snapshot 2022-04-01): nb-asr-e40-{1234,1235,1236}, e10-1234, e5-1234, info, bench-{gtx-1080ti,jetson-nano}-fp32 .pickle — cite this enumeration in the paper's footnote.
- [x] NB-ASR repo-state record (verified observations only; deletion history NOT asserted): front page 404 via API 2026-08-10 AND via HTML 2026-08-11; release page 404; 8/8 assets 404; Wayback CDX for the download path empty. An earlier claim of "front page live 2026-08-10" could not be reproduced on either date — discrepancy logged, weekly liveness monitor running through camera-ready (logs/nb_asr_liveness.log).

---

## 9. Kill criteria and pivots (with dates)

- **Pilot fails spread even at 35-class (by ~Aug 12):** the benchmark cannot rank proxies → fall back to the documented corruption-KWS plan (reliability-aware activations under Gilbert–Elliott loss); its spec is in the project history.
- **Noise ceiling too low (test-retest Spearman < 0.75):** single-seed ground truth is too noisy → add a second seed to all Space-B architectures (doubles B-sweep cost; still ≤ 1 day) before abandoning anything.
- **Week-3 sanity gate fails (evolved ≤ #params with CIs overlapping):** the finding-robust framing still holds ONLY if N=0 transfer is informative; re-center the paper on the transfer/generalization result and the benchmark release, and say plainly that specialization did not beat trivial baselines at these budgets. Do not stretch claims; do not attempt a pure negative-result 4-pager (FALE lesson).
- **NB-ASR pickles recovered before ~Aug 25:** add NB-ASR as a third evaluation column (validation gate per §4 applies first); do NOT re-plan the paper around it. After Aug 25: camera-ready/arXiv extension only. Notification is Jan 13, 2027 — recovered data can still enter the camera-ready.

---

## 9b. PRE-REGISTERED replication/expansion (declared 2026-08-10; SUPERSEDED 2026-08-11 by the revised plan — replication now runs PRE-submission)

**AMENDMENT (2026-08-11, verbatim intent of the revised plan; committed before the first replication architecture was sampled):** the replication sets are trained NOW (target: sweeps done ~Aug 16–18, sealed evaluation by Aug 20), not post-acceptance. Everything else pinned on 2026-08-10 stands unchanged: seed streams `random.Random("Atest-expansion-2027")` (150 new A-test-rep archs) and `random.Random("B-replication-2027")` (200 new Space-B-rep archs); arch_id disjointness against the released benchmark; grammar v2.1 and recipe pilot-v2 UNCHANGED; statistics caches per §5; evaluated ONCE by evaluate.py after ALL replication training completes; reported as replication BESIDE the original sets, never merged. **Honesty clause:** if the tighter precision places FLOPs below the ceiling rather than at it — or weakens any headline claim — that is reported as-is and the manuscript wording updated. Artifacts: results/replication.json + delta table vs originals.

**What:** (1) expand A-test from 50 to 200: 150 NEW Space-A architectures; (2) add a disjoint Space-B replication set of 200 NEW architectures. Both sampled by the FROZEN grammar v2.1 samplers, deduplicated by arch_id against ALL previously sampled architectures of their space (including rejected/capped ones is not required — arch_id disjointness against the released benchmark is), trained with the FROZEN recipe pilot-v2, seed 0, on the Mac (one overnight, ~350 models).

**Seed streams (pinned now):** `random.Random("Atest-expansion-2027")` and `random.Random("B-replication-2027")`, consumed by the existing `sample_archs` machinery with rejection of already-released arch_ids.

**Evaluation:** the frozen proxies and the ALREADY-COMMITTED evolved elites (no re-evolution, no re-selection, no new model choices of any kind) are scored on the new sets by `evaluate.py`, run once, same bootstrap protocol; results reported as REPLICATION columns in the camera-ready/arXiv version, labeled as such, whichever way they land. Any change to grammar, recipe, or elite set voids the replication claim.

**Purpose:** shrinks A-test CIs from ±~0.13 to ±~0.06 and tests the headline findings (FLOPs-at-ceiling; specialization curve ordering) on data that did not exist at submission time.

**REPLICATION OUTCOME (2026-09-02; artifacts results/replication.json + replication.md; sealed run at git fe4fb236):** 150 A_rep + 200 B_rep trained 2026-09-02 (0 invalid; spread 25.9–94.1% / 56.4–94.6%, matching originals; arch_id disjointness verified A∩A_rep = B∩B_rep = A_rep∩B_rep = 0; trained sets equal the committed splits). Caches per §5 (`cache_stats --target A_rep|B_rep`). `evaluate.py --replication` run ONCE, after a selfcheck that the replication code path reproduces evaluation.json on the original Space B (11/11 proxies, ρ and CI exact). **REPLICATES:** (1) specialization curve — B_rep N=0 0.524±0.149 → N=50 0.714±0.049 → N=200 0.745±0.013, every budget within ±0.02 of the original except N=0 (+0.07); A_rep N=200 0.825 (orig 0.824); (2) no proxy beats FLOPs on B — FLOPs 0.834 > nwot 0.810 > params 0.755 ≈ evolved 0.745; Wilcoxon evolved>params p≥0.985 at every budget; (3) family dependence, sharpened by A-test n 50→150 — A_rep params 0.823 [0.76, 0.87] vs FLOPs 0.445 [0.31, 0.56], CI-separated; synflow 0.838 is now the best fixed ranker on A; (4) tercile mechanism — B_rep params 0.80/0.12/0.04, nwot 0.54/0.39/0.55, evolved N=200 0.76/0.10/0.06. **HONESTY CLAUSE FIRED:** FLOPs on B_rep = 0.834 [0.772, 0.878]; the CI no longer contains the 0.909 ceiling (original 0.879 [0.837, 0.910] did). Manuscript wording "at the ceiling" / "statistically indistinguishable from the ceiling" (abstract, contribution 4, §5.2 body, §6, conclusion) must become "near" / "within 0.05–0.08 of the ceiling point estimate". The ceiling's own bootstrap CI is [0.799, 0.964] (n=50), so at-vs-below is unresolved at current ceiling precision; only a pre-registered extra-seeds run (~1 h on B) can resolve it. §5.2's title ("nothing beats trivial compute on this space") stands. Replication sets stay reported beside the originals, never merged.

**PRE-REGISTERED SEED EXPANSION ("9b-seeds"; declared 2026-09-02 BEFORE any run; adversarially reviewed, then committed — the commit carrying this paragraph precedes the first new seed file):** *Purpose.* The test–retest ceiling (0.909; n=50; bootstrap CI ≈ [0.80, 0.96]) is the least precise number in the paper, and Space A has no ceiling at all; the replication put FLOPs on B_rep at 0.834 [0.772, 0.878], so at-vs-below-ceiling is unresolvable at the current ceiling precision. *Runs* (recipe pilot-v2 UNCHANGED; seeds 1 and 2, exactly as the original noise subset; Mac/MPS — `--device mps` is part of every pinned command; launched in this order, each resume-safe): (1) original Space B, sweep indices 50–199 (150 archs) → all 200 Space-B archs at 3 seeds; (2) A-test — the 50 archs under "test" in splits/a_pool_test.json, written to results/gt_A/<id>_s{1,2}.json carrying the arch's Space-A sweep index and `seed_role: ceiling-only`; (3) B_rep (200); (4) A_rep (150). Commands: `train_gt --space B --count 200 --range 50:200 --seed {1,2} --device mps`; `replication --train {Atest,B_rep,A_rep} --seed {1,2} --device mps`. *Invariant.* The ground truth behind every proxy / evolved correlation REMAINS seed 0. Seeds 1–2 are used ONLY to estimate ceilings: no targets are averaged, no ranker is re-scored, evaluation.json and replication.json are untouched by this amendment. *What the seeds vary.* Weight initialization and minibatch order (the frozen recipe's seed); the 10k stratified training subsample, the official val/test splits and the cached features are fixed for all seeds — the ceiling is therefore a test–retest reliability under the frozen recipe, an approximate (not strict) upper bound on proxy–accuracy rank correlation. *Ceiling definition* (unchanged from the original estimate): mean of the three pairwise Spearman correlations of seed-wise test accuracies over the architectures of a set; percentile bootstrap over architectures, 10k resamples, seed 0, with a content-defined architecture order (sample_archs position for A/B, the sealed A-test list, the replication split order — never filesystem order). *Computed ONCE* by `evaluate.py --ceiling` after all four runs complete (the flag refuses to overwrite an existing results/noise_ceiling.json) → reporting: B (n=200), B_orig50 (the original 50-arch subset, for traceability — must equal 0.9092), Atest (n=50), B_rep (n=200), A_rep (n=150). A set whose seeds are incomplete is reported with its actual n, never padded or dropped; architectures with non-finite accuracies are dropped and counted in the artifact. `evaluate.py --ceiling-selfcheck` (B_orig50 must reproduce 0.9092; writes nothing) runs before the sealed computation. *Decision pairing (pinned).* PRIMARY verdict = original Space B: FLOPs 0.879 [0.837, 0.910] (evaluation.json) vs the n=200 Space-B ceiling — the same 200 architectures and the same seed-0 targets. REPLICATION verdict = B_rep: FLOPs 0.834 [0.772, 0.878] (replication.json) vs the B_rep ceiling. If the two verdicts disagree, the paper states both, primary first. Rule: ceiling CI entirely above FLOPs' CI → "below the ceiling"; CIs overlap → "near the ceiling"; FLOPs' CI contains the ceiling point estimate → "at the ceiling". *Supplementary, reported but not deciding:* the paired-bootstrap 95% CI (identical resamples) of ceiling − Spearman(FLOPs, seed-0 accuracy) on the same architectures, written to the same artifact (Space-A sets: the same statistic against #params). *Reporting rule.* Every Space-B ceiling reference in the paper (Table 1 row, Fig. 1 dashed line, §3 prose, §5.2, §6, conclusion) uses the n=200 estimate with its CI; 0.909 (n=50) remains only as the traceability value; the B_rep and Space-A ceilings are reported beside it, labeled; nothing is merged. *Pre-launch environment check (done 2026-09-02, before run (1)).* Three architectures of the original noise subset (Space-B sweep indices 3, 25, 41) retrained at seed 1 on the current environment reproduce their stored 2026-08-05 files exactly — test-accuracy difference 0.00 pts, identical epoch counts (results/envcheck_seed1_retrain.json) — so the n=200 ceiling mixes no numeric drift between the August and September seed runs. *Honesty clause.* The at/near/below wording follows the pinned rule wherever the ceilings land; nothing else is changed by this amendment.

**9b-SEEDS OUTCOME (2026-09-03; artifact results/noise_ceiling.json at git 412445e; 1,100 seed runs, 0 non-finite; selfcheck B_orig50 = 0.9092 OK; comparator cross-checks vs evaluation.json / replication.json exact):** ceilings — **B (n=200) 0.953 [0.922, 0.972]**, seed-σ median 0.39 pts (pairwise 0.948 / 0.966 / 0.944); B_orig50 0.909 [0.800, 0.965] (traceability only); **B_rep (n=200) 0.931 [0.885, 0.960]**; **Atest (n=50) 0.970 [0.940, 0.980]**; **A_rep (n=150) 0.979 [0.967, 0.984]**. **PINNED RULE APPLIED → "BELOW THE CEILING", both verdicts agree:** PRIMARY (original B) FLOPs [0.837, 0.910] vs ceiling [0.922, 0.972] — ceiling CI entirely above; paired gap +0.073 [+0.036, +0.114], P(gap≤0)=0.0003. REPLICATION (B_rep) FLOPs [0.772, 0.878] vs ceiling [0.885, 0.960]; gap +0.096 [+0.038, +0.159]. Space A: #params 0.833 / 0.823 sit 0.14 / 0.16 below its ceilings, CI-separated. **Consequences (mechanical, per the amendment):** every Space-B ceiling reference in the paper (Table 1 row, Fig. 1 dashed line, §3, §5.2, §6, conclusion, abstract) → 0.953 [0.922, 0.972]; 0.909 appears only as the n=50 traceability value; "at the ceiling" / "statistically indistinguishable" / "near-sufficient" → "below the ceiling": trivial compute is the best available ranker on Space B and no proxy beats it, but it leaves a resolved 0.07 [0.04, 0.11] of rank correlation (0.10 on replication) that no proxy — evolved or fixed — captures: bounded headroom, not sufficiency. §5.2's title ("nothing beats trivial compute on this space") stands. The benchmark now carries three seeds on all 600 evaluation architectures (B, A-test, B_rep, A_rep): 800 architectures, 2,000 training runs.

**PRE-REGISTERED SECOND REPLICATION ("9b-rep2"; declared 2026-09-03 BEFORE any sampling or training; brings the benchmark to 1,000 architectures):** *What.* 100 new Space-A architectures (A_rep2) and 100 new Space-B architectures (B_rep2), sampled by the FROZEN grammar v2.1 samplers from the pinned streams `random.Random("Atest-expansion-2027-r2")` and `random.Random("B-replication-2027-r2")`, rejecting any arch_id present in the released benchmark (A 250 / B 200) or in the first replication (A_rep 150 / B_rep 200); trained with recipe pilot-v2 UNCHANGED at seeds 0, 1, 2 (Mac/MPS, `--device mps`) in this order: A_rep2 s0, B_rep2 s0, then seeds 1 and 2 for both. Totals on completion: Space A 500 architectures, Space B 500 — 1,000 architectures, three seeds on all 800 evaluation architectures, 2,600 training runs. *Evaluation.* The frozen proxies and the ALREADY-COMMITTED elites (no re-evolution, no re-selection, no new model choices) are scored ONCE by `evaluate.py --replication2` after both seed-0 sweeps and their §5 caches complete → results/replication2.json + replication2.md, reporting (a) A_rep2 and B_rep2 alone and (b) the POOLED replication sets A_rep ∪ A_rep2 (n=250) and B_rep ∪ B_rep2 (n=300) — the pooling is declared here, before rep2 exists; pooled standard-proxy scores are the union of the per-set cached scores (nothing recomputed). If the paper carries a replication column it reports the pooled sets labeled with their n; nothing is ever merged with the originals, which remain Table 1's primary numbers. *Selfcheck before the sealed run:* the generalized replication code path, re-run on {B_rep, A_rep}, must reproduce replication.json exactly. *Ceilings.* After seeds 1–2 complete, `evaluate.py --ceiling --extend` appends A_rep2, B_rep2 and the pooled sets to results/noise_ceiling.json without recomputing or altering any existing block (definition, bootstrap and ordering rule unchanged; ordering = the rep2 split order, pooled = rep order followed by rep2 order). *Honesty clause.* Whatever rep2 shows is reported as-is beside the originals and the first replication; the pinned at/near/below rule is applied additionally to the pooled Space-B replication (FLOPs on the pooled set vs the pooled ceiling) and reported beside the two existing verdicts.

**9b-rep2 OUTCOME (2026-09-05; artifacts results/replication2.json + .md (commit 1b35cf4) and results/noise_ceiling.json extended, both at code git 496389b; 600 rep2 runs, 0 invalid; selfchecks 11/11 and 202/202 exact; comparator cross-checks exact):** rep2 alone (n=100 each) — B_rep2: FLOPs 0.823 [0.725, 0.888] > nwot 0.813 > params 0.719 ≈ evolved N=200 0.705±0.021; N=0 0.316±0.269 → N=50 0.654; A_rep2: params 0.818 ≈ evolved N=200 0.824 > FLOPs 0.667. **POOLED rep ∪ rep2 (the paper's replication column, pre-declared):** Space B (n=300): FLOPs 0.838 [0.793, 0.874] > nwot 0.819 [0.770, 0.857] > params 0.744 ≈ evolved N=200 0.732±0.016 (below FLOPs' CI); curve N=0 0.462 → N=50 0.695 → N=200 0.732; Wilcoxon evolved>params p≥0.986 at every budget; terciles params 0.72/0.17/−0.02, nwot 0.57/0.40/0.51, evolved N=200 0.69/0.16/−0.03. Space A (n=250): params 0.821 [0.768, 0.862] vs FLOPs 0.538 [0.445, 0.623], CI-separated; evolved N=200 0.826±0.038. Ceilings: B_rep2 0.951 [0.911, 0.973]; A_rep2 0.979 [0.958, 0.987]; **B pooled 0.939 [0.908, 0.960]**; A pooled 0.980 [0.972, 0.985]; the sealed 9b-seeds blocks are byte-identical after --extend. **PINNED RULE on pooled B → "BELOW THE CEILING"** (ceiling CI entirely above FLOPs' CI; paired gap +0.101 [+0.060, +0.146]) — all three verdicts (original, first replication, pooled replication) agree. Every qualitative finding replicates a second time; both replications put FLOPs/nwot ≈0.04 below the original draw. **Benchmark totals: 1,000 architectures (A 500 / B 500), three seeds on all 800 evaluation architectures, 2,600 training runs.** Original sets remain Table 1's primary numbers; the paper's replication numbers are the pooled sets (250 A / 300 B).

## 10. Timeline (today: Aug 4)

- **W1 (Aug 4–10):** §4 proxy suite + NB201 validation gate; `spaces.py` grammars; pilot (10+10 archs, both machines); launch full ground-truth sweep on PASS. §8 checks in parallel.
- **W2 (Aug 11–17):** finish sweeps + noise subset; build all caches (speech + NB201); typed-GP engine + unit tests (known formulas reproduce hand-computed scores on toy nets).
- **W3 (Aug 18–24):** learning-curve runs (all budgets × 10 seeds); week-3 sanity gate; warm-start ablation.
- **W4 (Aug 25–31):** transfer matrix; search demo; Table 1 + all bootstrap/Wilcoxon machinery; freeze figures.
- **W5 (Sep 1–7):** numbers frozen; full draft (intro leads with the lost-benchmark motivation); 100% code–paper consistency pass (every number ↔ artifact).
- **W6 (Sep 8–16):** polish, co-author/advisor review, Zenodo/HF DOI minted, submit ≤ Sep 16, target Sep 9.

## 11. Machines

- **PC (Ryzen 5 5600G, 24 GB, Windows):** ground-truth sweeps, cache building, 4 evolution seeds. CPU-only — the RX 6500 XT is unusable for PyTorch (no ROCm for Navi 24; DirectML not research-grade). Enable XMP/DOCP (RAM currently 2133 → target 3200).
- **MacBook M3 Pro:** development, 6 evolution seeds, analysis, figures, paper. MPS-vs-CPU decided by the 10-arch benchmark, per workload.

## 12. Repo layout

```
speech_zcp/
  spaces.py        # Space A/B sampling grammars (seeded, released)
  train_gt.py      # frozen ground-truth training recipe + sweep runner
  proxies.py       # standard suite + NB201 validation gate tests
  bench_201.py     # NB201 sample loader; NB-Suite-Zero score/accuracy access
  cache_stats.py   # statistics extraction workers (speech + vision)
  gp_engine.py     # typed GP (adapted from 3C-EA core.py) + unit tests
  evolve.py        # budget-curve evolution entry point (CONFIG dict at top)
  evaluate.py      # the ONLY script that touches Space B / A-test; bootstrap + Wilcoxon
  search_demo.py   # aging-evolution demo over precomputed scores
  results/  cache/  logs/  figures/  splits/
```

## 13. Key references

Abdelfattah, Mehrotra, Dudziak, Lane — Zero-Cost Proxies for Lightweight NAS, ICLR 2021 (incl. NB-ASR results). Mehrotra et al. — NAS-Bench-ASR, ICLR 2021. Akhauri et al. — EZNAS, NeurIPS 2022. GreenMachine (2024); GreenFactory (2025). Mellor et al. — NAS without Training, ICML 2021. Krishnakumar et al. — NAS-Bench-Suite-Zero, NeurIPS 2022. TPAMI zero-shot NAS survey (2024). SANAS, ICASSP 2019; Zhang et al., ICASSP 2021; Peter, Roth, Pernkopf, ICASSP 2022 (KWS-NAS precedents). Zhang et al. — Hello Edge (DS-CNN). Choi et al. — TC-ResNet, Interspeech 2019. Warden — Speech Commands. Real et al. — aging evolution. Derrac et al. 2011 — statistical protocol.
