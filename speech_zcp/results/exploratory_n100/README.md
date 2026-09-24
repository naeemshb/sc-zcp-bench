# Exploratory N=100 search-setting variant (9f) — not part of the benchmark or the paper's tables

A post-hoc variant of the N=100 evolution (population 50, 20 generations, depth cap 3, 25 seeds;
everything else identical to the frozen runs), pre-registered as exploratory before its single sealed
scoring (`PREREGISTRATION_9f.md`). Result on sealed Space B: 0.644 +- 0.196 over 25 seeds versus the
frozen N=100 row's 0.669 +- 0.177 — no improvement, so the frozen row stands. The paper mentions this
run only as one further scoring of the sealed set.

- `elites/`          the 25 elite records produced by `code/evolve_explore.py` (fitting side; no sealed set read)
- `sealed_9f.json`   the single sealed scoring on Space B, A-test, pooled B replication and A_rep,
                     produced by `speech_zcp.evaluate.run_elite_dir` after its 40/40 self-check
- `sealed_report.md` per-seed table and summary of that scoring
- `summary.{json,md}` fitting-side summary of the 25 runs
- `code/`            the search script as run (a copy of evolve.py with the changed settings) and the
                     engine copy it imported; they were executed from a directory at the repository root
                     and are kept for the record, not as an entry point

Commit hashes cited inside these files refer to the pre-release history; see `COMMIT_HASH_MAP.txt`.
