"""P0.1 (FEEDBACK.md): within-FLOPs-tercile correlations for EVERY ranker.

Post-evaluation re-analysis of the already-spent holdouts: evolved scores are
recomputed deterministically from the frozen caches (identical to what
evaluate.py scored); ground truth is the committed benchmark. No model or
config selection happens here — only the pre-registered framing decision:

  DECISION RULE (verbatim from FEEDBACK.md P0.1): if evolved (N=200 AND N=50)
  exceeds BOTH #params and FLOPs within >= 2 of 3 Space-B terciles with CI
  separation -> TWO-ACT FRAMING VIABLE, else SOBER FRAMING FINAL.

  Operationalization (recorded here before results were seen): per budget and
  tercile, evolved = mean rho over the 10 cold seeds with a t-based 95% seed
  interval; baseline = 10k paired-percentile-bootstrap 95% CI over
  architectures (shared resample indices across rankers, seed 0); "exceeds
  with CI separation" = evolved seed-interval lower bound > baseline bootstrap
  upper bound.

Artifacts: results/terciles.json, results/terciles.md (rendered table + flag).
"""

import glob
import json
import os

import numpy as np
import scipy.stats as st

from . import gp_engine as gp
from .cache_stats import CACHE_DIR
from .evaluate import gt_map, load_ctx
from .evolve import a_pool_and_test, tree_from_json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
EVOLVED_DIR = os.path.join(RESULTS_DIR, "evolved")
N_BOOT = 10_000
BOOT_SEED = 0


def _rank(a: np.ndarray) -> np.ndarray:
    return st.rankdata(a, axis=-1)


def boot_ci(s, a, idx):
    """Vectorized percentile bootstrap of Spearman over shared resamples."""
    sr, ar = _rank(s[idx]), _rank(a[idx])
    sr = sr - sr.mean(-1, keepdims=True)
    ar = ar - ar.mean(-1, keepdims=True)
    num = (sr * ar).sum(-1)
    den = np.sqrt((sr**2).sum(-1) * (ar**2).sum(-1))
    rhos = num / den
    return [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))]


def tercile_masks(flops):
    f = np.asarray(flops)
    lo, hi = np.percentile(f, [33.3, 66.7])
    return [f < lo, (f >= lo) & (f < hi), f >= hi], [lo, hi]


def eval_target(name, ids, accs, flops, std_scores, evolved_scores):
    accs = np.asarray(accs, float)
    masks, cuts = tercile_masks(flops)
    rng = np.random.default_rng(BOOT_SEED)
    out = {"n": len(ids), "tercile_cuts_flops": cuts, "rankers": {}}
    shared_idx = []
    for m in masks:
        n = int(m.sum())
        shared_idx.append(rng.integers(0, n, size=(N_BOOT, n)))
    for rname, scores in std_scores.items():
        s = np.asarray([scores[a] for a in ids], float)
        cells = []
        for m, idx in zip(masks, shared_idx):
            if not np.all(np.isfinite(s[m])):
                cells.append(None)
                continue
            rho = float(st.spearmanr(s[m], accs[m]).statistic)
            cells.append({"rho": rho, "ci95": boot_ci(s[m], accs[m], idx), "n": int(m.sum())})
        out["rankers"][rname] = {"type": "fixed", "terciles": cells}
    for rname, seed_score_lists in evolved_scores.items():
        per_seed = []
        for s in seed_score_lists:
            s = np.asarray(s, float)
            per_seed.append([
                float(st.spearmanr(s[m], accs[m]).statistic) if np.all(np.isfinite(s[m])) else None
                for m in masks
            ])
        cells = []
        for t in range(3):
            vals = [ps[t] for ps in per_seed if ps[t] is not None]
            if len(vals) < 2:
                cells.append(None)
                continue
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            half = float(st.t.ppf(0.975, len(vals) - 1) * sd / np.sqrt(len(vals)))
            cells.append({"rho": mean, "seed_ci95": [mean - half, mean + half],
                          "n_seeds": len(vals), "n": int(masks[t].sum())})
        out["rankers"][rname] = {"type": "evolved", "terciles": cells}
    return out


def main():
    gtB, gtA = gt_map("B"), gt_map("A")
    b_index = json.load(open(os.path.join(CACHE_DIR, "B_index.json")))
    b_ids = sorted(b_index["archs"], key=lambda a: b_index["archs"][a]["i"])
    _, atest_ids = a_pool_and_test()

    stdB = json.load(open(os.path.join(RESULTS_DIR, "proxy_scores_B.json")))
    stdA = json.load(open(os.path.join(RESULTS_DIR, "proxy_scores_Atest.json")))

    print("loading ctxs + scoring evolved trees...", flush=True)
    ctxB = [load_ctx("B", a) for a in b_ids]
    ctxA = [load_ctx("A", a) for a in atest_ids]
    evolvedB, evolvedA = {}, {}
    for budget in [0, 25, 50, 100, 200]:
        pat = f"nb201_N0_s*.json" if budget == 0 else f"A_N{budget}_s*.json"
        for f in sorted(glob.glob(os.path.join(EVOLVED_DIR, pat))):
            if "_warm" in f:
                continue
            tree = tree_from_json(json.load(open(f))["tree_json"])
            key = f"evolved_N{budget}"
            sb, sa = gp.scores_for(tree, ctxB), gp.scores_for(tree, ctxA)
            if sb:
                evolvedB.setdefault(key, []).append(sb)
            if sa:
                evolvedA.setdefault(key, []).append(sa)

    res = {
        "B": eval_target("B", b_ids, [gtB[a]["test_acc"] for a in b_ids],
                         [gtB[a]["flops"] for a in b_ids], stdB, evolvedB),
        "Atest": eval_target("Atest", atest_ids, [gtA[a]["test_acc"] for a in atest_ids],
                             [gtA[a]["flops"] for a in atest_ids], stdA, evolvedA),
        "protocol": {"n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                     "note": "shared resample indices across rankers (paired bootstrap); "
                             "evolved cells: mean over cold seeds, t-based 95% seed interval"},
    }

    # ---- pre-registered decision rule ----
    def separated_above(ev_cell, base_cell):
        return (ev_cell and base_cell
                and ev_cell["seed_ci95"][0] > base_cell["ci95"][1])

    verdicts = {}
    for budget in [50, 200]:
        wins = 0
        for t in range(3):
            ev = res["B"]["rankers"][f"evolved_N{budget}"]["terciles"][t]
            if (separated_above(ev, res["B"]["rankers"]["params"]["terciles"][t])
                    and separated_above(ev, res["B"]["rankers"]["flops"]["terciles"][t])):
                wins += 1
        verdicts[f"N{budget}"] = wins
    two_act = verdicts["N50"] >= 2 and verdicts["N200"] >= 2
    flag = "TWO-ACT FRAMING VIABLE" if two_act else "SOBER FRAMING FINAL"
    res["decision"] = {"rule": "FEEDBACK.md P0.1", "tercile_wins": verdicts, "flag": flag}
    json.dump(res, open(os.path.join(RESULTS_DIR, "terciles.json"), "w"), indent=1)

    # ---- rendered markdown ----
    lines = ["# Within-FLOPs-tercile Spearman (P0.1)", "",
             f"Bootstrap: {N_BOOT} paired resamples, seed {BOOT_SEED}. "
             "Evolved cells: mean over 10 cold seeds [t-based 95% seed interval]. "
             "Fixed cells: rho [bootstrap 95% CI].", ""]
    for tgt, label in [("B", "Space B (n=200; tercile n≈67)"),
                       ("Atest", "A-test (n=50; tercile n≈17 — wide CIs, reported per P1.3)")]:
        lines += [f"## {label}", "",
                  "| Ranker | T1 (small) | T2 (mid) | T3 (large) |", "|---|---|---|---|"]
        order = ["flops", "nwot", "params", "zen", "l2_norm", "snip", "synflow",
                 "grad_norm", "fisher", "grasp", "plain",
                 "evolved_N0", "evolved_N25", "evolved_N50", "evolved_N100", "evolved_N200"]
        for rname in order:
            r = res[tgt]["rankers"].get(rname)
            if r is None:
                continue
            row = [rname]
            for c in r["terciles"]:
                if c is None:
                    row.append("n/a")
                elif "ci95" in c:
                    row.append(f"{c['rho']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]")
                else:
                    row.append(f"{c['rho']:.2f} [{c['seed_ci95'][0]:.2f}, {c['seed_ci95'][1]:.2f}]")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += [f"## Decision (pre-registered rule, applied {__import__('datetime').date.today()})",
              "",
              f"Tercile wins with CI separation over BOTH params and flops: "
              f"N=50 -> {verdicts['N50']}/3, N=200 -> {verdicts['N200']}/3.",
              f"**FLAG: {flag}**"]
    open(os.path.join(RESULTS_DIR, "terciles.md"), "w").write("\n".join(lines))
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
