"""Figure 2: cross-space / cross-modality transfer matrix (PROTOCOL.md section 7).

Rows: evolved proxies (vision N=0; Space A N=200) + reference proxies
(flops, nwot, params). Columns: NB201 held-out (200 archs outside the
evolution sample), NinaPro-on-NB201 (NB-Suite-Zero precomputed, non-vision,
established), A-test (50), Space B (200).

Cell = Spearman rho vs ground-truth accuracy. Evolved rows average over the
seeds whose formulas are computable on that column (speech-only terminals
Gn/An/AstdT/AnstdT do not exist in vision caches; excluded seeds counted in
the artifact). Evolved x NinaPro is structurally N/A (no raw statistics
exist for NinaPro). Speech cells reuse evaluation.json — no holdout re-runs.

Artifacts: results/transfer_matrix.json, figures/fig2_transfer_matrix.png.
"""

import glob
import json
import os
import random

import numpy as np
import scipy.stats as st
import torch

from . import gp_engine as gp
from .cache_stats import CACHE_DIR, load_arch
from .evaluate import load_ctx
from .evolve import tree_from_json
from .proxies import compute_proxy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
EVOLVED_DIR = os.path.join(RESULTS_DIR, "evolved")


def nb201_holdout_cells():
    idx = json.load(open(os.path.join(CACHE_DIR, "nb201_holdout_index.json")))["archs"]
    aids = sorted(idx, key=lambda a: idx[a]["i"])
    accs = [idx[a]["acc"] for a in aids]
    ctxs = [load_ctx("nb201_holdout", a) for a in aids]

    from .bench_201 import build_nb201, fixed_cifar_batch

    xc, yc = fixed_cifar_batch()
    x, y = xc[:8], yc[:8]
    std = {}
    for p in ["flops", "nwot", "params"]:
        scores = [compute_proxy(p, lambda: build_nb201(idx[a]["arch_str"]), x, y) for a in aids]
        std[p] = float(st.spearmanr(scores, accs).statistic)

    evolved = {}
    for variant, pattern in [("vision_N0", "nb201_N0_s*.json"), ("speechA_N200", "A_N200_s*.json")]:
        rhos, skipped = [], 0
        for f in sorted(glob.glob(os.path.join(EVOLVED_DIR, pattern))):
            if "_warm" in f:
                continue
            r = json.load(open(f))
            if any(t not in ("W", "G", "A") for t in r["terminals_used"]):
                skipped += 1  # speech-only terminals: not computable on vision
                continue
            scores = gp.scores_for(tree_from_json(r["tree_json"]), ctxs)
            if scores:
                rhos.append(float(st.spearmanr(scores, accs).statistic))
        evolved[variant] = {"rho": float(np.mean(rhos)), "n_seeds": len(rhos), "skipped": skipped}
    return std, evolved


def ninapro_cells(n_sample=500):
    zc = json.load(open("data/nbs_zero/zc_nasbench201.json"))["ninapro"]
    pool = sorted(zc.keys())
    keys = random.Random(0).sample(pool, min(n_sample, len(pool)))
    print(f"ninapro: {len(keys)} archs available", flush=True)
    accs = [zc[k]["val_accuracy"] for k in keys]
    out = {}
    for p in ["flops", "nwot", "params"]:
        scores = [zc[k][p]["score"] for k in keys]
        out[p] = float(st.spearmanr(scores, accs).statistic)
    return out


def speech_cells():
    ev = json.load(open(os.path.join(RESULTS_DIR, "evaluation.json")))
    out = {"std": {}, "evolved": {}}
    for p in ["flops", "nwot", "params"]:
        out["std"][p] = {c: ev["standard"][p][c]["spearman"] for c in ["Atest", "B"]}
    for variant, budget in [("vision_N0", 0), ("speechA_N200", 200)]:
        for col in ["Atest", "B"]:
            rhos = [v[col]["spearman"] for k, v in ev["evolved"].items()
                    if v["budget"] == budget and not k.endswith("_warm")
                    and v[col].get("spearman") is not None]
            out["evolved"].setdefault(variant, {})[col] = {
                "rho": float(np.mean(rhos)), "n_seeds": len(rhos)}
    return out


def main():
    torch.set_num_threads(max(2, os.cpu_count() - 2))
    print("nb201 holdout cells...", flush=True)
    std_nb, ev_nb = nb201_holdout_cells()
    print("ninapro cells...", flush=True)
    nina = ninapro_cells()
    sp = speech_cells()

    rows = ["evolved: vision (N=0)", "evolved: Space A (N=200)", "FLOPs", "nwot", "#params"]
    cols = ["NB201 held-out", "NinaPro*", "A-test", "Space B"]
    M = np.full((5, 4), np.nan)
    M[0] = [ev_nb["vision_N0"]["rho"], np.nan,
            sp["evolved"]["vision_N0"]["Atest"]["rho"], sp["evolved"]["vision_N0"]["B"]["rho"]]
    M[1] = [ev_nb["speechA_N200"]["rho"], np.nan,
            sp["evolved"]["speechA_N200"]["Atest"]["rho"], sp["evolved"]["speechA_N200"]["B"]["rho"]]
    for i, p in enumerate(["flops", "nwot", "params"]):
        M[2 + i] = [std_nb[p], nina[p], sp["std"][p]["Atest"], sp["std"][p]["B"]]

    art = {"rows": rows, "cols": cols, "matrix": M.tolist(),
           "nb201_holdout_detail": ev_nb, "ninapro_source": "NB-Suite-Zero precomputed (500-arch seed-0 sample)",
           "note": "evolved x NinaPro N/A: no raw statistics exist; evolved NB201 cells average only "
                   "seeds without speech-only terminals (skipped counts in nb201_holdout_detail)"}
    json.dump(art, open(os.path.join(RESULTS_DIR, "transfer_matrix.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    for i in range(5):
        for j in range(4):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="black")
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("Transfer matrix: Spearman rho vs ground truth", fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.text(0.01, 0.01, "*NinaPro: NB-Suite-Zero precomputed scores (sEMG, non-vision); "
             "no raw statistics -> evolved rows n/a", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_transfer_matrix.png"))
    print("\nrows x cols:")
    for r, row in zip(rows, M):
        print(f"{r:26s} " + "  ".join("  n/a" if np.isnan(v) else f"{v:5.2f}" for v in row))


if __name__ == "__main__":
    main()
