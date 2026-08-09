"""Training-free search demo over Space B (PROTOCOL.md section 7).

Protocol: proxy-rank-ordered querying over the 200 precomputed Space-B
architectures (pure lookups — the plan's "aging evolution" is not meaningful
on a 200-arch tabular pool where mutations leave the trained set; rank-ordered
querying is the standard tabular protocol, cf. Abdelfattah et al. warmup).

For each guide: query architectures in descending proxy order and track the
best ground-truth test accuracy found after k queries. Random baseline:
mean +- 95% CI over 1000 shuffles. The "evolved" guide is the N=200 CONSENSUS
formula asum(rl1(W)) — 6/10 evolution seeds converged to it, so its selection
required no holdout peeking.

Artifacts: results/search_demo.json, figures/fig3_search_demo.png.
"""

import json
import os

import numpy as np

from . import gp_engine as gp
from .cache_stats import CACHE_DIR
from .evaluate import gt_map, load_ctx
from .evolve import tree_from_json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
K_MAX = 50
N_RANDOM_REPS = 1000
# the N=200 consensus formula asum(rl1(W)) — 6/10 seeds converged to it;
# loaded from seed 1's artifact so the tree JSON is provenance-exact
CONSENSUS_ARTIFACT = os.path.join(RESULTS_DIR, "evolved", "A_N200_s1.json")


def best_found_curve(order, accs, k_max=K_MAX):
    best, out = -1.0, []
    for aid in order[:k_max]:
        best = max(best, accs[aid])
        out.append(best)
    return out


def main():
    gtB = gt_map("B")
    b_index = json.load(open(os.path.join(CACHE_DIR, "B_index.json")))
    b_ids = sorted(b_index["archs"], key=lambda a: b_index["archs"][a]["i"])
    accs = {a: gtB[a]["test_acc"] for a in b_ids}
    oracle = max(accs.values())

    std = json.load(open(os.path.join(RESULTS_DIR, "proxy_scores_B.json")))
    consensus = json.load(open(CONSENSUS_ARTIFACT))
    assert consensus["tree"] == "asum(rl1(W))", consensus["tree"]
    tree = tree_from_json(consensus["tree_json"])
    ctxs = [load_ctx("B", a) for a in b_ids]
    evolved_scores = dict(zip(b_ids, gp.scores_for(tree, ctxs)))

    guides = {
        "evolved_consensus": evolved_scores,
        "flops": {a: std["flops"][a] for a in b_ids},
        "nwot": {a: std["nwot"][a] for a in b_ids},
        "params": {a: std["params"][a] for a in b_ids},
    }
    out = {"oracle_acc": oracle, "k_max": K_MAX, "curves": {}}
    for name, scores in guides.items():
        order = sorted(b_ids, key=lambda a: -scores[a])
        curve = best_found_curve(order, accs)
        q_hit = next((k + 1 for k, v in enumerate(curve) if v >= oracle - 0.0025), None)
        out["curves"][name] = {"best_found": curve, "queries_to_oracle_0.25pt": q_hit,
                               "best_at_10": curve[9]}
        print(f"{name:18s} best@10={curve[9]:.4f}  queries to within 0.25pt of oracle: {q_hit}")

    rng = np.random.default_rng(0)
    rand = np.empty((N_RANDOM_REPS, K_MAX))
    for r in range(N_RANDOM_REPS):
        order = list(rng.permutation(b_ids))
        rand[r] = best_found_curve(order, accs)
    out["curves"]["random"] = {
        "mean": rand.mean(0).tolist(),
        "ci95": [np.percentile(rand, 2.5, axis=0).tolist(),
                 np.percentile(rand, 97.5, axis=0).tolist()],
        "best_at_10": float(rand.mean(0)[9]),
    }
    print(f"{'random':18s} best@10={rand.mean(0)[9]:.4f} (mean of {N_RANDOM_REPS})")
    json.dump(out, open(os.path.join(RESULTS_DIR, "search_demo.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4), dpi=150)
    ks = np.arange(1, K_MAX + 1)
    ax.axhline(oracle, ls="--", c="gray", lw=1); ax.text(30, oracle + 0.0006, "oracle (best of 200)", fontsize=8, color="gray")
    ax.fill_between(ks, *[np.array(v) for v in out["curves"]["random"]["ci95"]], alpha=0.15, color="k")
    ax.plot(ks, out["curves"]["random"]["mean"], c="k", lw=1.2, label="random (mean, 95% band)")
    for name, color in [("evolved_consensus", "#1f77b4"), ("flops", "#d62728"),
                        ("nwot", "#9467bd"), ("params", "#2ca02c")]:
        ax.plot(ks, out["curves"][name]["best_found"], lw=1.6, color=color, label=name)
    ax.set_xlabel("ground-truth queries"); ax.set_ylabel("best test accuracy found")
    ax.set_xlim(1, K_MAX)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_search_demo.png"))
    print("figure saved")


if __name__ == "__main__":
    main()
