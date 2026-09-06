"""Figure 2 candidate: signal beyond compute (9c-partial).

Partial Spearman of each ranker with test accuracy, controlling for log10 FLOPs,
on Space B (original, n=200) and Space A (pooled replication, n=250 — A-test alone
is n=50 and too wide to show the sign structure). Fixed rankers: 10k-bootstrap
95% CIs; evolved N=200: mean ± seed spread. Reads results/partial_flops.json only.
Usage: .venv/bin/python -m speech_zcp.make_fig2 [--png out.png]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
ROOT = os.path.dirname(HERE)
SETS = [("B", "Space B (TC-ResNet), n=200", "#e6550d", "o"), ("A_rep_pooled", "Space A (DS-CNN), n=250", "#1b9e77", "s")]
LABEL = {"nwot": "nwot", "grasp": "grasp", "plain": "plain", "l2_norm": "l2_norm", "params": "#params", "snip": "snip",
         "fisher": "fisher", "grad_norm": "grad_norm", "synflow": "synflow", "zen": "zen"}


def main(png=None):
    d = json.load(open(os.path.join(RES, "partial_flops.json")))
    rankers = [p for p in LABEL if d["standard"][p]["B"].get("partial") is not None]
    rankers.sort(key=lambda p: d["standard"][p]["B"]["partial"])  # by Space-B value
    rows = [("evolved N=200", None)] + [(LABEL[p], p) for p in rankers]
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    y = np.arange(len(rows))
    ax.axvline(0, color="0.5", lw=1)
    for k, (name, label, color, marker) in enumerate(SETS):
        off = 0.18 if k == 0 else -0.18
        for i, (lab, p) in enumerate(rows):
            if p is None:
                e = d["evolved_by_budget"]["N200"][name]
                ax.errorbar(e["mean"], y[i] + off, xerr=e["std"], fmt=marker, color=color, ms=5, capsize=2, lw=1.2,
                            markerfacecolor="white", markeredgewidth=1.4)
            else:
                b = d["standard"][p][name]
                lo, hi = b["ci95"]
                ax.errorbar(b["partial"], y[i] + off, xerr=[[b["partial"] - lo], [hi - b["partial"]]], fmt=marker,
                            color=color, ms=4.5, capsize=2, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("partial Spearman with accuracy  |  log FLOPs")
    ax.set_xlim(-0.55, 1.0); ax.set_ylim(-0.7, len(rows) + 1.9)  # headroom for the legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=5, label=l) for _, l, c, m in SETS]
    handles.append(Line2D([], [], color="0.3", marker="o", ls="", ms=5, markerfacecolor="white", markeredgewidth=1.4,
                          label="hollow: evolved N=200 (± seed spread)"))
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    for out in (os.path.join(ROOT, "fig2_partial_flops.pdf"), os.path.join(HERE, "figures", "fig2_partial_flops.pdf")):
        fig.savefig(out, bbox_inches="tight"); print("wrote", out)
    if png:
        fig.savefig(png, dpi=180, bbox_inches="tight"); print("wrote", png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--png", default=None)
    main(ap.parse_args().png)
