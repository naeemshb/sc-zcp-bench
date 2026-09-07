"""Figure 1 of main.tex (fig:curve), regenerated from committed artifacts only.

(a) Specialization learning curve on sealed Space B: mean Spearman of the cold
    evolved elites per budget N (t-based 95% interval over the 10 evolution
    seeds), fixed-ranker reference lines with their 10k-bootstrap 95% CI bands,
    and the test-retest noise ceiling (9b-seeds, n=200) as a dashed line.
(b) Space B accuracy vs. MACs, colored by FLOPs tercile (Table terciles).

Inputs : results/evaluation.json, results/noise_ceiling.json, results/gt_B/*_s0.json
Outputs: <repo>/fig1_curve_scatter.pdf and speech_zcp/figures/fig1_curve_scatter.pdf
Usage  : .venv/bin/python -m speech_zcp.make_fig1 [--png out.png]
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats as st  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
ROOT = os.path.dirname(HERE)
BUDGETS = [0, 25, 50, 100, 200]
REFS = [("flops", "FLOPs", "#e6550d"), ("nwot", "nwot", "#d6609a"), ("params", "#params", "#1b9e77")]
CURVE = "#1f6fb4"
TERC_COLORS = ["#fdc692", "#e6550d", "#a63603"]


def seed_curve(ev):
    xs, ms, hs = [], [], []
    for N in BUDGETS:
        r = [v["B"]["spearman"] for v in ev["evolved"].values() if v["budget"] == N and not v["warm"]]
        r = [x for x in r if x is not None]
        m = float(np.mean(r))
        h = float(st.t.ppf(0.975, len(r) - 1) * np.std(r, ddof=1) / np.sqrt(len(r)))
        xs.append(N); ms.append(m); hs.append(h)
    return ms, hs


def main(png=None, single=False):
    """single=True: panel (a) only -> fig1_curve.pdf (page-budget variant; same data, same code)."""
    ev = json.load(open(os.path.join(RES, "evaluation.json")))
    nc = json.load(open(os.path.join(RES, "noise_ceiling.json")))["B"]
    gt = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "gt_B", "*_s0.json"))]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42})
    if single:
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.7)); bx = None
    else:
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.6, 5.4), gridspec_kw={"hspace": 0.55})

    # ---- (a) learning curve -------------------------------------------------
    ms, hs = seed_curve(ev)
    xpos = np.arange(len(BUDGETS))
    for key, label, color in REFS:
        blk = ev["standard"][key]["B"]
        ax.axhspan(blk["ci95"][0], blk["ci95"][1], color=color, alpha=0.12, lw=0)
        ax.axhline(blk["spearman"], color=color, lw=1.6)
    # label placement: right edge, stacked so labels do not collide
    ax.text(4.32, ev["standard"]["flops"]["B"]["spearman"] + 0.004, f"FLOPs {ev['standard']['flops']['B']['spearman']:.3f}",
            color=REFS[0][2], ha="right", va="bottom", fontsize=8.5)
    ax.text(4.32, ev["standard"]["nwot"]["B"]["spearman"] - 0.004, f"nwot {ev['standard']['nwot']['B']['spearman']:.3f}",
            color=REFS[1][2], ha="right", va="top", fontsize=8.5)
    ax.text(-0.2, ev["standard"]["params"]["B"]["spearman"] - 0.006, f"#params {ev['standard']['params']['B']['spearman']:.3f}",
            color=REFS[2][2], ha="left", va="top", fontsize=8.5)
    ax.axhspan(nc["ci95"][0], nc["ci95"][1], color="0.45", alpha=0.10, lw=0)
    ax.axhline(nc["spearman"], color="0.35", lw=1.4, ls="--")
    ax.text(-0.2, nc["spearman"] + 0.008, f"noise ceiling {nc['spearman']:.3f} [{nc['ci95'][0]:.3f}, {nc['ci95'][1]:.3f}]",
            color="0.35", ha="left", va="bottom", fontsize=8.5)
    ax.errorbar(xpos, ms, yerr=hs, color=CURVE, marker="o", ms=5, lw=2, capsize=3, zorder=5)
    ax.annotate("vision-only", (0, ms[0]), xytext=(0.12, ms[0] - 0.045), color=CURVE, fontsize=8.5)
    ax.annotate("50 models recover\nmost of the gap", (2, ms[2]), xytext=(2.35, ms[2] - 0.22), color=CURVE,
                fontsize=8.5, arrowprops={"arrowstyle": "-|>", "color": CURVE, "lw": 1})
    ax.set_xticks(xpos); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlim(-0.35, 4.35); ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("speech ground-truth budget $N$"); ax.set_ylabel("Spearman $\\rho$ on Space B")
    ax.set_title("evolved proxies vs. in-domain budget" if single else "(a) evolved proxies vs. in-domain budget")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (b) compute orders the space ---------------------------------------
    if single:
        outs = (os.path.join(ROOT, "fig1_curve.pdf"), os.path.join(HERE, "figures", "fig1_curve.pdf"))
        for out in outs:
            fig.savefig(out, bbox_inches="tight"); print("wrote", out)
        if png:
            fig.savefig(png, dpi=180, bbox_inches="tight"); print("wrote", png)
        return
    flops = np.array([g["flops"] for g in gt]); acc = np.array([g["test_acc"] for g in gt]) * 100
    terc = np.percentile(flops, [33.3, 66.7])
    tier = np.digitize(flops, terc)
    for t in range(3):
        m = tier == t
        bx.scatter(flops[m], acc[m], s=14, color=TERC_COLORS[t], alpha=0.9, lw=0)
    for x, lab in zip(terc, ["T1 | T2", "T2 | T3"]):
        bx.axvline(x, color="0.6", ls=":", lw=1)
        bx.text(x, bx.get_ylim()[1] if False else 96.5, lab, color="0.5", ha="center", va="bottom", fontsize=8)
    bx.set_xscale("log")
    f = ev["standard"]["flops"]["B"]
    bx.text(0.03, 0.04, f"$\\rho$(FLOPs, acc) = {f['spearman']:.3f} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}]",
            transform=bx.transAxes, color=REFS[0][2], fontsize=8.5)
    bx.set_ylim(min(acc) - 2, 98)
    bx.set_xlabel("MACs (log scale)"); bx.set_ylabel("test accuracy (%)")
    bx.set_title("(b) Space B: compute nearly orders the space")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    for out in (os.path.join(ROOT, "fig1_curve_scatter.pdf"), os.path.join(HERE, "figures", "fig1_curve_scatter.pdf")):
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)
    if png:
        fig.savefig(png, dpi=180, bbox_inches="tight"); print("wrote", png)
    print(f"ceiling {nc['spearman']:.4f} {nc['ci95']}  curve means {[round(m, 3) for m in ms]}  half-widths {[round(h, 3) for h in hs]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    ap.add_argument("--single", action="store_true", help="panel (a) only -> fig1_curve.pdf")
    a = ap.parse_args()
    main(a.png, a.single)
