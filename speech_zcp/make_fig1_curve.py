"""Fig. 1(a) of the paper: the specialization learning curve on sealed Space B.
Curve = mean +- 1 s.d. over 10 seeds at N = 0..300; fixed rankers and the noise ceiling as labeled lines with
their 10k-bootstrap 95% CIs as right-margin whiskers; every fixed proxy shown. Reads speech_zcp/results/
{evaluation,n300,noise_ceiling}.json; writes fig1_curve.pdf at the repository root."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "speech_zcp", "results")
BUDGETS = [0, 25, 50, 100, 200, 300]
CURVE, MID, MIDTXT, WEAK = "#1f6fb4", "0.72", "0.5", "#a6761d"

ev = json.load(open(os.path.join(RES, "evaluation.json")))
n300 = json.load(open(os.path.join(RES, "n300.json")))["evolved"]
nc = json.load(open(os.path.join(RES, "noise_ceiling.json")))["B"]
ms, sds = [], []
for N in BUDGETS:
    src = n300 if N == 300 else ev["evolved"]
    r = [v["B"]["spearman"] for v in src.values() if v["budget"] == N and not v.get("warm") and v["B"].get("spearman") is not None]
    assert len(r) == 10, (N, len(r))
    ms.append(float(np.mean(r))); sds.append(float(np.std(r)))  # ddof=0 == Table 1's +-
std = {k: ev["standard"][k]["B"]["spearman"] for k in ev["standard"]}
ci = {k: ev["standard"][k]["B"]["ci95"] for k in ev["standard"]}

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.7})
fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.3))
fig.subplots_adjust(right=0.74)
xpos = np.arange(len(BUDGETS)); xr = len(BUDGETS) - 0.28
WX = {"flops": len(BUDGETS) - 0.56, "params": len(BUDGETS) - 0.56, "ceiling": len(BUDGETS) - 0.56, "nwot": len(BUDGETS) - 0.44}

def whisker(x, lo, hi, color):
    ax.plot([x, x], [lo, hi], color=color, lw=1.1, solid_capstyle="butt", clip_on=False, zorder=2)
    for y in (lo, hi):
        ax.plot([x - 0.035, x + 0.035], [y, y], color=color, lw=0.8, clip_on=False, zorder=2)

# strong references + CI whiskers
for key, label, color, va, dy in [("flops", "FLOPs", "#d95f02", "bottom", 0.006), ("nwot", "nwot", "#7570b3", "top", -0.006),
                                  ("params", "#params", "#1b9e77", "center", 0.0)]:
    ax.axhline(std[key], color=color, lw=1.0, alpha=0.95, zorder=1)
    ax.text(xr, std[key] + dy, f"{label} {std[key]:.3f}", color=color, fontsize=7.5, va=va, ha="left", clip_on=False)
    whisker(WX[key], ci[key][0], ci[key][1], color)
ax.axhline(nc["spearman"], color="0.3", lw=1.0, ls=(0, (4, 2)), zorder=1)
ax.text(xr, nc["spearman"], f"noise ceiling {nc['spearman']:.3f}", color="0.3", fontsize=7.5, va="center", ha="left", clip_on=False)
whisker(WX["ceiling"], nc["ci95"][0], nc["ci95"][1], "0.3")
# mid-strength group: grey dotted
for key in ("zen", "l2_norm", "snip", "synflow"):
    ax.axhline(std[key], color=MID, lw=0.7, ls=(0, (2, 2)), zorder=0)
ax.text(xr, (std["zen"] + std["l2_norm"]) / 2 + 0.008, f"zen {std['zen']:.2f} / l2 {std['l2_norm']:.2f}", color=MIDTXT, fontsize=6.8, va="center", ha="left", clip_on=False)
ax.text(xr, std["snip"], f"snip {std['snip']:.2f}", color=MIDTXT, fontsize=6.8, va="center", ha="left", clip_on=False)
ax.text(xr, std["synflow"] - 0.002, f"synflow {std['synflow']:.2f}", color=MIDTXT, fontsize=6.8, va="center", ha="left", clip_on=False)
# weakest group: bronze dotted
for key, label, va, dy in [("grad_norm", "grad_norm", "bottom", 0.004), ("fisher", "fisher", "bottom", 0.004),
                           ("grasp", "grasp", "center", 0.0), ("plain", "plain", "top", -0.004)]:
    ax.axhline(std[key], color=WEAK, lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.text(xr, std[key] + dy, f"{label} {std[key]:.2f}".replace("-", "−"), color=WEAK, fontsize=6.8, va=va, ha="left", clip_on=False)
ax.axhline(0, color="0.8", lw=0.6, zorder=0)
# the curve: mean +- 1 seed s.d.
ax.errorbar(xpos, ms, yerr=sds, color=CURVE, marker="o", ms=4.5, mec="white", mew=0.6, lw=1.7, capsize=2.5, capthick=0.9, elinewidth=0.9, zorder=5)
ax.set_xticks(xpos); ax.set_xticklabels(["0\n(vision)"] + [str(b) for b in BUDGETS[1:]])
ax.set_xlim(-0.35, len(BUDGETS) - 0.65)
ax.set_ylim(-0.2, 1.0); ax.set_yticks(np.round(np.arange(-0.2, 1.01, 0.1), 1))
ax.set_xlabel("in-domain budget $N$ (trained speech models)")
ax.set_ylabel("Spearman $\\rho$, sealed Space B")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
out = os.path.join(REPO, "fig1_curve.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
print("means", [round(m, 3) for m in ms], "sd", [round(s, 3) for s in sds])
