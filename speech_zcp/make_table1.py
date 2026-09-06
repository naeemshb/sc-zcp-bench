"""Table 1 (main.tex tab:main) generated from committed artifacts.

Columns: Spearman on sealed Space B (n=200) [10k-bootstrap CI]; the pooled
pre-registered replication (n=300); partial Spearman controlling for log FLOPs
(9c-partial, Space B). Evolved rows: mean ± seed spread. Writes table1.tex (an
\\input-able tabular) so no number in the table is typed by hand.
Usage: .venv/bin/python -m speech_zcp.make_table1
"""

import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
ROOT = os.path.dirname(HERE)


def ci(b, d=3):
    return f"{b['spearman']:.{d}f} & \\ci{{{b['ci95'][0]:.{d}f}}}{{{b['ci95'][1]:.{d}f}}}"


def main():
    ev = json.load(open(os.path.join(RES, "evaluation.json")))
    r2 = json.load(open(os.path.join(RES, "replication2.json")))
    nc = json.load(open(os.path.join(RES, "noise_ceiling.json")))
    pf = json.load(open(os.path.join(RES, "partial_flops.json")))

    def evo(N, key, d):
        v = [x[key]["spearman"] for x in d["evolved"].values() if x["budget"] == N and not x["warm"]]
        return float(np.mean(v)), float(np.std(v))

    def part(p):
        b = pf["standard"][p]["B"]
        return "---" if b.get("partial") is None or not np.isfinite(b["partial"]) else f"${b['partial']:+.2f}$"

    L = [r"\begin{tabular}{l cc c c}", r"\toprule",
         r"Ranker & \multicolumn{2}{c}{Space B, $n{=}200$} & Repl.\ $n{=}300$ & Beyond FLOPs \\",
         r" & $\rho$ & 95\% CI & $\rho$ & partial $\rho$ \\", r"\midrule",
         f"Noise ceiling (test--retest) & {ci(nc['B'])} & {nc['B_rep_pooled']['spearman']:.3f} & --- \\\\"]
    for p, lab in [("flops", "FLOPs (trivial)"), ("nwot", r"\texttt{nwot} (best fixed)"), ("params", r"\#params (trivial)")]:
        L.append(f"{lab} & {ci(ev['standard'][p]['B'])} & {r2['standard'][p]['B_rep_pooled']['spearman']:.3f} & {part(p)} \\\\")
    for N, lab in [(200, r"Evolved, $N{=}200$"), (50, r"Evolved, $N{=}50$"), (25, r"Evolved, $N{=}25$"), (0, r"Evolved, $N{=}0$ (vision-only)")]:
        m, s = evo(N, "B", ev); m2, _ = evo(N, "B_rep_pooled", r2)
        pb = pf["evolved_by_budget"][f"N{N}"]["B"]
        L.append(f"{lab} & {m:.3f} & $\\pm{s:.3f}$ & {m2:.3f} & ${pb['mean']:+.2f}$ \\\\")
    weak = [("zen", "zen"), ("l2_norm", "l2"), ("snip", "snip"), ("synflow", "synflow"), ("grad_norm", "grad\\_n."), ("fisher", "fisher"), ("grasp", "grasp"), ("plain", "plain")]
    for grp in (weak[:4], weak[4:]):
        names = " / ".join(l for _, l in grp)
        rho = " / ".join(f"${ev['standard'][p]['B']['spearman']:.2f}$" for p, _ in grp)
        rep = " / ".join(f"${r2['standard'][p]['B_rep_pooled']['spearman']:.2f}$" for p, _ in grp)
        pa = " / ".join(f"${pf['standard'][p]['B']['partial']:+.2f}$" for p, _ in grp)
        L.append(f"{names} & \\multicolumn{{2}}{{c}}{{{rho}}} & {rep} & {pa} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    out = os.path.join(ROOT, "table1.tex")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out); print("\n".join(L))


if __name__ == "__main__":
    main()
