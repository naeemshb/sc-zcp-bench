"""9d-search-scale analysis (PROTOCOL.md 9d): compare the committed N=200 cold elites
(population 30 x 15 generations) with the big-search elites (100 x 30), FITTING
SIDE ONLY. Never reads Space B / A-test / any sealed artifact.

Outputs results/search_scale.json + .md and prints the pinned reading (a/b/c).
Usage: .venv/bin/python -m speech_zcp.search_scale
"""

import glob
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
SPEECH_ONLY = {"Gn", "An", "AstdT", "AnstdT"}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        if d["tag"].endswith("_warm"):
            continue
        t = set(d["terminals_used"])
        rows.append({"seed": d["seed"], "tree": d["tree"], "terminals": sorted(t),
                     "weight_only": t <= {"W"}, "uses_speech_terminal": bool(t & SPEECH_ONLY),
                     "fit_rho": d["fit_rho"], "selection_fitness": d["selection_fitness"],
                     "found_at_gen": d["found_at_gen"], "nodes": d["tree_json"].get("n", None),
                     "config": {k: d["config"][k] for k in ("population", "generations", "max_depth", "complexity_lambda")}})
    return rows


def summarize(rows):
    return {"n": len(rows),
            "weight_only_size_proxies": sum(r["weight_only"] for r in rows),
            "using_speech_terminals": sum(r["uses_speech_terminal"] for r in rows),
            "distinct_formulas": len({r["tree"] for r in rows}),
            "modal_formula": Counter(r["tree"] for r in rows).most_common(1)[0],
            "fit_rho_mean": float(np.mean([r["fit_rho"] for r in rows])),
            "fit_rho_sd": float(np.std([r["fit_rho"] for r in rows])),
            "selection_fitness_mean": float(np.mean([r["selection_fitness"] for r in rows])),
            "found_at_gen_median": float(np.median([r["found_at_gen"] for r in rows])),
            "generations": rows[0]["config"]["generations"], "population": rows[0]["config"]["population"]}


def main():
    small = load(os.path.join(RES, "evolved", "A_N200_s*.json"))
    big = load(os.path.join(RES, "evolved_bigsearch", "A_N200_s*_big.json"))
    assert small and big, "missing elites"
    S, B = summarize(small), summarize(big)
    if B["weight_only_size_proxies"] >= 6:
        reading = "(a) size attractor is search-scale-robust: >=6/10 big-search elites are weight-only size proxies"
    elif B["using_speech_terminals"] >= 6:
        reading = "(b) larger search favours speech-terminal (noise-gradient / time-axis) formulas; sealed quality unknown until a pre-registered evaluation"
    else:
        reading = "(c) mixed: neither weight-only nor speech-terminal formulas reach 6/10"
    out = {"protocol": "PROTOCOL.md 9d-search-scale; fitting side only; no sealed set read",
           "committed_30x15": S, "bigsearch_100x30": B, "per_seed_committed": small, "per_seed_big": big,
           "pinned_reading": reading}
    json.dump(out, open(os.path.join(RES, "search_scale.json"), "w"), indent=1)
    lines = ["# 9d-search-scale: N=200 cold evolution, committed (30x15) vs big search (100x30) — fitting side only", "",
             "| | committed 30x15 | big search 100x30 |", "|---|---|---|"]
    for k, lab in [("weight_only_size_proxies", "weight-only size proxies (of 10)"), ("using_speech_terminals", "elites using speech terminals (of 10)"),
                   ("distinct_formulas", "distinct formulas"), ("found_at_gen_median", "median generation found")]:
        lines.append(f"| {lab} | {S[k]} | {B[k]} |")
    lines.append(f"| modal formula | `{S['modal_formula'][0]}` ×{S['modal_formula'][1]} | `{B['modal_formula'][0]}` ×{B['modal_formula'][1]} |")
    lines.append(f"| fit-ρ (first 80% of chain), mean ± sd | {S['fit_rho_mean']:.3f} ± {S['fit_rho_sd']:.3f} | {B['fit_rho_mean']:.3f} ± {B['fit_rho_sd']:.3f} |")
    lines.append(f"| selection fitness (held-back 20%), mean | {S['selection_fitness_mean']:.3f} | {B['selection_fitness_mean']:.3f} |")
    lines += ["", "## Big-search elites per seed", "", "| seed | formula | terminals | fit-ρ | sel | gen |", "|---|---|---|---|---|---|"]
    for r in sorted(big, key=lambda r: r["seed"]):
        lines.append(f"| {r['seed']} | `{r['tree']}` | {','.join(r['terminals'])} | {r['fit_rho']:.3f} | {r['selection_fitness']:.3f} | {r['found_at_gen']} |")
    lines += ["", f"**Pinned reading:** {reading}", "", "Fit-side ρ is not comparable with Space-B numbers; no sealed set was scored."]
    open(os.path.join(RES, "search_scale.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
