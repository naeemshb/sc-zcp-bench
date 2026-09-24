"""EXPLORATORY COPY of speech_zcp/evolve.py (2026-09-12). NOT part of the pre-registered pipeline.

Re-runs the N=100 evolution with a different search configuration:
    population 50, generations 20, max_depth 3, seeds 1-25
Everything else is byte-identical to the committed runs: the same nested chain per seed
(nested_chain), the same 80/20 fit/selection split, tournament k=3, p_crossover 0.5,
elitism 2, complexity penalty 0.002/node, the same top-5 selection rule, the same terminals.

Separation guarantees
  * imports the COPIED engine (gp_engine_copy.py); never imports speech_zcp.evolve or
    speech_zcp.gp_engine
  * reads only cache/A, cache/A_index.json, splits/a_pool_test.json, results/gt_A (read-only)
  * writes only under exploratory/gp_copy/results/<tag>/ ; no script in speech_zcp reads that path
  * never loads Space B, A-test, or any replication set (PROTOCOL.md hard rule 3)
Extra per-seed field: hold_rho = Spearman of the elite on the 100 pool architectures outside
the seed's chain (never fitted by that run): a within-Space-A pseudo-holdout, NOT a sealed metric.

Usage (from the repo root):
  .venv/bin/python exploratory/gp_copy/evolve_explore.py --seeds 1-25
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(1, REPO)
import gp_engine_copy as gp  # noqa: E402  (the copied engine)
from speech_zcp.cache_stats import CACHE_DIR, load_arch  # noqa: E402  (read-only cache access)

CONFIG = {
    "population": 50,            # committed: 30
    "generations": 20,           # committed: 15
    "tournament_k": 3,
    "p_crossover": 0.5,
    "elitism": 2,
    "max_depth": 3,              # committed: 6  (depth 3 = Agg(Reduce(terminal)) only)
    "complexity_lambda": 0.002,  # penalty per node, Spearman scale
    "fit_fraction": 0.8,         # used only for N >= 100
    "small_n_threshold": 50,     # N <= this: no split (frozen protocol)
    "n_strata": 5,               # FLOPs strata for nested chains
    "version": "evolve-explore-v1 (copy of evolve-v1; pop50 gen20 depth3)",
}
TAG = "n100_pop50_gen20_depth3"
RESULTS_DIR = os.path.join(HERE, "results", TAG)
SPLITS_DIR = os.path.join(REPO, "speech_zcp", "splits")
GT_A_DIR = os.path.join(REPO, "speech_zcp", "results", "gt_A")


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Splits: read the committed artifact ONLY (never regenerate)
# ---------------------------------------------------------------------------
def a_pool_and_test() -> tuple[list[str], list[str]]:
    d = json.load(open(os.path.join(SPLITS_DIR, "a_pool_test.json")))
    return d["pool"], d["test"]


def nested_chain(seed: int, index: dict, pool: list[str]) -> list[str]:
    """VERBATIM copy of speech_zcp.evolve.nested_chain (interleaved groups, seeded shuffle,
    round-robin) so every seed's chain is identical to the committed one."""
    flops = {a: index["archs"][a].get("flops") for a in pool}
    if any(v is None for v in flops.values()):  # fall back to gt files
        for f in glob.glob(os.path.join(GT_A_DIR, "*_s0.json")):
            r = json.load(open(f))
            if r["arch_id"] in flops:
                flops[r["arch_id"]] = r["flops"]
    ordered = sorted(pool, key=lambda a: flops[a])
    k = CONFIG["n_strata"]
    strata = [ordered[i::k] for i in range(k)]  # interleaved => equal FLOPs coverage
    rng = random.Random(f"chain-{seed}")
    for s in strata:
        rng.shuffle(s)
    chain = []
    for j in range(max(len(s) for s in strata)):
        for s in strata:
            if j < len(s):
                chain.append(s[j])
    return chain


# ---------------------------------------------------------------------------
# Fitness and evolution: VERBATIM copies of speech_zcp.evolve
# ---------------------------------------------------------------------------
def penalized_rho(tree, ctxs, accs) -> tuple[float, float]:
    """-> (selection_fitness, raw_rho); worst fitness on invalid formulas."""
    scores = gp.scores_for(tree, ctxs)
    if scores is None:
        return -2.0, float("nan")
    rho = st.spearmanr(scores, accs).statistic
    if not np.isfinite(rho):
        return -2.0, float("nan")
    return rho - CONFIG["complexity_lambda"] * tree.count(), float(rho)


def evolve(seed: int, ctxs_fit, accs_fit, ctxs_sel, accs_sel, terminals, warm_trees=None):
    """ctxs_sel/accs_sel may be the same objects as fit (small-N protocol).
    warm_trees: elite trees to inject into the initial population (ablation)."""
    rng = random.Random(f"evolve-{seed}")
    pop = [gp.random_tree(rng, terminals, CONFIG["max_depth"])
           for _ in range(CONFIG["population"])]
    for i, t in enumerate((warm_trees or [])[: CONFIG["population"] // 2]):
        if gp.validate(t, terminals):
            pop[i] = t.clone()
    best = None  # (sel_fitness, tree, gen)
    history = []
    for gen in range(CONFIG["generations"]):
        fits = [penalized_rho(t, ctxs_fit, accs_fit)[0] for t in pop]
        order = np.argsort(fits)[::-1]
        # selection criterion evaluated on the selection set
        for i in order[: CONFIG["elitism"] + 3]:
            sel_fit, _ = penalized_rho(pop[i], ctxs_sel, accs_sel)
            if best is None or sel_fit > best[0]:
                best = (sel_fit, pop[i].clone(), gen)
        history.append({"gen": gen, "best_fit": float(max(fits)),
                        "mean_fit": float(np.mean([f for f in fits if f > -2]) if any(f > -2 for f in fits) else -2)})
        elites = [pop[i].clone() for i in order[: CONFIG["elitism"]]]

        def tournament():
            idx = rng.sample(range(len(pop)), CONFIG["tournament_k"])
            return pop[max(idx, key=lambda i: fits[i])]

        children = []
        while len(children) < CONFIG["population"] - CONFIG["elitism"]:
            if rng.random() < CONFIG["p_crossover"]:
                c = gp.crossover(rng, tournament(), tournament(), CONFIG["max_depth"])
            else:
                c = gp.mutate(rng, tournament(), terminals, CONFIG["max_depth"])
            if gp.validate(c, terminals):
                children.append(c)
        pop = elites + children
    return best, history


def tree_to_json(t) -> dict:
    return {"op": t.op, "children": [tree_to_json(c) for c in t.children]}


def tree_from_json(d) -> gp.Node:
    return gp.Node(d["op"], [tree_from_json(c) for c in d["children"]])


def ctx_of(aid: str) -> dict:
    arrs = load_arch("A", aid)
    ctx = {}
    for kind in ["W", "G", "A", "Gn", "An", "AstdT", "AnstdT"]:
        keys = sorted((k for k in arrs if k.startswith(kind + "_")),
                      key=lambda k: int(k.split("_")[1]))
        if keys:
            ctx[kind] = [arrs[k] for k in keys]
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seeds", default="1-25", help="inclusive range, e.g. 1-25")
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split("-"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    index = json.load(open(os.path.join(CACHE_DIR, "A_index.json")))
    pool, atest = a_pool_and_test()
    assert len(pool) == 200 and not set(pool) & set(atest)
    t0 = time.time()
    print(f"[{TAG}] loading the 200 pool ctxs once...", flush=True)
    CTX = {aid: ctx_of(aid) for aid in pool}
    ACC = {aid: index["archs"][aid]["acc"] for aid in pool}
    print(f"[{TAG}] loaded in {time.time() - t0:.0f}s; config={CONFIG}", flush=True)
    for seed in range(lo, hi + 1):
        out_path = os.path.join(RESULTS_DIR, f"A_N{args.budget}_s{seed}_{TAG}.json")
        if os.path.exists(out_path):
            print(f"[s{seed}] exists, skipping (resume-safe)", flush=True)
            continue
        chain_full = nested_chain(seed, index, pool)
        chain = chain_full[: args.budget]
        small = 0 < args.budget <= CONFIG["small_n_threshold"]
        if small:
            fit_ids, sel_ids = chain, chain
        else:
            n_fit = int(CONFIG["fit_fraction"] * len(chain))
            fit_ids, sel_ids = chain[:n_fit], chain[n_fit:]
        hold_ids = chain_full[args.budget:]
        mk = lambda ids: ([CTX[a] for a in ids], [ACC[a] for a in ids])
        cf, af = mk(fit_ids)
        cs, as_ = mk(sel_ids)
        ch_, ah_ = mk(hold_ids)
        t1 = time.time()
        best, history = evolve(seed, cf, af, cs, as_, gp.SPEECH_TERMINALS)
        sel_fit, tree, gen = best
        _, fit_rho = penalized_rho(tree, cf, af)
        _, sel_rho = penalized_rho(tree, cs, as_)
        hold_rho = penalized_rho(tree, ch_, ah_)[1] if hold_ids else float("nan")
        out = {
            "tag": f"A_N{args.budget}_s{seed}_{TAG}", "budget": args.budget, "seed": seed,
            "exploratory": True,
            "settings_changed_vs_committed": {"population": "30->50", "generations": "15->20",
                                              "max_depth": "6->3", "seeds": "10->25"},
            "tree": str(tree), "tree_json": tree_to_json(tree),
            "terminals_used": sorted(gp.terminal_usage(tree)),
            "selection_fitness": float(sel_fit), "sel_rho": float(sel_rho),
            "fit_rho": float(fit_rho), "hold_rho": float(hold_rho),
            "n_fit": len(fit_ids), "n_sel": len(sel_ids), "n_hold": len(hold_ids),
            "found_at_gen": gen, "history": history,
            "protocol": "no-split" if small else "80/20",
            "config": CONFIG, "chain": chain, "hold_ids": hold_ids,
            "git_hash": git_hash(), "wall_s": round(time.time() - t1, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"[s{seed:>2}] {str(tree):40s} gen={gen:>2} fit={fit_rho:.3f} sel={sel_rho:.3f} "
              f"hold={hold_rho:.3f} terms={sorted(gp.terminal_usage(tree))} ({out['wall_s']}s)", flush=True)
    print(f"[{TAG}] DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
