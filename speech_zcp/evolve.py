"""Budget-curve evolution entry point (PROTOCOL.md section 6).

For (seed s, budget N): evolve proxy formulas over the cached statistics of a
nested, FLOPs-stratified chain drawn from the Space-A specialization pool.
N=0 evolves on the NB201 cache against NB201 accuracies (no speech data).

Model-selection protocol (frozen 2026-08-05, recorded in PROTOCOL.md):
  N >= 100: fitness on first 80% of chain, elite selection on the last 20%.
  N <= 50:  fitness AND selection on the full chain; complexity penalty is
            the sole regularizer.
Holdout sanctity: this script NEVER touches Space B or the A-test split.
evaluate.py is the only script that does.

Usage:
  .venv/bin/python -m speech_zcp.evolve --budget 25 --seed 1
  .venv/bin/python -m speech_zcp.evolve --budget 0 --seed 1          # vision
  .venv/bin/python -m speech_zcp.evolve --budget 25 --seed 1 \
      --warm-start results/evolved/nb201_N0_s1.json                  # ablation
  # 9d-search-scale (2026-09-10): larger search, fitting side only, separate dir
  .venv/bin/python -m speech_zcp.evolve --budget 200 --seed 1 \
      --population 100 --generations 30 --out-dir speech_zcp/results/evolved_bigsearch --tag-suffix _big
"""

import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy.stats as st

from . import gp_engine as gp
from .cache_stats import CACHE_DIR, load_arch
from . import spaces

CONFIG = {
    "population": 30,
    "generations": 15,
    "tournament_k": 3,
    "p_crossover": 0.5,
    "elitism": 2,
    "max_depth": 6,
    "complexity_lambda": 0.002,  # penalty per node, Spearman scale
    "fit_fraction": 0.8,         # used only for N >= 100
    "small_n_threshold": 50,     # N <= this: no split (frozen protocol)
    "n_strata": 5,               # FLOPs strata for nested chains
    "version": "evolve-v1",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "evolved")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Splits (deterministic artifacts under splits/)
# ---------------------------------------------------------------------------
def a_pool_and_test() -> tuple[list[str], list[str]]:
    """250 Space-A archs -> 200 specialization pool + 50 fixed A-test."""
    path = os.path.join(SPLITS_DIR, "a_pool_test.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return d["pool"], d["test"]
    ids = [spaces.arch_id(c) for c in spaces.sample_archs("A", 250)]
    test_idx = set(random.Random("Atest-2027").sample(range(250), 50))
    pool = [a for i, a in enumerate(ids) if i not in test_idx]
    test = [a for i, a in enumerate(ids) if i in test_idx]
    os.makedirs(SPLITS_DIR, exist_ok=True)
    json.dump({"pool": pool, "test": test, "rule": "random.Random('Atest-2027').sample(range(250), 50)"},
              open(path, "w"))
    return pool, test


def nested_chain(seed: int, index: dict, pool: list[str]) -> list[str]:
    """FLOPs-stratified nested chain over the pool: 25 c 50 c 100 c 200.

    Pool is sorted by FLOPs into n_strata equal strata; each stratum is
    shuffled by `seed`; taking the first k of every stratum (round-robin)
    yields nested, stratified prefixes for every budget."""
    flops = {a: index["archs"][a].get("flops") for a in pool}
    if any(v is None for v in flops.values()):  # fall back to gt files
        import glob

        for f in glob.glob(os.path.join(os.path.dirname(RESULTS_DIR), "gt_A", "*_s0.json")):
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
# Fitness
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


# ---------------------------------------------------------------------------
# Node (de)serialization
# ---------------------------------------------------------------------------
def tree_to_json(t) -> dict:
    return {"op": t.op, "children": [tree_to_json(c) for c in t.children]}


def tree_from_json(d) -> gp.Node:
    return gp.Node(d["op"], [tree_from_json(c) for c in d["children"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True, choices=[0, 25, 50, 100, 200])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--warm-start", default=None,
                    help="path to an N=0 elite json to seed the population (ablation)")
    ap.add_argument("--population", type=int, default=None, help="9d: override CONFIG population")
    ap.add_argument("--generations", type=int, default=None, help="9d: override CONFIG generations")
    ap.add_argument("--out-dir", default=None, help="9d: write elites here instead of results/evolved")
    ap.add_argument("--tag-suffix", default="", help="9d: appended to the artifact tag")
    args = ap.parse_args()
    if args.population or args.generations:  # non-default search scale: mark the config
        CONFIG["population"] = args.population or CONFIG["population"]
        CONFIG["generations"] = args.generations or CONFIG["generations"]
        CONFIG["version"] = "evolve-v1-scale"
    out_dir = args.out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.budget == 0:
        index = json.load(open(os.path.join(CACHE_DIR, "nb201_index.json")))
        ids = sorted(index["archs"], key=lambda a: index["archs"][a]["i"])
        terminals = gp.VISION_TERMINALS
        cache_name, tag = "nb201", f"nb201_N0_s{args.seed}"
        # 300-arch cap: full 500 uncompressed (~10+ GB) risks the 16 GB RAM
        # budget; 300 is ~4x the EZNAS per-space evolution sample (80)
        chain = ids[:300]
    else:
        index = json.load(open(os.path.join(CACHE_DIR, "A_index.json")))
        pool, _ = a_pool_and_test()
        terminals = gp.SPEECH_TERMINALS
        cache_name, tag = "A", f"A_N{args.budget}_s{args.seed}"
        chain = nested_chain(args.seed, index, pool)[: args.budget]

    print(f"[{tag}] loading {len(chain)} cached archs...", flush=True)
    ctxs = []
    for aid in chain:
        arrs = load_arch(cache_name, aid)
        ctx = {}
        for kind in ["W", "G", "A", "Gn", "An", "AstdT", "AnstdT"]:
            keys = sorted((k for k in arrs if k.startswith(kind + "_")),
                          key=lambda k: int(k.split("_")[1]))
            if keys:
                ctx[kind] = [arrs[k] for k in keys]
        ctxs.append(ctx)
    accs = [index["archs"][a]["acc"] for a in chain]

    small = 0 < args.budget <= CONFIG["small_n_threshold"]
    if args.budget == 0 or not small:
        n_fit = int(CONFIG["fit_fraction"] * len(chain)) if args.budget else len(chain)
        if args.budget == 0:
            ctxs_fit, accs_fit, ctxs_sel, accs_sel = ctxs, accs, ctxs, accs
        else:
            ctxs_fit, accs_fit = ctxs[:n_fit], accs[:n_fit]
            ctxs_sel, accs_sel = ctxs[n_fit:], accs[n_fit:]
    else:
        ctxs_fit, accs_fit, ctxs_sel, accs_sel = ctxs, accs, ctxs, accs

    tag += args.tag_suffix
    warm_trees = None
    if args.warm_start:
        ws = json.load(open(args.warm_start))
        warm_trees = [tree_from_json(ws["tree_json"])]
        tag += "_warm"  # keep cold-start artifacts intact (ablation pairs)

    best, history = evolve(args.seed, ctxs_fit, accs_fit, ctxs_sel, accs_sel,
                           terminals, warm_trees)
    sel_fit, tree, gen = best
    _, fit_rho = penalized_rho(tree, ctxs_fit, accs_fit)
    out = {
        "tag": tag, "budget": args.budget, "seed": args.seed,
        "tree": str(tree), "tree_json": tree_to_json(tree),
        "terminals_used": sorted(gp.terminal_usage(tree)),
        "selection_fitness": float(sel_fit), "fit_rho": float(fit_rho),
        "found_at_gen": gen, "history": history,
        "protocol": "no-split" if small else "80/20",
        "config": CONFIG, "chain": chain,
        "warm_start": args.warm_start,
        "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(out_dir, f"{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{tag}] best: {tree}")
    print(f"[{tag}] sel_fitness={sel_fit:.4f} fit_rho={fit_rho:.4f} "
          f"terminals={sorted(gp.terminal_usage(tree))} -> {path}")


if __name__ == "__main__":
    main()
