"""THE ONLY SCRIPT THAT TOUCHES SPACE B AND A-TEST (hard rule 3).

Run once, after evolution configs are frozen. Computes, for every standard
proxy and every evolved elite (cold + warm, all budgets/seeds):
  Spearman [95% bootstrap CI], Kendall tau, Precision@10%
on (a) Space B (primary, never fitted) and (b) the 50-arch A-test split.
Plus: within-FLOPs-tercile Spearman (size-bias control), Wilcoxon
signed-rank across the 10 evolution seeds for evolved-vs-#params at each
budget, and the test-retest noise ceiling.

Bootstrap: percentile, 10,000 resamples over architectures, seed 0
(resample indices deterministic given n and seed; logged in the artifact).

Artifacts: results/evaluation.json (+ results/proxy_scores_{B,Atest}.json,
idempotent so standard-proxy computation is not repeated on re-run).
"""

import glob
import itertools
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy.stats as st
import torch

from . import gp_engine as gp
from . import spaces
from .cache_stats import CACHE_DIR, fixed_speech_batch, load_arch
from .evolve import RESULTS_DIR as EVOLVED_DIR
from .evolve import a_pool_and_test, tree_from_json
from .proxies import ALL_PROXIES, compute_proxy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_BOOT = 10_000
BOOT_SEED = 0


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


def gt_map(space: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, f"gt_{space}", "*_s0.json")):
        r = json.load(open(f))
        out[r["arch_id"]] = r
    return out


def load_ctx(cache_name: str, aid: str) -> dict:
    arrs = load_arch(cache_name, aid)
    ctx = {}
    for kind in ["W", "G", "A", "Gn", "An", "AstdT", "AnstdT"]:
        keys = sorted((k for k in arrs if k.startswith(kind + "_")),
                      key=lambda k: int(k.split("_")[1]))
        if keys:
            ctx[kind] = [arrs[k] for k in keys]
    return ctx


def standard_scores(tag: str, arch_ids: list[str], cfgs: dict) -> dict:
    """proxy -> {arch_id: score}; cached to results/proxy_scores_<tag>.json."""
    path = os.path.join(RESULTS_DIR, f"proxy_scores_{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))
    x, y = fixed_speech_batch()
    out = {p: {} for p in ALL_PROXIES}
    for i, aid in enumerate(arch_ids):
        for p in ALL_PROXIES:
            try:
                out[p][aid] = compute_proxy(
                    p, lambda: spaces.build_model(cfgs[aid], init_seed=0), x, y
                )
            except Exception:
                out[p][aid] = float("nan")
        if (i + 1) % 25 == 0:
            print(f"standard proxies [{tag}]: {i + 1}/{len(arch_ids)}", flush=True)
    json.dump(out, open(path, "w"))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_k(scores, accs, frac=0.10) -> float:
    n = len(scores)
    k = max(1, int(round(frac * n)))
    top_pred = set(np.argsort(scores)[::-1][:k])
    top_true = set(np.argsort(accs)[::-1][:k])
    return len(top_pred & top_true) / k


def metric_block(scores, accs, flops=None) -> dict:
    scores = np.asarray(scores, dtype=float)
    accs = np.asarray(accs, dtype=float)
    ok = np.isfinite(scores)
    if ok.sum() < 0.9 * len(scores):
        return {"spearman": None, "note": f"only {int(ok.sum())}/{len(scores)} finite"}
    s, a = scores[ok], accs[ok]
    rho = float(st.spearmanr(s, a).statistic)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        boots[b] = st.spearmanr(s[idx[b]], a[idx[b]]).statistic
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    out = {
        "spearman": rho, "ci95": [float(lo), float(hi)],
        "kendall": float(st.kendalltau(s, a).statistic),
        "p_at_10": precision_at_k(s, a),
        "n": int(ok.sum()),
    }
    if flops is not None:
        f = np.asarray(flops, dtype=float)[ok]
        terc = np.percentile(f, [33.3, 66.7])
        rows = []
        for lo_f, hi_f in [(-np.inf, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)]:
            m = (f >= lo_f) & (f < hi_f)
            rows.append(float(st.spearmanr(s[m], a[m]).statistic) if m.sum() > 5 else None)
        out["tercile_spearman"] = rows
    return out


def main():
    torch.set_num_threads(max(2, os.cpu_count() - 2))
    # ---- targets ----
    gtB, gtA = gt_map("B"), gt_map("A")
    b_index = json.load(open(os.path.join(CACHE_DIR, "B_index.json")))
    b_ids = sorted(b_index["archs"], key=lambda a: b_index["archs"][a]["i"])
    _, atest_ids = a_pool_and_test()
    b_accs = [gtB[a]["test_acc"] for a in b_ids]
    b_flops = [gtB[a]["flops"] for a in b_ids]
    at_accs = [gtA[a]["test_acc"] for a in atest_ids]
    at_flops = [gtA[a]["flops"] for a in atest_ids]

    b_cfgs = {a: gtB[a]["config"] for a in b_ids}
    at_cfgs = {a: gtA[a]["config"] for a in atest_ids}

    # ---- standard proxies (live computation, frozen batch) ----
    stdB = standard_scores("B", b_ids, b_cfgs)
    stdA = standard_scores("Atest", atest_ids, at_cfgs)

    # ---- evolved proxies (cached statistics) ----
    print("loading eval ctxs...", flush=True)
    ctxB = [load_ctx("B", a) for a in b_ids]
    ctxA = [load_ctx("A", a) for a in atest_ids]

    results = {"standard": {}, "evolved": {}, "wilcoxon": {}}
    for p in ALL_PROXIES:
        results["standard"][p] = {
            "B": metric_block([stdB[p][a] for a in b_ids], b_accs, b_flops),
            "Atest": metric_block([stdA[p][a] for a in atest_ids], at_accs, at_flops),
        }
        rb = results["standard"][p]["B"]
        print(f"std {p:10s} B rho={rb.get('spearman')}", flush=True)

    evolved_rhos_B = {}  # (budget, variant) -> [rho per seed]
    for f in sorted(glob.glob(os.path.join(EVOLVED_DIR, "*.json"))):
        r = json.load(open(f))
        tree = tree_from_json(r["tree_json"])
        tag = r["tag"] + ("" if "warm" not in f or r["tag"].endswith("_warm") else "_warm")
        sb = gp.scores_for(tree, ctxB)
        sa = gp.scores_for(tree, ctxA)
        results["evolved"][tag] = {
            "tree": r["tree"], "budget": r["budget"], "seed": r["seed"],
            "warm": r["tag"].endswith("_warm") if "tag" in r else False,
            "terminals_used": r["terminals_used"],
            "B": metric_block(sb, b_accs, b_flops) if sb else {"spearman": None, "note": "invalid on B"},
            "Atest": metric_block(sa, at_accs, at_flops) if sa else {"spearman": None, "note": "invalid on Atest"},
        }
        variant = "warm" if tag.endswith("_warm") else ("vision" if r["budget"] == 0 else "cold")
        rho = results["evolved"][tag]["B"].get("spearman")
        if rho is not None:
            evolved_rhos_B.setdefault((r["budget"], variant), []).append(rho)
        print(f"evolved {tag:20s} B rho={rho}", flush=True)

    # ---- Wilcoxon: evolved vs #params across seeds, per budget/variant ----
    params_rho_B = results["standard"]["params"]["B"]["spearman"]
    for (budget, variant), rhos in sorted(evolved_rhos_B.items()):
        if len(rhos) >= 6:
            diffs = np.array(rhos) - params_rho_B
            w = st.wilcoxon(diffs, alternative="greater")
            results["wilcoxon"][f"N{budget}_{variant}_vs_params"] = {
                "n_seeds": len(rhos), "mean_rho": float(np.mean(rhos)),
                "params_rho": params_rho_B, "p_value": float(w.pvalue),
            }

    results["meta"] = {
        "noise_ceiling_spearman": 0.9092,
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = os.path.join(RESULTS_DIR, "evaluation.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")

    # ---- learning-curve summary ----
    print("\n=== LEARNING CURVE (Space B Spearman, mean over seeds) ===")
    print(f"noise ceiling: 0.909   #params: {params_rho_B:.3f}   "
          f"flops: {results['standard']['flops']['B']['spearman']:.3f}")
    best_std = max(((p, d['B']['spearman']) for p, d in results['standard'].items()
                    if d['B'].get('spearman') is not None), key=lambda t: t[1])
    print(f"best standard proxy: {best_std[0]} rho={best_std[1]:.3f}")
    for (budget, variant), rhos in sorted(evolved_rhos_B.items()):
        print(f"N={budget:>3} {variant:6s}: mean={np.mean(rhos):.3f} +- {np.std(rhos):.3f} "
              f"(n={len(rhos)} seeds)  min={min(rhos):.3f} max={max(rhos):.3f}")


if __name__ == "__main__":
    main()
