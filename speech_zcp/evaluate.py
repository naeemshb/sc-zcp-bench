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

--replication (PROTOCOL.md 9b): the same frozen proxies and committed elites
scored ONCE on the pre-registered replication sets (A_rep, B_rep) ->
results/replication.json + replication.md delta table. evaluation.json is
never rewritten; the sets are never merged. --selfcheck re-runs the
replication code path on the original Space B and asserts it reproduces
evaluation.json (run before the sealed replication).
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


# ---------------------------------------------------------------------------
# Pre-registered replication (PROTOCOL.md 9b)
# ---------------------------------------------------------------------------
REP_SETS = {"B_rep": "B", "A_rep": "Atest"}  # replication set -> original column it replicates


def _load_set(name: str) -> dict:
    gt = gt_map(name)
    index = json.load(open(os.path.join(CACHE_DIR, f"{name}_index.json")))
    ids = sorted(index["archs"], key=lambda a: index["archs"][a]["i"])
    return {"ids": ids, "accs": [gt[a]["test_acc"] for a in ids],
            "flops": [gt[a]["flops"] for a in ids],
            "cfgs": {a: gt[a]["config"] for a in ids}}


def _standard_blocks(name: str, s: dict) -> dict:
    std = standard_scores(name, s["ids"], s["cfgs"])
    return {p: metric_block([std[p][a] for a in s["ids"]], s["accs"], s["flops"])
            for p in ALL_PROXIES}


def selfcheck() -> bool:
    """The replication code path, run on the ORIGINAL Space B, must reproduce
    evaluation.json (frozen batch, cached proxy scores, seeded bootstrap)."""
    orig = json.load(open(os.path.join(RESULTS_DIR, "evaluation.json")))["standard"]
    blocks = _standard_blocks("B", _load_set("B"))
    bad = [p for p in ALL_PROXIES
           if blocks[p].get("spearman") is None
           or abs(blocks[p]["spearman"] - orig[p]["B"]["spearman"]) > 1e-12
           or np.max(np.abs(np.array(blocks[p]["ci95"]) - np.array(orig[p]["B"]["ci95"]))) > 1e-12]
    print(f"selfcheck: {len(ALL_PROXIES) - len(bad)}/{len(ALL_PROXIES)} standard proxies reproduce "
          f"evaluation.json on B (rho + CI)" + (f"  MISMATCH: {bad}" if bad else ""), flush=True)
    return not bad


def _fmt(b) -> str:
    if not b or b.get("spearman") is None:
        return "n/a"
    return f"{b['spearman']:.3f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]"


def _write_replication_md(res: dict, orig: dict):
    lines = ["# Replication (PROTOCOL.md 9b): frozen rankers on the pre-registered new sets", "",
             "Bootstrap: 10k resamples, seed 0. Evolved rows: mean +- std over seeds. "
             "Replication sets are reported beside the originals, never merged.", "",
             "| Ranker | B orig (n=200) | B_rep (n=200) | d | A-test orig (n=50) | A_rep (n=150) | d |",
             "|---|---|---|---|---|---|---|"]
    order = sorted(ALL_PROXIES, key=lambda p: -(orig["standard"][p]["B"].get("spearman") or -9))
    for p in order:
        dB, dA = res["delta"][f"{p}:B->B_rep"], res["delta"][f"{p}:Atest->A_rep"]
        fd = lambda d: "n/a" if d["diff"] is None else f"{d['diff']:+.3f}"
        lines.append(f"| {p} | {_fmt(orig['standard'][p]['B'])} | {_fmt(res['standard'][p]['B_rep'])} | {fd(dB)} "
                     f"| {_fmt(orig['standard'][p]['Atest'])} | {_fmt(res['standard'][p]['A_rep'])} | {fd(dA)} |")
    keys = sorted({k.split(":")[0] for k in res["curve"]},
                  key=lambda k: (int(k[1:].split("_")[0]), k))
    for k in keys:
        cB, cA = res["curve"].get(f"{k}:B->B_rep"), res["curve"].get(f"{k}:Atest->A_rep")
        f = lambda c, side: "n/a" if not c else f"{c[side + '_mean']:.3f} +- {c[side + '_std']:.3f}"
        fd = lambda c: "n/a" if not c else f"{c['diff_mean']:+.3f}"
        lines.append(f"| evolved {k.replace('_', ' ')} | {f(cB, 'orig')} | {f(cB, 'rep')} | {fd(cB)} "
                     f"| {f(cA, 'orig')} | {f(cA, 'rep')} | {fd(cA)} |")
    lines += ["", f"Noise ceiling (original 50x3 seeds): {orig['meta']['noise_ceiling_spearman']} "
              "- no replication seeds were pre-registered."]
    open(os.path.join(RESULTS_DIR, "replication.md"), "w").write("\n".join(lines) + "\n")


def main_replication():
    torch.set_num_threads(max(2, os.cpu_count() - 2))
    if not selfcheck():
        raise SystemExit("selfcheck FAILED - sealed replication not run")
    orig = json.load(open(os.path.join(RESULTS_DIR, "evaluation.json")))
    sets = {name: _load_set(name) for name in REP_SETS}
    results = {"standard": {}, "evolved": {}, "wilcoxon": {}, "delta": {}, "curve": {}}

    # ---- standard proxies (live computation, frozen batch; cached per set) ----
    std_blocks = {name: _standard_blocks(name, s) for name, s in sets.items()}
    for p in ALL_PROXIES:
        results["standard"][p] = {name: std_blocks[name][p] for name in sets}
        print(f"std {p:10s} B_rep rho={results['standard'][p]['B_rep'].get('spearman')}", flush=True)

    # ---- committed elites (cached statistics) ----
    print("loading replication ctxs...", flush=True)
    ctx = {name: [load_ctx(name, a) for a in s["ids"]] for name, s in sets.items()}
    evolved_rhos = {}
    for f in sorted(glob.glob(os.path.join(EVOLVED_DIR, "*.json"))):
        r = json.load(open(f))
        tree = tree_from_json(r["tree_json"])
        tag = r["tag"] + ("" if "warm" not in f or r["tag"].endswith("_warm") else "_warm")
        block = {"tree": r["tree"], "budget": r["budget"], "seed": r["seed"],
                 "warm": r["tag"].endswith("_warm") if "tag" in r else False,
                 "terminals_used": r["terminals_used"]}
        for name, s in sets.items():
            sc = gp.scores_for(tree, ctx[name])
            block[name] = (metric_block(sc, s["accs"], s["flops"]) if sc
                           else {"spearman": None, "note": f"invalid on {name}"})
        results["evolved"][tag] = block
        variant = "warm" if tag.endswith("_warm") else ("vision" if r["budget"] == 0 else "cold")
        rho = block["B_rep"].get("spearman")
        if rho is not None:
            evolved_rhos.setdefault((r["budget"], variant), []).append(rho)
        print(f"evolved {tag:20s} B_rep rho={rho}", flush=True)

    # ---- Wilcoxon: evolved vs #params across seeds on B_rep ----
    params_rho = results["standard"]["params"]["B_rep"]["spearman"]
    for (budget, variant), rhos in sorted(evolved_rhos.items()):
        if len(rhos) >= 6:
            w = st.wilcoxon(np.array(rhos) - params_rho, alternative="greater")
            results["wilcoxon"][f"N{budget}_{variant}_vs_params"] = {
                "n_seeds": len(rhos), "mean_rho": float(np.mean(rhos)),
                "params_rho": params_rho, "p_value": float(w.pvalue)}

    # ---- delta vs originals ----
    for p in ALL_PROXIES:
        for name, okey in REP_SETS.items():
            o, n = orig["standard"][p][okey], results["standard"][p][name]
            ok = o.get("spearman") is not None and n.get("spearman") is not None
            results["delta"][f"{p}:{okey}->{name}"] = {
                "orig": o.get("spearman"), "orig_ci95": o.get("ci95"),
                "rep": n.get("spearman"), "rep_ci95": n.get("ci95"),
                "diff": (n["spearman"] - o["spearman"]) if ok else None}
    for name, okey in REP_SETS.items():
        by = {}
        for tag, v in results["evolved"].items():
            key = (v["budget"], "warm" if v["warm"] else ("vision" if v["budget"] == 0 else "cold"))
            ro = orig["evolved"].get(tag, {}).get(okey, {}).get("spearman")
            rn = v[name].get("spearman")
            if ro is not None and rn is not None:
                by.setdefault(key, []).append((ro, rn))
        for (budget, variant), pairs in sorted(by.items()):
            o_, n_ = np.array(pairs).T
            results["curve"][f"N{budget}_{variant}:{okey}->{name}"] = {
                "n_seeds": len(pairs), "orig_mean": float(o_.mean()), "orig_std": float(o_.std()),
                "rep_mean": float(n_.mean()), "rep_std": float(n_.std()),
                "diff_mean": float((n_ - o_).mean())}

    results["meta"] = {
        "protocol": "PROTOCOL.md 9b: frozen proxies + committed elites scored once on the "
                    "pre-registered replication sets; reported beside the originals, never merged",
        "sets": {name: {"n": len(sets[name]["ids"]), "replicates": okey} for name, okey in REP_SETS.items()},
        "noise_ceiling_spearman": orig["meta"]["noise_ceiling_spearman"],
        "noise_ceiling_note": "original 50x3-seed estimate; no replication seeds were pre-registered",
        "original_evaluation_git_hash": orig["meta"]["git_hash"],
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat()}
    out = os.path.join(RESULTS_DIR, "replication.json")
    json.dump(results, open(out, "w"), indent=1)
    _write_replication_md(results, orig)
    print(f"\nsaved -> {out}")

    print("\n=== REPLICATION SUMMARY (B -> B_rep Spearman) ===")
    for p in ["flops", "nwot", "params"]:
        d = results["delta"][f"{p}:B->B_rep"]
        print(f"{p:8s} orig={d['orig']:.3f} rep={d['rep']:.3f} {d['rep_ci95']} d={d['diff']:+.3f}")
    for k, c in sorted(results["curve"].items(), key=lambda kv: int(kv[0][1:].split("_")[0])):
        if k.endswith(":B->B_rep"):
            print(f"{k.split(':')[0]:12s} orig={c['orig_mean']:.3f} rep={c['rep_mean']:.3f} "
                  f"+- {c['rep_std']:.3f} d={c['diff_mean']:+.3f} (n={c['n_seeds']})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--replication", action="store_true",
                    help="9b: score frozen rankers on A_rep/B_rep -> results/replication.json")
    ap.add_argument("--selfcheck", action="store_true",
                    help="replication code path must reproduce evaluation.json on original B")
    args = ap.parse_args()
    if args.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)
    main_replication() if args.replication else main()
