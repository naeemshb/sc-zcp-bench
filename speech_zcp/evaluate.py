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

--ceiling (PROTOCOL.md 9b-seeds): test-retest noise ceilings from seeds {0,1,2}
for B, B_orig50, Atest, B_rep, A_rep -> results/noise_ceiling.json. Ground
truth is never changed by extra seeds. --ceiling-selfcheck: B_orig50 must
reproduce evaluation.json's 0.9092 (no file written).
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
from .evolve import SPLITS_DIR, a_pool_and_test, tree_from_json
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
# Pre-registered replication (PROTOCOL.md 9b, 9b-rep2)
# ---------------------------------------------------------------------------
# replication set -> original column it replicates
REP_SETS = {"B_rep": "B", "A_rep": "Atest"}
REP2_SETS = {"B_rep2": "B", "A_rep2": "Atest"}
# pooled sets (declared in 9b-rep2 before rep2 existed): members in order, original column
REP2_POOLED = {"B_rep_pooled": (["B_rep", "B_rep2"], "B"), "A_rep_pooled": (["A_rep", "A_rep2"], "Atest")}


def _load_set(name: str) -> dict:
    gt = gt_map(name)
    index = json.load(open(os.path.join(CACHE_DIR, f"{name}_index.json")))
    ids = sorted(index["archs"], key=lambda a: index["archs"][a]["i"])
    return {"ids": ids, "accs": [gt[a]["test_acc"] for a in ids],
            "flops": [gt[a]["flops"] for a in ids],
            "cfgs": {a: gt[a]["config"] for a in ids}, "members": [name]}


def _load_pooled(members: list[str]) -> dict:
    parts = [_load_set(m) for m in members]
    return {"ids": sum((p["ids"] for p in parts), []), "accs": sum((p["accs"] for p in parts), []),
            "flops": sum((p["flops"] for p in parts), []),
            "cfgs": {k: v for p in parts for k, v in p["cfgs"].items()}, "members": list(members)}


def _std_scores_for(s: dict) -> dict:
    """proxy -> {arch_id: score}; per member from the cached per-set files (union for pooled)."""
    out = {p: {} for p in ALL_PROXIES}
    for m in s["members"]:
        sub = {a: s["cfgs"][a] for a in _load_set(m)["ids"]} if len(s["members"]) > 1 else s["cfgs"]
        sc = standard_scores(m, list(sub), sub)
        for p in ALL_PROXIES:
            out[p].update(sc[p])
    return out


def _ctx_for(s: dict) -> list:
    ctx = []
    for m in s["members"]:
        ctx += [load_ctx(m, a) for a in _load_set(m)["ids"]]
    return ctx


def _standard_blocks(name: str, s: dict) -> dict:
    std = _std_scores_for(s)
    return {p: metric_block([std[p][a] for a in s["ids"]], s["accs"], s["flops"]) for p in ALL_PROXIES}


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


def _write_replication_md(res: dict, orig: dict, set_cols: dict, path: str, title: str):
    """set_cols: ordered {set name: (original column key, header label, n)}"""
    heads = ["Ranker"]
    for okey in ["B", "Atest"]:
        heads.append(f"{'B' if okey == 'B' else 'A-test'} orig (n={200 if okey == 'B' else 50})")
        heads += [f"{lab} (n={n})" for name, (k, lab, n) in set_cols.items() if k == okey]
    lines = [f"# {title}", "",
             "Bootstrap: 10k resamples, seed 0. Evolved rows: mean +- std over seeds. "
             "Replication sets are reported beside the originals, never merged with them.", "",
             "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    order = sorted(ALL_PROXIES, key=lambda p: -(orig["standard"][p]["B"].get("spearman") or -9))
    for p in order:
        row = [p]
        for okey in ["B", "Atest"]:
            row.append(_fmt(orig["standard"][p][okey]))
            row += [_fmt(res["standard"][p][name]) for name, (k, _, _) in set_cols.items() if k == okey]
        lines.append("| " + " | ".join(row) + " |")
    keys = sorted({k.split(":")[0] for k in res["curve"]}, key=lambda k: (int(k[1:].split("_")[0]), k))
    for k in keys:
        row = [f"evolved {k.replace('_', ' ')}"]
        for okey in ["B", "Atest"]:
            c0 = next((c for kk, c in res["curve"].items() if kk.startswith(k + ":" + okey + "->")), None)
            row.append("n/a" if not c0 else f"{c0['orig_mean']:.3f} +- {c0['orig_std']:.3f}")
            for name, (kk_, _, _) in set_cols.items():
                if kk_ != okey:
                    continue
                c = res["curve"].get(f"{k}:{okey}->{name}")
                row.append("n/a" if not c else f"{c['rep_mean']:.3f} +- {c['rep_std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def run_replication(sets: dict, pooled: dict | None, out_name: str | None, title: str) -> dict:
    """Score the frozen proxies and committed elites once on `sets` (name -> original
    column) plus `pooled` (name -> (members, original column)); write
    results/<out_name>.json/.md unless out_name is None (selfcheck mode)."""
    torch.set_num_threads(max(2, os.cpu_count() - 2))
    orig = json.load(open(os.path.join(RESULTS_DIR, "evaluation.json")))
    loaded, okeys = {}, {}
    for name, okey in sets.items():
        loaded[name], okeys[name] = _load_set(name), okey
    for name, (members, okey) in (pooled or {}).items():
        loaded[name], okeys[name] = _load_pooled(members), okey
    results = {"standard": {}, "evolved": {}, "wilcoxon": {}, "delta": {}, "curve": {}}

    std_blocks = {name: _standard_blocks(name, s) for name, s in loaded.items()}
    for p in ALL_PROXIES:
        results["standard"][p] = {name: std_blocks[name][p] for name in loaded}
        print(f"std {p:10s} " + "  ".join(f"{n}={results['standard'][p][n].get('spearman'):.3f}"
                                          if results['standard'][p][n].get('spearman') is not None else f"{n}=n/a"
                                          for n in loaded), flush=True)

    print("loading ctxs...", flush=True)
    ctx = {name: _ctx_for(s) for name, s in loaded.items()}
    evolved_rhos = {name: {} for name in loaded}
    for f in sorted(glob.glob(os.path.join(EVOLVED_DIR, "*.json"))):
        r = json.load(open(f))
        tree = tree_from_json(r["tree_json"])
        tag = r["tag"] + ("" if "warm" not in f or r["tag"].endswith("_warm") else "_warm")
        block = {"tree": r["tree"], "budget": r["budget"], "seed": r["seed"],
                 "warm": r["tag"].endswith("_warm") if "tag" in r else False,
                 "terminals_used": r["terminals_used"]}
        variant = "warm" if block["warm"] else ("vision" if r["budget"] == 0 else "cold")
        for name, s in loaded.items():
            sc = gp.scores_for(tree, ctx[name])
            block[name] = (metric_block(sc, s["accs"], s["flops"]) if sc
                           else {"spearman": None, "note": f"invalid on {name}"})
            rho = block[name].get("spearman")
            if rho is not None:
                evolved_rhos[name].setdefault((r["budget"], variant), []).append(rho)
        results["evolved"][tag] = block
        print(f"evolved {tag:20s} " + "  ".join(f"{n}={block[n].get('spearman'):.3f}"
                                                if block[n].get('spearman') is not None else f"{n}=n/a" for n in loaded), flush=True)

    # ---- Wilcoxon: evolved vs #params across seeds, on every Space-B-family set ----
    for name in loaded:
        if okeys[name] != "B":
            continue
        params_rho = results["standard"]["params"][name]["spearman"]
        for (budget, variant), rhos in sorted(evolved_rhos[name].items()):
            if len(rhos) >= 6:
                w = st.wilcoxon(np.array(rhos) - params_rho, alternative="greater")
                results["wilcoxon"][f"{name}:N{budget}_{variant}_vs_params"] = {
                    "n_seeds": len(rhos), "mean_rho": float(np.mean(rhos)),
                    "params_rho": params_rho, "p_value": float(w.pvalue)}

    # ---- delta vs originals ----
    for p in ALL_PROXIES:
        for name, okey in okeys.items():
            o, n = orig["standard"][p][okey], results["standard"][p][name]
            ok = o.get("spearman") is not None and n.get("spearman") is not None
            results["delta"][f"{p}:{okey}->{name}"] = {
                "orig": o.get("spearman"), "orig_ci95": o.get("ci95"),
                "rep": n.get("spearman"), "rep_ci95": n.get("ci95"),
                "diff": (n["spearman"] - o["spearman"]) if ok else None}
    for name, okey in okeys.items():
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
        "protocol": "PROTOCOL.md 9b / 9b-rep2: frozen proxies + committed elites scored once on "
                    "pre-registered replication sets; reported beside the originals, never merged",
        "sets": {name: {"n": len(loaded[name]["ids"]), "replicates": okeys[name],
                        "members": loaded[name]["members"]} for name in loaded},
        "noise_ceiling_spearman": orig["meta"]["noise_ceiling_spearman"],
        "noise_ceiling_note": "see results/noise_ceiling.json for the seed-replicated ceilings",
        "original_evaluation_git_hash": orig["meta"]["git_hash"],
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat()}
    if out_name:
        out = os.path.join(RESULTS_DIR, f"{out_name}.json")
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: replication evaluations run once; delete it deliberately to recompute")
        json.dump(results, open(out, "w"), indent=1)
        cols = {name: (okeys[name], name.replace("_pooled", " pooled").replace("_", " "), len(loaded[name]["ids"]))
                for name in loaded}
        _write_replication_md(results, orig, cols, os.path.join(RESULTS_DIR, f"{out_name}.md"), title)
        print(f"\nsaved -> {out}")
        print("\n=== SUMMARY (Spearman) ===")
        for name in loaded:
            for p in ["flops", "nwot", "params"]:
                d = results["delta"][f"{p}:{okeys[name]}->{name}"]
                print(f"{name:13s} {p:7s} orig={d['orig']:.3f} rep={d['rep']:.3f} {[round(x, 3) for x in d['rep_ci95']]} d={d['diff']:+.3f}")
            for k, c in sorted(results["curve"].items(), key=lambda kv: int(kv[0][1:].split("_")[0])):
                if k.endswith(f"->{name}") and "_cold" in k or (k.endswith(f"->{name}") and "vision" in k):
                    print(f"{name:13s} {k.split(':')[0]:11s} orig={c['orig_mean']:.3f} rep={c['rep_mean']:.3f} +- {c['rep_std']:.3f} d={c['diff_mean']:+.3f}")
    return results


def main_replication():
    if not selfcheck():
        raise SystemExit("selfcheck FAILED - sealed replication not run")
    run_replication(REP_SETS, None, "replication",
                    "Replication (PROTOCOL.md 9b): frozen rankers on the pre-registered new sets")


def replication_selfcheck() -> bool:
    """The generalized code path re-run on {B_rep, A_rep} must reproduce replication.json
    exactly (every standard and evolved rho + CI). Writes nothing."""
    ref = json.load(open(os.path.join(RESULTS_DIR, "replication.json")))
    got = run_replication(REP_SETS, None, None, "")
    bad = []
    for p in ALL_PROXIES:
        for name in REP_SETS:
            a, b = got["standard"][p][name], ref["standard"][p][name]
            if a.get("spearman") != b.get("spearman") or a.get("ci95") != b.get("ci95"):
                bad.append(f"std:{p}:{name}")
    for tag, v in ref["evolved"].items():
        for name in REP_SETS:
            a, b = got["evolved"].get(tag, {}).get(name, {}), v[name]
            if a.get("spearman") != b.get("spearman") or a.get("ci95") != b.get("ci95"):
                bad.append(f"evolved:{tag}:{name}")
    n_checked = len(ALL_PROXIES) * 2 + len(ref["evolved"]) * 2
    print(f"replication-selfcheck: {n_checked - len(bad)}/{n_checked} blocks reproduce replication.json exactly"
          + (f"  MISMATCH: {bad[:8]}" if bad else ""), flush=True)
    return not bad


def main_replication2():
    if not selfcheck() or not replication_selfcheck():
        raise SystemExit("selfcheck FAILED - sealed rep2 evaluation not run")
    run_replication(REP2_SETS, REP2_POOLED, "replication2",
                    "Second replication (PROTOCOL.md 9b-rep2): rep2 alone and pooled rep ∪ rep2")


# ---------------------------------------------------------------------------
# Test-retest noise ceilings (PROTOCOL.md 9b-seeds, declared 2026-09-02; 9b-rep2 --extend)
# ---------------------------------------------------------------------------
SEEDS = (0, 1, 2)
# set -> dict(dirs=[gt dirs in order], sub=subset rule, comp=comparator proxy, tags=[proxy-score tags], n=declared n)
CEILING_SETS = {
    "B":        {"dirs": ["B"], "sub": None, "comp": "flops", "tags": ["B"], "n": 200},
    "B_orig50": {"dirs": ["B"], "sub": "orig50", "comp": "flops", "tags": ["B"], "n": 50},
    "Atest":    {"dirs": ["A"], "sub": "atest", "comp": "params", "tags": ["Atest"], "n": 50},
    "B_rep":    {"dirs": ["B_rep"], "sub": None, "comp": "flops", "tags": ["B_rep"], "n": 200},
    "A_rep":    {"dirs": ["A_rep"], "sub": None, "comp": "params", "tags": ["A_rep"], "n": 150},
}
CEILING_SETS_EXTEND = {  # 9b-rep2: appended by --ceiling --extend, existing blocks untouched
    "B_rep2":       {"dirs": ["B_rep2"], "sub": None, "comp": "flops", "tags": ["B_rep2"], "n": 100},
    "A_rep2":       {"dirs": ["A_rep2"], "sub": None, "comp": "params", "tags": ["A_rep2"], "n": 100},
    "B_rep_pooled": {"dirs": ["B_rep", "B_rep2"], "sub": None, "comp": "flops", "tags": ["B_rep", "B_rep2"], "n": 300},
    "A_rep_pooled": {"dirs": ["A_rep", "A_rep2"], "sub": None, "comp": "params", "tags": ["A_rep", "A_rep2"], "n": 250},
}


def _seed_accs(gt_dirs) -> dict:
    """arch_id -> {seed: test_acc} over every seed file in results/gt_<dir>/ (dirs merged)."""
    out = {}
    for gt_dir in ([gt_dirs] if isinstance(gt_dirs, str) else gt_dirs):
        for f in glob.glob(os.path.join(RESULTS_DIR, f"gt_{gt_dir}", "*.json")):
            r = json.load(open(f))
            out.setdefault(r["arch_id"], {})[r["seed"]] = r["test_acc"]
    return out


def _canonical_order(gt_dirs, sub) -> list[str]:
    """Content-defined architecture order (never filesystem order) so the seeded
    bootstrap reproduces on any machine: sample_archs position (== sweep index)
    for the released spaces, the sealed A-test list, or the replication split(s)
    in member order."""
    if sub == "atest":
        return list(a_pool_and_test()[1])
    order = []
    for gt_dir in ([gt_dirs] if isinstance(gt_dirs, str) else gt_dirs):
        if gt_dir in ("A", "B"):
            o = [spaces.arch_id(c) for c in spaces.sample_archs(gt_dir, 250 if gt_dir == "A" else 200)]
            order += o[:50] if sub == "orig50" else o
        else:
            order += list(json.load(open(os.path.join(SPLITS_DIR, f"replication_{gt_dir}.json")))["arch_ids"])
    return order


def _mean_pairwise(M: np.ndarray, pairs) -> float:
    return float(np.mean([st.spearmanr(M[:, i], M[:, j]).statistic for i, j in pairs]))


def ceiling_block(A: np.ndarray, comp=None) -> dict:
    """A: (n_arch, n_seed) test accuracies, canonical arch order. Ceiling = mean
    pairwise Spearman across seeds; percentile bootstrap over architectures
    (N_BOOT, BOOT_SEED). comp: optional comparator proxy scores in the same
    order -> supplementary paired-bootstrap CI of ceiling - Spearman(comp,
    seed-0 acc) using identical resamples for both terms."""
    pairs = list(itertools.combinations(range(A.shape[1]), 2))
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(A), size=(N_BOOT, len(A)))
    boots = np.array([_mean_pairwise(A[idx[b]], pairs) for b in range(N_BOOT)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    out = {"spearman": _mean_pairwise(A, pairs), "ci95": [float(lo), float(hi)], "n": int(len(A)),
           "pairwise": {f"{i}-{j}": float(st.spearmanr(A[:, i], A[:, j]).statistic) for i, j in pairs},
           "seed_std_median_pts": float(np.median(A.std(axis=1)) * 100)}  # ddof=0: the paper's 0.43 pts
    if comp is not None:
        comp_rho = float(st.spearmanr(comp, A[:, 0]).statistic)
        gaps = np.array([boots[b] - st.spearmanr(comp[idx[b]], A[idx[b], 0]).statistic for b in range(N_BOOT)])
        glo, ghi = np.nanpercentile(gaps, [2.5, 97.5])
        out["paired_gap"] = {"comparator_spearman": comp_rho, "gap": out["spearman"] - comp_rho,
                             "ci95": [float(glo), float(ghi)],
                             "frac_resamples_gap_le_0": float(np.mean(gaps <= 0))}
    return out


def _ceiling_for(name: str, spec: dict) -> dict:
    accs = _seed_accs(spec["dirs"])
    ids = [a for a in _canonical_order(spec["dirs"], spec["sub"]) if a in accs and all(s in accs[a] for s in SEEDS)]
    ok = [a for a in ids if all(isinstance(accs[a][s], (int, float)) and np.isfinite(accs[a][s]) for s in SEEDS)]
    dropped = len(ids) - len(ok)
    if len(ok) < 10:
        print(f"ceiling {name:13s} n={len(ok):3d}/{spec['n']} (incomplete)", flush=True)
        return {"spearman": None, "n": len(ok), "declared_n": spec["n"], "n_dropped_nonfinite": dropped,
                "note": "fewer than 10 archs with all 3 seeds"}
    A = np.array([[accs[a][s] for s in SEEDS] for a in ok], dtype=float)
    scores = {}
    for tag in spec["tags"]:
        sp = os.path.join(RESULTS_DIR, f"proxy_scores_{tag}.json")
        if os.path.exists(sp):
            scores.update(json.load(open(sp))[spec["comp"]])
    comp = np.array([scores[a] for a in ok], dtype=float) if all(a in scores for a in ok) else None
    r = ceiling_block(A, comp)
    r.update({"declared_n": spec["n"], "n_dropped_nonfinite": dropped,
              "comparator": spec["comp"] if comp is not None else None, "members": spec["dirs"]})
    g = r.get("paired_gap")
    print(f"ceiling {name:13s} n={r['n']:3d}/{spec['n']} rho={r['spearman']:.4f} "
          f"ci=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] seed_std_med={r['seed_std_median_pts']:.2f}pts"
          + (f" | {spec['comp']} {g['comparator_spearman']:.3f} gap {g['gap']:+.3f} "
             f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]" if g else ""), flush=True)
    return r


def _cross_check(name: str, r: dict, ev: dict, rep: dict | None, rep2: dict | None):
    """The comparator's full-set Spearman must equal the sealed artifacts."""
    g = r.get("paired_gap")
    if not g or r["n"] != r["declared_n"]:
        return
    ref = None
    if name == "B":
        ref = ev["standard"][r["comparator"]]["B"]["spearman"]
    elif name in ("B_rep", "A_rep") and rep:
        ref = rep["standard"][r["comparator"]][name]["spearman"]
    elif rep2 and name in rep2["standard"][r["comparator"]]:
        ref = rep2["standard"][r["comparator"]][name]["spearman"]
    if ref is None:
        return
    same = abs(g["comparator_spearman"] - ref) < 1e-9
    print(f"  cross-check {name}: {r['comparator']} rho {g['comparator_spearman']:.4f} vs sealed {ref:.4f} -> "
          f"{'OK' if same else 'MISMATCH'}", flush=True)
    if not same:
        raise SystemExit(f"{name}: comparator scores misaligned with sealed artifact")


def noise_ceilings(write: bool = True, extend: bool = False) -> dict:
    out_path = os.path.join(RESULTS_DIR, "noise_ceiling.json")
    ev = json.load(open(os.path.join(RESULTS_DIR, "evaluation.json")))
    _load = lambda n: json.load(open(os.path.join(RESULTS_DIR, n))) if os.path.exists(os.path.join(RESULTS_DIR, n)) else None
    rep, rep2 = _load("replication.json"), _load("replication2.json")
    if extend:
        if not os.path.exists(out_path):
            raise SystemExit("--extend needs an existing results/noise_ceiling.json")
        results = json.load(open(out_path))
        todo = {k: v for k, v in CEILING_SETS_EXTEND.items() if k not in results}
        if not todo:
            raise SystemExit("nothing to extend: all rep2 sets already present")
    else:
        if write and os.path.exists(out_path):
            raise SystemExit(f"{out_path} exists: the ceiling is computed once (9b-seeds); use --extend for new sets")
        results, todo = {}, CEILING_SETS
    for name, spec in todo.items():
        results[name] = _ceiling_for(name, spec)
        _cross_check(name, results[name], ev, rep, rep2)
    if not extend:
        orig = ev["meta"]["noise_ceiling_spearman"]
        ok_ = results["B_orig50"].get("spearman") is not None and abs(results["B_orig50"]["spearman"] - orig) < 5e-5
        print(f"selfcheck: B_orig50 rho {results['B_orig50'].get('spearman')} vs evaluation.json {orig} -> "
              f"{'OK' if ok_ else 'MISMATCH'}", flush=True)
        if not ok_:
            raise SystemExit("ceiling selfcheck FAILED")
    if write:
        if extend:
            results["meta"].setdefault("extended", []).append(
                {"sets": list(todo), "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            results["meta"] = {
                "definition": "mean of pairwise Spearman over seeds (0,1,2) of test_acc; percentile bootstrap over archs",
                "ordering": "sample_archs position (A/B; orig50 = first 50), sealed A-test list (Atest), "
                            "replication split arch_ids (rep sets, members in order) - never filesystem order",
                "seed_std_convention": "median over archs of np.std(test_acc over 3 seeds, ddof=0) x 100",
                "paired_gap": "supplementary: ceiling - Spearman(comparator, seed-0 acc), identical resamples; "
                              "reported, not deciding (PROTOCOL.md 9b-seeds decision pairing)",
                "provenance": "B_orig50 reproduces the 0.9092 literal in evaluation.json meta",
                "seeds": list(SEEDS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                "declared_n": {k: v["n"] for k, v in CEILING_SETS.items()},
                "ground_truth_note": "all proxy/evolved correlations use seed-0 accuracies only; extra seeds estimate ceilings",
                "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat()}
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"saved -> {out_path}")
    return results


# ---------------------------------------------------------------------------
# 9c-partial: signal beyond compute — partial Spearman controlling for log10 FLOPs
# ---------------------------------------------------------------------------
PARTIAL_SETS = ["B", "B_rep_pooled", "Atest", "A_rep_pooled"]


def _load_partial_set(name: str) -> dict:
    if name == "Atest":
        gt = gt_map("A")
        ids = list(a_pool_and_test()[1])
        std = json.load(open(os.path.join(RESULTS_DIR, "proxy_scores_Atest.json")))
        return {"ids": ids, "accs": [gt[a]["test_acc"] for a in ids], "flops": [gt[a]["flops"] for a in ids],
                "std": std, "ctx": [load_ctx("A", a) for a in ids]}
    s = _load_pooled(REP2_POOLED[name][0]) if name in REP2_POOLED else _load_set(name)
    return {"ids": s["ids"], "accs": s["accs"], "flops": s["flops"], "std": _std_scores_for(s), "ctx": _ctx_for(s)}


def partial_spearman(x, y, z) -> float:
    """Pearson correlation of the residuals of rank(x) and rank(y) after regressing
    each on rank(z) (average ranks for ties). NaN if x or y is degenerate."""
    rx, ry, rz = (st.rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))
    rz = rz - rz.mean()
    if not rz.any():
        return float("nan")
    def resid(r):
        r = r - r.mean()
        return r - rz * (r @ rz) / (rz @ rz)
    ex, ey = resid(rx), resid(ry)
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d > 0 else float("nan")


def _partial_block(scores, accs, logf, boot: bool) -> dict:
    scores, accs, logf = (np.asarray(v, dtype=float) for v in (scores, accs, logf))
    ok = np.isfinite(scores)
    if ok.sum() < 0.9 * len(scores):
        return {"partial": None, "note": f"only {int(ok.sum())}/{len(scores)} finite"}
    sc, ac, lf = scores[ok], accs[ok], logf[ok]
    out = {"partial": partial_spearman(sc, ac, lf), "n": int(ok.sum())}
    if boot:
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(sc), size=(N_BOOT, len(sc)))
        b = np.array([partial_spearman(sc[i], ac[i], lf[i]) for i in idx])
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        out["ci95"] = [float(lo), float(hi)]
    return out


def partial_flops():
    out_path = os.path.join(RESULTS_DIR, "partial_flops.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists: computed once; delete deliberately to recompute")
    torch.set_num_threads(max(2, os.cpu_count() - 2))
    results = {"standard": {p: {} for p in ALL_PROXIES}, "evolved": {}, "evolved_by_budget": {}}
    for name in PARTIAL_SETS:
        s = _load_partial_set(name)
        logf = np.log10(np.asarray(s["flops"], dtype=float))
        for p in ALL_PROXIES:
            results["standard"][p][name] = _partial_block([s["std"][p][a] for a in s["ids"]], s["accs"], logf, boot=True)
        by = {}
        for f in sorted(glob.glob(os.path.join(EVOLVED_DIR, "*.json"))):
            r = json.load(open(f))
            tag = r["tag"] + ("" if "warm" not in f or r["tag"].endswith("_warm") else "_warm")
            if tag.endswith("_warm"):
                continue  # cold seeds only, as in the tercile table
            sc = gp.scores_for(tree_from_json(r["tree_json"]), s["ctx"])
            blk = _partial_block(sc, s["accs"], logf, boot=False) if sc else {"partial": None, "note": "invalid"}
            results["evolved"].setdefault(tag, {})[name] = blk
            if blk.get("partial") is not None and np.isfinite(blk["partial"]):
                by.setdefault(r["budget"], []).append(blk["partial"])
        for N, v in by.items():
            results["evolved_by_budget"].setdefault(f"N{N}", {})[name] = {
                "mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
        print(f"{name:13s} n={len(s['ids'])}  " + "  ".join(
            f"{p}={results['standard'][p][name]['partial']:+.2f}" for p in ["nwot", "params", "synflow", "zen"])
              + "  evolvedN200=" + f"{results['evolved_by_budget'].get('N200', {}).get(name, {}).get('mean', float('nan')):+.2f}", flush=True)
    results["meta"] = {
        "statistic": "partial Spearman: Pearson correlation of rank residuals after regressing rank(score) and "
                     "rank(test_acc) on rank(log10 FLOPs); FLOPs itself is degenerate (NaN)",
        "standard_ci": "10k percentile bootstrap over architectures, seed 0, canonical order",
        "evolved": "cold seeds, per-seed point estimates; evolved_by_budget = mean/std over seeds (no bootstrap)",
        "sets": PARTIAL_SETS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat()}
    json.dump(results, open(out_path, "w"), indent=1)
    # markdown
    hdr = {"B": "Space B (n=200)", "B_rep_pooled": "B pooled rep (n=300)", "Atest": "A-test (n=50)", "A_rep_pooled": "A pooled rep (n=250)"}
    lines = ["# 9c-partial: partial Spearman with test accuracy, controlling for log10 FLOPs", "",
             "Fixed rankers: [10k-bootstrap 95% CI]. Evolved: mean ± seed spread (cold seeds). FLOPs itself is n/a by construction.", "",
             "| Ranker | " + " | ".join(hdr[n] for n in PARTIAL_SETS) + " |", "|---|" + "---|" * len(PARTIAL_SETS)]
    order = sorted(ALL_PROXIES, key=lambda p: -(results["standard"][p]["B"].get("partial") if results["standard"][p]["B"].get("partial") is not None and np.isfinite(results["standard"][p]["B"]["partial"]) else -9))
    for p in order:
        cells = []
        for n in PARTIAL_SETS:
            b = results["standard"][p][n]
            cells.append("n/a" if b.get("partial") is None or not np.isfinite(b["partial"]) else f"{b['partial']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    for N in [0, 25, 50, 100, 200]:
        cells = []
        for n in PARTIAL_SETS:
            c = results["evolved_by_budget"].get(f"N{N}", {}).get(n)
            cells.append("n/a" if not c else f"{c['mean']:+.2f} ± {c['std']:.2f}")
        lines.append(f"| evolved N={N} | " + " | ".join(cells) + " |")
    open(os.path.join(RESULTS_DIR, "partial_flops.md"), "w").write("\n".join(lines) + "\n")
    print(f"saved -> {out_path}")

# ---------------------------------------------------------------------------
# 9e-N300: sealed evaluation of the N=300 elites (PROTOCOL.md 9e)
# ---------------------------------------------------------------------------
N300_DIR = os.path.join(RESULTS_DIR, "evolved_n300")
N300_SETS = ["B", "Atest", "B_rep_pooled", "A_rep"]


def _prep_named(name: str) -> dict:
    """ids/accs/flops/cfgs + cached standard scores + ctx list, in exactly the
    order main() / run_replication use for that set."""
    if name == "Atest":
        gtA = gt_map("A")
        ids = list(a_pool_and_test()[1])
        s = {"ids": ids, "accs": [gtA[a]["test_acc"] for a in ids], "flops": [gtA[a]["flops"] for a in ids],
             "cfgs": {a: gtA[a]["config"] for a in ids}, "members": ["A"]}
        s["std"] = standard_scores("Atest", ids, s["cfgs"])
        s["ctx"] = [load_ctx("A", a) for a in ids]
        return s
    s = _load_pooled(["B_rep", "B_rep2"]) if name == "B_rep_pooled" else _load_set(name)
    s["std"] = _std_scores_for(s)
    s["ctx"] = _ctx_for(s)
    return s


def _sealed_refs() -> dict:
    """set name -> (artifact dict, column key) for exact cross-checks."""
    ev = json.load(open(os.path.join(RESULTS_DIR, "evaluation.json")))
    rep = json.load(open(os.path.join(RESULTS_DIR, "replication.json")))
    rep2 = json.load(open(os.path.join(RESULTS_DIR, "replication2.json")))
    return {"B": (ev, "B"), "Atest": (ev, "Atest"), "A_rep": (rep, "A_rep"), "B_rep_pooled": (rep2, "B_rep_pooled")}


def _tint(v):
    v = np.asarray(v, dtype=float)
    h = float(st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v)), "t95": [float(v.mean() - h), float(v.mean() + h)]}


def run_elite_dir(elite_dir: str, keep, out_name, title: str) -> dict:
    """Score every elite json in elite_dir accepted by keep(record) on N300_SETS.
    Standard-proxy rows are recomputed from cached scores and must equal the
    sealed artifacts exactly (a built-in cross-check of ordering and inputs)."""
    torch.set_num_threads(max(2, os.cpu_count() - 2))
    refs = _sealed_refs()
    sets = {n: _prep_named(n) for n in N300_SETS}
    results = {"standard": {}, "evolved": {}, "wilcoxon": {}, "summary": {}}
    for p in ALL_PROXIES:
        results["standard"][p] = {}
        for n, s in sets.items():
            blk = metric_block([s["std"][p][a] for a in s["ids"]], s["accs"], s["flops"])
            art, col = refs[n]
            ref = art["standard"][p][col]
            if blk.get("spearman") != ref.get("spearman") or blk.get("ci95") != ref.get("ci95"):
                raise SystemExit(f"cross-check FAILED: standard {p} on {n} does not reproduce the sealed artifact")
            results["standard"][p][n] = blk
    print("cross-check: all 11 standard proxies reproduce the sealed artifacts exactly on " + ", ".join(sets), flush=True)
    rhos_B = []
    for f in sorted(glob.glob(os.path.join(elite_dir, "*.json"))):
        r = json.load(open(f))
        if not keep(r):
            continue
        tree = tree_from_json(r["tree_json"])
        block = {"tree": r["tree"], "budget": r["budget"], "seed": r["seed"], "terminals_used": r["terminals_used"]}
        for n, s in sets.items():
            sc = gp.scores_for(tree, s["ctx"])
            block[n] = metric_block(sc, s["accs"], s["flops"]) if sc else {"spearman": None, "note": f"invalid on {n}"}
        results["evolved"][r["tag"]] = block
        if block["B"].get("spearman") is not None:
            rhos_B.append(block["B"]["spearman"])
        print(f"elite {r['tag']:14s} " + "  ".join(f"{n}={block[n].get('spearman'):.3f}" if block[n].get("spearman") is not None else f"{n}=n/a" for n in sets), flush=True)
    params_rho = results["standard"]["params"]["B"]["spearman"]
    if len(rhos_B) >= 6:
        w = st.wilcoxon(np.array(rhos_B) - params_rho, alternative="greater")
        results["wilcoxon"]["vs_params_on_B"] = {"n_seeds": len(rhos_B), "mean_rho": float(np.mean(rhos_B)),
                                                 "params_rho": params_rho, "p_value": float(w.pvalue)}
    # summary vs the committed N=200 cold elites on the same sets (from the sealed artifacts)
    for n in sets:
        art, col = refs[n]
        n200 = [v[col]["spearman"] for k, v in art["evolved"].items()
                if v["budget"] == 200 and not v.get("warm") and v[col].get("spearman") is not None]
        new = [b[n]["spearman"] for b in results["evolved"].values() if b[n].get("spearman") is not None]
        results["summary"][n] = {"new": _tint(new), "n200_committed": _tint(n200)}
    results["meta"] = {"protocol": "PROTOCOL.md 9e-N300: sealed, once; standard rows cross-checked exactly against evaluation/replication/replication2.json",
                       "elite_dir": elite_dir, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                       "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat()}
    if out_name:
        out = os.path.join(RESULTS_DIR, f"{out_name}.json")
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: sealed evaluations run once; delete it deliberately to recompute")
        json.dump(results, open(out, "w"), indent=1)
        sm = results["summary"]
        b, b200 = sm["B"]["new"], sm["B"]["n200_committed"]
        if b200["t95"][0] <= b["mean"] <= b200["t95"][1]:
            reading = f"(a) flat beyond the pool: N=300 mean {b['mean']:.3f} lies within N=200's t-interval [{b200['t95'][0]:.3f}, {b200['t95'][1]:.3f}]"
        elif b["t95"][0] > b200["t95"][1]:
            reading = f"(b) rise: N=300 t-interval [{b['t95'][0]:.3f}, {b['t95'][1]:.3f}] lies above N=200's [{b200['t95'][0]:.3f}, {b200['t95'][1]:.3f}]"
        elif b["t95"][1] < b200["t95"][0]:
            reading = "(c) N=300 lies CI-separated BELOW N=200"
        else:
            reading = f"(a') N=300 mean {b['mean']:.3f} outside N=200's t-interval but intervals overlap: no separation"
        lines = [f"# {title}", "", "| Ranker | Space B (n=200) | B pooled rep (n=300) | A-test (n=50) | A_rep (n=150) |", "|---|---|---|---|---|"]
        for p, lab in [("flops", "FLOPs"), ("nwot", "nwot"), ("params", "#params")]:
            lines.append(f"| {lab} | " + " | ".join(_fmt(results["standard"][p][n]) for n in sets) + " |")
        lines.append("| Evolved N=200 (committed) | " + " | ".join(f"{sm[n]['n200_committed']['mean']:.3f} ± {sm[n]['n200_committed']['sd']:.3f}" for n in sets) + " |")
        lines.append("| **Evolved N=300** | " + " | ".join(f"**{sm[n]['new']['mean']:.3f} ± {sm[n]['new']['sd']:.3f}**" for n in sets) + " |")
        lines += ["", "| seed | formula | terminals | B | B pooled | A-test | A_rep |", "|---|---|---|---|---|---|---|"]
        for tag, bl in sorted(results["evolved"].items(), key=lambda kv: kv[1]["seed"]):
            lines.append(f"| {bl['seed']} | `{bl['tree']}` | {','.join(bl['terminals_used'])} | " + " | ".join(f"{bl[n]['spearman']:.3f}" if bl[n].get("spearman") is not None else "n/a" for n in sets) + " |")
        wl = results["wilcoxon"].get("vs_params_on_B")
        lines += ["", f"Wilcoxon evolved-N300 > #params on Space B: p = {wl['p_value']:.3f} (n={wl['n_seeds']})" if wl else "", f"**Pinned reading (Space B):** {reading}"]
        open(os.path.join(RESULTS_DIR, f"{out_name}.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines[-2:])); print(f"saved -> {out}")
    return results


def n300_selfcheck() -> bool:
    """The N=300 code path, applied to the committed N=200 cold elites, must reproduce
    their sealed numbers on all four sets exactly (writes nothing)."""
    refs = _sealed_refs()
    got = run_elite_dir(EVOLVED_DIR, lambda r: r["budget"] == 200 and not r["tag"].endswith("_warm"), None, "")
    bad, n = [], 0
    for tag, bl in got["evolved"].items():
        for name, (art, col) in refs.items():
            n += 1
            ref = art["evolved"][tag][col]
            if bl[name].get("spearman") != ref.get("spearman") or bl[name].get("ci95") != ref.get("ci95"):
                bad.append(f"{tag}:{name}")
    print(f"n300-selfcheck: {n - len(bad)}/{n} elite blocks reproduce the sealed artifacts exactly" + (f"  MISMATCH: {bad[:6]}" if bad else ""), flush=True)
    return not bad


def main_n300():
    if not n300_selfcheck():
        raise SystemExit("n300-selfcheck FAILED - sealed N=300 evaluation not run")
    run_elite_dir(N300_DIR, lambda r: r["budget"] == 300, "n300",
                  "9e-N300: sealed evaluation of the ten N=300 cold elites (PROTOCOL.md 9e)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--replication", action="store_true",
                   help="9b: score frozen rankers on A_rep/B_rep -> results/replication.json")
    g.add_argument("--replication2", action="store_true",
                   help="9b-rep2: rep2 alone + pooled rep∪rep2 -> results/replication2.json")
    g.add_argument("--selfcheck", action="store_true",
                   help="replication code path must reproduce evaluation.json on original B")
    g.add_argument("--replication-selfcheck", action="store_true",
                   help="generalized path must reproduce replication.json on {B_rep, A_rep}; writes nothing")
    g.add_argument("--ceiling", action="store_true",
                   help="9b-seeds: test-retest ceilings -> results/noise_ceiling.json (run once)")
    g.add_argument("--ceiling-selfcheck", action="store_true",
                   help="B_orig50 must reproduce 0.9092; writes nothing")
    g.add_argument("--n300", action="store_true",
                   help="9e: score the N=300 elites once on B / A-test / pooled B rep / A_rep -> results/n300.json")
    g.add_argument("--n300-selfcheck", action="store_true",
                   help="N=300 code path must reproduce the committed N=200 elites' sealed numbers; writes nothing")
    g.add_argument("--partial", action="store_true",
                   help="9c-partial: partial Spearman | log10 FLOPs -> results/partial_flops.json (run once)")
    ap.add_argument("--extend", action="store_true",
                    help="with --ceiling: append the 9b-rep2 sets to the existing artifact (existing blocks untouched)")
    args = ap.parse_args()
    if args.extend and not args.ceiling:
        ap.error("--extend requires --ceiling")
    if args.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)
    if args.replication_selfcheck:
        raise SystemExit(0 if replication_selfcheck() else 1)
    if args.n300_selfcheck:
        raise SystemExit(0 if n300_selfcheck() else 1)
    if args.n300:
        main_n300()
        raise SystemExit(0)
    if args.partial:
        partial_flops()
        raise SystemExit(0)
    if args.ceiling or args.ceiling_selfcheck:
        noise_ceilings(write=args.ceiling, extend=args.extend)
        raise SystemExit(0)
    if args.replication2:
        main_replication2()
    elif args.replication:
        main_replication()
    else:
        main()
