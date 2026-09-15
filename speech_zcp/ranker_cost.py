"""Per-architecture compute cost of every ranker family (2026-09-14), measured on the laptop CPU.
Timing only: reads architecture configs (never accuracies), computes nothing that enters a result.
Writes exploratory/cost/ranker_cost.{json,md}. Nothing under speech_zcp/ is touched."""
import json, glob, os, sys, time, platform
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, REPO)
from speech_zcp import spaces, gp_engine as gp
from speech_zcp.proxies import ALL_PROXIES, compute_proxy, params_count, flops_count
from speech_zcp.cache_stats import fixed_speech_batch, matched_noise_batch, extract_speech, load_arch, CACHE_DIR
from speech_zcp.evolve import tree_from_json

torch.set_num_threads(4)
R = os.path.join(REPO, "speech_zcp", "results")
def cfgs(space, k, seed=0):
    fs = sorted(glob.glob(os.path.join(R, f"gt_{space}", "*_s0.json")))
    rng = np.random.default_rng(seed); pick = rng.choice(len(fs), size=k, replace=False)
    return [(json.load(open(fs[i]))["arch_id"], json.load(open(fs[i]))["config"]) for i in pick]
sample = {"A": cfgs("A", 20), "B": cfgs("B", 20)}
x, y = fixed_speech_batch(); xn = matched_noise_batch(x)
def med(ts): return float(np.median(ts))
out = {"protocol": {"device": "cpu", "threads": 4, "batch": int(x.shape[0]), "n_archs_per_space": 20,
                    "machine": platform.platform(), "note": "median seconds per architecture; builder() includes model construction"}}
for sp, lst in sample.items():
    res = {}
    # trivial statistics
    t = []
    for aid, cfg in lst:
        t0 = time.perf_counter(); m = spaces.build_model(cfg, init_seed=0); params_count(m); t.append(time.perf_counter() - t0)
    res["params (build + count)"] = med(t)
    t = []
    for aid, cfg in lst:
        t0 = time.perf_counter(); m = spaces.build_model(cfg, init_seed=0); flops_count(m, x[:1]); t.append(time.perf_counter() - t0)
    res["flops (build + hooked forward, batch 1)"] = med(t)
    # fixed ZC proxies (frozen batch of 8, as in the benchmark)
    for p in ALL_PROXIES:
        if p in ("params", "flops"): continue
        t = []
        for aid, cfg in lst:
            t0 = time.perf_counter(); compute_proxy(p, lambda: spaces.build_model(cfg, init_seed=0), x, y); t.append(time.perf_counter() - t0)
        res[f"proxy {p}"] = med(t)
    # evolved proxies: statistics cache (two forward/backward passes: real + matched-noise batch) + formula evaluation
    t = []
    for aid, cfg in lst:
        t0 = time.perf_counter(); extract_speech(cfg, x, y, xn); t.append(time.perf_counter() - t0)
    res["evolved: statistics extraction (real + noise batch)"] = med(t)
    tree = tree_from_json(json.load(open(os.path.join(R, "evolved", "A_N200_s1.json")))["tree_json"])  # asum(rl1(W))
    tree6 = tree_from_json(json.load(open(os.path.join(R, "evolved", "A_N200_s6.json")))["tree_json"])  # Gn formula
    ctxs = []
    for aid, cfg in lst:
        arrs = load_arch(sp, aid); ctx = {}
        for kind in ["W", "G", "A", "Gn", "An", "AstdT", "AnstdT"]:
            keys = sorted((k for k in arrs if k.startswith(kind + "_")), key=lambda k: int(k.split("_")[1]))
            if keys: ctx[kind] = [arrs[k] for k in keys]
        ctxs.append(ctx)
    for name, tr in [("evolved: formula eval asum(rl1(W)) on cached stats", tree), ("evolved: formula eval Gn-formula on cached stats", tree6)]:
        t = []
        for c in ctxs:
            t0 = time.perf_counter(); gp.evaluate(tr, c); t.append(time.perf_counter() - t0)
        res[name] = med(t)
    out[sp] = res
    print(f"== Space {sp} (median s/arch, CPU 4 threads) =="); [print(f"  {k:55s} {v:9.4f}") for k, v in res.items()]
json.dump(out, open(os.path.join(os.path.dirname(__file__), "ranker_cost.json"), "w"), indent=1)
print("saved")
