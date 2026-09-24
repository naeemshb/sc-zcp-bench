"""Post-hoc descriptive check (2026-09-23; not pre-registered; selects nothing) behind the Sec. 4.4
mechanism sentence: on the TC-ResNet family, at fixed FLOPs, does a higher parameter count go with more
temporal striding, and does striding cost accuracy at fixed FLOPs?

Partial Spearman = Pearson correlation of rank residuals after regressing on rank(log10 FLOPs), as in the
9c-partial analysis. 10,000-resample percentile bootstrap over architectures, seed 0, architectures in
sorted arch_id order. Reads seed-0 ground-truth records only (config, params, flops, test_acc).

Usage: .venv/bin/python -m speech_zcp.striding_check   -> results/striding_check.{json,md}
"""
import glob
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy.stats as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_BOOT, BOOT_SEED = 10_000, 0
SETS = {"B": ["gt_B"], "B_rep_pooled": ["gt_B_rep", "gt_B_rep2"]}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return "unknown"


def load(dirs):
    recs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(RESULTS_DIR, d, "*_s0.json")):
            r = json.load(open(f)); recs[r["arch_id"]] = r
    ids = sorted(recs)
    cfg = lambda a: recs[a]["config"]
    return {
        "ids": ids,
        "stride": np.array([float(np.prod([b["stride"] for b in cfg(a)["blocks"]])) for a in ids]),
        "n_blocks": np.array([float(len(cfg(a)["blocks"])) for a in ids]),
        "params": np.array([float(recs[a]["params"]) for a in ids]),
        "log_flops": np.log10([float(recs[a]["flops"]) for a in ids]),
        "acc": np.array([float(recs[a]["test_acc"]) for a in ids]),
    }


def partial(x, y, z):
    rx, ry, rz = st.rankdata(x), st.rankdata(y), st.rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(st.pearsonr(res(rx), res(ry))[0])


def block(x, y, z):
    n = len(x); rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n); boots[b] = partial(x[i], y[i], z[i])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"partial": partial(x, y, z), "ci95": [float(lo), float(hi)], "n": int(n)}


def main():
    out = {"statistic": "partial Spearman controlling for log10 FLOPs (rank residuals); stride = product of block strides",
           "sets": {}, "meta": {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "order": "sorted arch_id",
                                "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat(),
                                "note": "post-hoc descriptive check, not pre-registered; ground truth seed 0; nothing selected"}}
    lines = ["# Striding check (post-hoc, descriptive): TC-ResNet family, controlling for log10 FLOPs", "",
             "| set | n | #params vs stride | stride vs accuracy | #params vs accuracy | #blocks vs accuracy |", "|---|---|---|---|---|---|"]
    for name, dirs in SETS.items():
        s = load(dirs)
        r = {"params_vs_stride": block(s["params"], s["stride"], s["log_flops"]),
             "stride_vs_acc": block(s["stride"], s["acc"], s["log_flops"]),
             "params_vs_acc": block(s["params"], s["acc"], s["log_flops"]),
             "nblocks_vs_acc": block(s["n_blocks"], s["acc"], s["log_flops"]),
             "plain_spearman_stride_vs_acc": float(st.spearmanr(s["stride"], s["acc"]).statistic),
             "stride_values": {int(k): int(v) for k, v in zip(*np.unique(s["stride"], return_counts=True))}}
        out["sets"][name] = r
        f = lambda b: f"{b['partial']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]"
        lines.append(f"| {name} | {r['params_vs_stride']['n']} | {f(r['params_vs_stride'])} | {f(r['stride_vs_acc'])} | {f(r['params_vs_acc'])} | {f(r['nblocks_vs_acc'])} |")
        print(f"{name:13s} n={r['params_vs_stride']['n']}  params~stride {f(r['params_vs_stride'])}  stride~acc {f(r['stride_vs_acc'])}  params~acc {f(r['params_vs_acc'])}", flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "striding_check.json"), "w"), indent=1)
    open(os.path.join(RESULTS_DIR, "striding_check.md"), "w").write("\n".join(lines) + "\n")
    print("saved -> results/striding_check.{json,md}")


if __name__ == "__main__":
    main()
