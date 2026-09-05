"""9c-recipe (PROTOCOL.md): are the benchmark's rankings stable under a heavier recipe?

Trains the original Space-B noise subset (sweep indices 0-49) with RECIPE_FULL
(all training clips, <=30 epochs, patience 5) into results/gt_B_fullrecipe/ and
correlates the resulting accuracies with the frozen pilot-v2 ground truth.
The full-v1 files are NEVER mixed into gt_B or used as ranker targets.

Usage:
  .venv/bin/python -m speech_zcp.recipe_check --train --device mps
  .venv/bin/python -m speech_zcp.recipe_check --analyze      # once -> results/recipe_check.json
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy.stats as st

from . import spaces
from .data_sc import load_split
from .train_gt import RECIPE_FULL, RESULTS_DIR, train_one

OUT_DIR = os.path.join(RESULTS_DIR, "gt_B_fullrecipe")
N_ARCH, N_BOOT, BOOT_SEED = 50, 10_000, 0


def subset():
    return spaces.sample_archs("B", 200)[:N_ARCH]  # sweep order == content-defined order


def train(device: str, data_root: str = "data"):
    data = {s: load_split(data_root, s) for s in ["train", "val", "test"]}
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, cfg in enumerate(subset()):
        aid = spaces.arch_id(cfg)
        out_path = os.path.join(OUT_DIR, f"{aid}_s0.json")
        if os.path.exists(out_path):
            print(f"[{i}] {aid} skip (done)", flush=True)
            continue
        res = train_one(cfg, data, device, seed=0, recipe=RECIPE_FULL)
        res["sweep_index"] = i
        res["recipe_role"] = "9c-recipe robustness check only; not ground truth"
        tmp = out_path + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, out_path)
        print(f"[{i}] {aid} val={res['val_acc']:.4f} test={res['test_acc']:.4f} "
              f"ep={res['epochs_run']} {res['wall_clock_s']:.0f}s", flush=True)


def _boot_spearman(x, y):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    b = np.array([st.spearmanr(x[i], y[i]).statistic for i in idx])
    return {"spearman": float(st.spearmanr(x, y).statistic),
            "ci95": [float(v) for v in np.nanpercentile(b, [2.5, 97.5])], "n": int(len(x))}


def analyze():
    out_path = os.path.join(RESULTS_DIR, "recipe_check.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists: computed once; delete deliberately to recompute")
    ids = [spaces.arch_id(c) for c in subset()]
    full, gt = {}, {}
    for f in os.listdir(OUT_DIR):
        r = json.load(open(os.path.join(OUT_DIR, f)))
        full[r["arch_id"]] = r
    for f in os.listdir(os.path.join(RESULTS_DIR, "gt_B")):
        r = json.load(open(os.path.join(RESULTS_DIR, "gt_B", f)))
        gt.setdefault(r["arch_id"], {})[r["seed"]] = r
    ids = [a for a in ids if a in full and all(s in gt[a] for s in (0, 1, 2))]
    acc_full = np.array([full[a]["test_acc"] for a in ids])
    acc = {s: np.array([gt[a][s]["test_acc"] for a in ids]) for s in (0, 1, 2)}
    flops = np.array([gt[a][0]["flops"] for a in ids]); params = np.array([gt[a][0]["params"] for a in ids])
    nwot = json.load(open(os.path.join(RESULTS_DIR, "proxy_scores_B.json")))["nwot"]
    nwot = np.array([nwot[a] for a in ids])
    res = {
        "n": len(ids), "recipe_full": RECIPE_FULL,
        "full_vs_pilot_seed0": _boot_spearman(acc_full, acc[0]),
        "full_vs_pilot_seed1": _boot_spearman(acc_full, acc[1]),
        "full_vs_pilot_seed2": _boot_spearman(acc_full, acc[2]),
        "pilot_test_retest_same50": {"mean_pairwise": float(np.mean([st.spearmanr(acc[i], acc[j]).statistic
                                                                     for i, j in [(0, 1), (0, 2), (1, 2)]]))},
        "rankers_vs_full": {"flops": _boot_spearman(flops, acc_full), "params": _boot_spearman(params, acc_full),
                            "nwot": _boot_spearman(nwot, acc_full)},
        "rankers_vs_pilot_seed0_same50": {"flops": float(st.spearmanr(flops, acc[0]).statistic),
                                          "params": float(st.spearmanr(params, acc[0]).statistic),
                                          "nwot": float(st.spearmanr(nwot, acc[0]).statistic)},
        "accuracy": {"full_mean": float(acc_full.mean()), "pilot_seed0_mean": float(acc[0].mean()),
                     "full_minus_pilot_mean_pts": float((acc_full - acc[0]).mean() * 100),
                     "full_range": [float(acc_full.min()), float(acc_full.max())]},
        "epochs_full_median": float(np.median([full[a]["epochs_run"] for a in ids])),
        "wall_clock_full_mean_s": float(np.mean([full[a]["wall_clock_s"] for a in ids])),
        "meta": {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "order": "Space-B sweep index 0-49",
                 "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "timestamp": datetime.now(timezone.utc).isoformat()},
    }
    json.dump(res, open(out_path, "w"), indent=1)
    f0 = res["full_vs_pilot_seed0"]
    print(f"n={res['n']}  full-v1 vs pilot-v2 seed0: rho={f0['spearman']:.3f} {f0['ci95']}  "
          f"(seed1 {res['full_vs_pilot_seed1']['spearman']:.3f}, seed2 {res['full_vs_pilot_seed2']['spearman']:.3f}; "
          f"pilot test-retest on these 50 = {res['pilot_test_retest_same50']['mean_pairwise']:.3f})")
    print(f"accuracy: full mean {res['accuracy']['full_mean']*100:.1f}% vs pilot {res['accuracy']['pilot_seed0_mean']*100:.1f}% "
          f"(+{res['accuracy']['full_minus_pilot_mean_pts']:.1f} pts); median epochs {res['epochs_full_median']:.0f}; "
          f"{res['wall_clock_full_mean_s']:.0f} s/model")
    for k, v in res["rankers_vs_full"].items():
        print(f"  {k:7s} vs full-v1 acc: {v['spearman']:.3f} {v['ci95']}   (vs pilot seed0 on same 50: {res['rankers_vs_pilot_seed0_same50'][k]:.3f})")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    a = ap.parse_args()
    train(a.device) if a.train else analyze()
