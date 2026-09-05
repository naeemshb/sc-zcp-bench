"""Frozen ground-truth training recipe + sweep runner (PROTOCOL.md section 3).

RECIPE below is FROZEN AFTER THE PILOT. Any change invalidates and restarts
the affected sweep (hard rule 6). Every trained model writes
results/gt_<space>/<arch_id>_s<seed>.json containing config + recipe +
git hash + timestamp + metrics (hard rule 1).

Pilot:      .venv/bin/python -m speech_zcp.train_gt --pilot --device cpu
Full sweep: .venv/bin/python -m speech_zcp.train_gt --space A --count 250 --range 0:125
            (split ranges across machines; arch lists are deterministic)
Noise subset (extra seeds on first 50 Space-B archs):
            .venv/bin/python -m speech_zcp.train_gt --space B --count 200 --range 0:50 --seed 1
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from . import spaces
from .data_sc import load_split
from .proxies import flops_count, params_count

RECIPE = {
    "optimizer": "adam",
    "lr": 3e-3,
    "lr_schedule": "cosine",  # annealed over max_epochs, per-epoch step
    "lr_min": 1e-5,
    "weight_decay": 0.0,
    "batch_size": 128,
    "max_epochs": 15,
    "early_stop_patience": 3,  # epochs without val-acc improvement
    # Stratified train subsample (seed 0). Full 36.9k x 20 epochs measured
    # 56 min/model on M3 CPU (pilot-v1 gate failure); 10k x 15 epochs on MPS
    # puts the largest Space-A models at ~2 min. Reduced data also widens
    # accuracy spread, which the benchmark needs. Val/test stay full-size.
    "train_subsample": 10000,
    "loss": "cross_entropy",
    "recipe_version": "pilot-v2",  # bump ONLY if the pilot forces a change
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# 9c-recipe robustness check ONLY (PROTOCOL.md 9c): never a ground-truth recipe.
RECIPE_FULL = {**RECIPE, "train_subsample": None, "max_epochs": 30, "early_stop_patience": 5,
               "recipe_version": "full-v1"}


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


def _stratified_subsample(x, y, n: int, seed: int = 0):
    """Deterministic per-class proportional subsample of the training split."""
    if n is None or n >= len(y):
        return x, y
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in y.unique(sorted=True):
        idx = (y == c).nonzero(as_tuple=True)[0]
        n_c = max(1, round(n * len(idx) / len(y)))
        keep.append(idx[torch.randperm(len(idx), generator=g)[:n_c]])
    keep = torch.cat(keep)
    return x[keep], y[keep]


def _epoch_iter(x, y, batch_size, generator):
    perm = torch.randperm(len(y), generator=generator)
    for i in range(0, len(y), batch_size):
        idx = perm[i : i + batch_size]
        yield x[idx], y[idx]


@torch.no_grad()
def _accuracy(model, x, y, device, batch_size=512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(y), batch_size):
        xb = x[i : i + batch_size].to(device)
        pred = model(xb).argmax(1).cpu()
        correct += int((pred == y[i : i + batch_size]).sum())
    return correct / len(y)


def train_one(cfg: dict, data: dict, device: str, seed: int = 0, recipe: dict = RECIPE) -> dict:
    """recipe defaults to the FROZEN pilot-v2 RECIPE; only 9c-recipe passes RECIPE_FULL."""
    torch.manual_seed(seed)
    model = spaces.build_model(cfg, init_seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=recipe["max_epochs"], eta_min=recipe["lr_min"]
    )
    gen = torch.Generator().manual_seed(seed)
    xtr, ytr = _stratified_subsample(
        data["train"]["x"], data["train"]["y"], recipe["train_subsample"]
    )

    best_val, best_state, best_epoch, bad_epochs = -1.0, None, -1, 0
    t0 = time.time()
    for epoch in range(recipe["max_epochs"]):
        model.train()
        for xb, yb in _epoch_iter(xtr, ytr, recipe["batch_size"], gen):
            opt.zero_grad()
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        sched.step()
        val_acc = _accuracy(model, data["val"]["x"], data["val"]["y"], device)
        if val_acc > best_val:
            best_val, best_epoch, bad_epochs = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= recipe["early_stop_patience"]:
                break
    wall = time.time() - t0

    model.load_state_dict(best_state)
    test_acc = _accuracy(model, data["test"]["x"], data["test"]["y"], device)
    cpu_model = spaces.build_model(cfg, init_seed=seed)
    dummy = torch.zeros(1, 1, spaces.N_MELS, spaces.N_FRAMES)
    return {
        "arch_id": spaces.arch_id(cfg),
        "config": cfg,
        "seed": seed,
        "val_acc": best_val,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "params": params_count(cpu_model),
        "flops": flops_count(cpu_model, dummy),
        "wall_clock_s": wall,
        "device": device,
        "recipe": recipe,
        "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": os.uname().nodename,
    }


def run_sweep(space, count, lo, hi, device, data_root, seed):
    data = {name: load_split(data_root, name) for name in ["train", "val", "test"]}
    archs = spaces.sample_archs(space, count)
    out_dir = os.path.join(RESULTS_DIR, f"gt_{space}")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(lo, hi):
        cfg = archs[i]
        aid = spaces.arch_id(cfg)
        out_path = os.path.join(out_dir, f"{aid}_s{seed}.json")
        if os.path.exists(out_path):
            print(f"[{i}] {aid} skip (done)")
            continue
        res = train_one(cfg, data, device, seed=seed)
        res["sweep_index"] = i
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        print(
            f"[{i}] {aid} val={res['val_acc']:.4f} test={res['test_acc']:.4f} "
            f"params={res['params']:.0f} {res['wall_clock_s']:.0f}s"
        )


def run_pilot(device, data_root, n_per_space: int = 20):
    """Pilot archs per space, full frozen recipe. Gate (section 3): wall-clock
    <= 5 min/model, spread >= 15 points, no divergence. Also checks
    params-accuracy correlation headroom (n raised 10->20/space for a usable
    Spearman estimate; +-0.15 CI at n=20 was too wide to gate on)."""
    data = {name: load_split(data_root, name) for name in ["train", "val", "test"]}
    results = []
    for space in ["A", "B"]:
        out_dir = os.path.join(RESULTS_DIR, "pilot")
        os.makedirs(out_dir, exist_ok=True)
        for i, cfg in enumerate(spaces.sample_archs(space, n_per_space)):
            aid = spaces.arch_id(cfg)
            out_path = os.path.join(out_dir, f"{space}_{aid}.json")
            if os.path.exists(out_path):
                results.append(json.load(open(out_path)))
                print(f"pilot {space}[{i}] {aid} skip (done)")
                continue
            res = train_one(cfg, data, device, seed=0)
            with open(out_path, "w") as f:
                json.dump(res, f, indent=1)
            results.append(res)
            print(
                f"pilot {space}[{i}] {aid} test={res['test_acc']:.4f} "
                f"{res['wall_clock_s']:.0f}s"
            )
    import scipy.stats as st

    accs = [r["test_acc"] for r in results]
    times = [r["wall_clock_s"] for r in results]
    spread = (max(accs) - min(accs)) * 100
    rho_params = st.spearmanr(accs, [r["params"] for r in results]).statistic
    print("\n=== PILOT GATE ===")
    print(f"models: {len(results)}  max wall-clock: {max(times):.0f}s (gate: <=300s)")
    print(f"accuracy spread: {spread:.1f} points (gate: >=15)")
    print(f"params-accuracy Spearman: {rho_params:.3f} (headroom check: want < ~0.8)")
    print(f"acc range: [{min(accs):.4f}, {max(accs):.4f}]")
    func = [r for r in results if r["test_acc"] > 0.5]
    if len(func) > 3:
        fa = [r["test_acc"] for r in func]
        print(
            f"functional-only (acc>0.5, n={len(func)}): spread "
            f"{(max(fa) - min(fa)) * 100:.1f} pts, params rho "
            f"{st.spearmanr(fa, [r['params'] for r in func]).statistic:.3f}"
        )
    for space in ["A", "B"]:
        sp = [r for r in results if r["config"]["space"] == space]
        rho = st.spearmanr(
            [r["test_acc"] for r in sp], [r["params"] for r in sp]
        ).statistic
        print(f"within-{space} (n={len(sp)}): params rho {rho:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["A", "B"])
    ap.add_argument("--count", type=int)
    ap.add_argument("--range", default=None, help="lo:hi sweep indices for this machine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(2, os.cpu_count() - 2))
    if args.pilot:
        run_pilot(args.device, args.data_root)
    else:
        lo, hi = (0, args.count) if args.range is None else map(int, args.range.split(":"))
        run_sweep(args.space, args.count, lo, hi, args.device, args.data_root, args.seed)


if __name__ == "__main__":
    main()
