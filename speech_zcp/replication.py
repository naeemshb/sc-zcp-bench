"""Pre-registered replication sets (PROTOCOL.md 9b, amended 2026-08-11).

Everything here was pinned before the first architecture was sampled:
- Seed streams: random.Random("Atest-expansion-2027") -> 150 new Space-A archs;
  random.Random("B-replication-2027") -> 200 new Space-B archs.
- Disjointness: arch_id rejection against the released benchmark lists.
- Grammar v2.1 samplers and recipe pilot-v2 UNCHANGED (imported frozen).
- Trained seed 0; statistics caches per section 5; evaluated ONCE by
  evaluate.py extensions after ALL replication training completes; reported
  beside the originals, never merged.

Usage:
  .venv/bin/python -m speech_zcp.replication --sample          # write splits/
  .venv/bin/python -m speech_zcp.replication --train A_rep --device mps
  .venv/bin/python -m speech_zcp.replication --train B_rep --device mps
"""

import argparse
import json
import os
import random

import torch

from . import spaces
from .train_gt import RESULTS_DIR, run_pilot  # noqa: F401  (RESULTS_DIR reuse)
from .train_gt import train_one
from .data_sc import load_split

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")

REP_SPECS = {
    "A_rep": {"space": "A", "n": 150, "stream": "Atest-expansion-2027", "exclude_n": 250},
    "B_rep": {"space": "B", "n": 200, "stream": "B-replication-2027", "exclude_n": 200},
}


def sample_replication(name: str) -> list[dict]:
    spec = REP_SPECS[name]
    released = {spaces.arch_id(c) for c in spaces.sample_archs(spec["space"], spec["exclude_n"])}
    rng = random.Random(spec["stream"])
    sampler = spaces.sample_space_a if spec["space"] == "A" else spaces.sample_space_b
    seen, out = set(released), []
    while len(out) < spec["n"]:
        cfg = sampler(rng)
        aid = spaces.arch_id(cfg)
        if aid in seen:
            continue
        seen.add(aid)
        if spaces._model_flops(cfg) > spaces.FLOPS_CAP:
            continue
        out.append(cfg)
    return out


def write_splits():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    for name in REP_SPECS:
        cfgs = sample_replication(name)
        path = os.path.join(SPLITS_DIR, f"replication_{name}.json")
        json.dump({"spec": REP_SPECS[name], "n": len(cfgs),
                   "arch_ids": [spaces.arch_id(c) for c in cfgs], "configs": cfgs},
                  open(path, "w"), indent=1)
        print(f"{name}: {len(cfgs)} archs -> {path}")


def train_rep(name: str, device: str, data_root: str = "data"):
    split = json.load(open(os.path.join(SPLITS_DIR, f"replication_{name}.json")))
    data = {s: load_split(data_root, s) for s in ["train", "val", "test"]}
    out_dir = os.path.join(RESULTS_DIR, f"gt_{name}")
    os.makedirs(out_dir, exist_ok=True)
    for i, cfg in enumerate(split["configs"]):
        aid = spaces.arch_id(cfg)
        out_path = os.path.join(out_dir, f"{aid}_s0.json")
        if os.path.exists(out_path):
            print(f"[{i}] {aid} skip (done)")
            continue
        res = train_one(cfg, data, device, seed=0)
        res["sweep_index"] = i
        res["replication_set"] = name
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[{i}] {aid} val={res['val_acc']:.4f} test={res['test_acc']:.4f} "
              f"{res['wall_clock_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--train", choices=list(REP_SPECS))
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(2, os.cpu_count() - 2))
    if args.sample:
        write_splits()
    elif args.train:
        train_rep(args.train, args.device)


if __name__ == "__main__":
    main()
