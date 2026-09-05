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
  # 9b-seeds (2026-09-02): ceiling seeds only; ground truth stays seed 0.
  .venv/bin/python -m speech_zcp.replication --train B_rep --seed 1 --device mps
  .venv/bin/python -m speech_zcp.replication --train Atest --seed 1 --device mps  # -> results/gt_A/
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
    # 9b-rep2 (declared 2026-09-03): also disjoint from the first replication sets.
    "A_rep2": {"space": "A", "n": 100, "stream": "Atest-expansion-2027-r2", "exclude_n": 250, "exclude_splits": ["A_rep"]},
    "B_rep2": {"space": "B", "n": 100, "stream": "B-replication-2027-r2", "exclude_n": 200, "exclude_splits": ["B_rep"]},
}


def sample_replication(name: str) -> list[dict]:
    spec = REP_SPECS[name]
    released = {spaces.arch_id(c) for c in spaces.sample_archs(spec["space"], spec["exclude_n"])}
    for prior in spec.get("exclude_splits", []):
        released |= set(json.load(open(os.path.join(SPLITS_DIR, f"replication_{prior}.json")))["arch_ids"])
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
        path = os.path.join(SPLITS_DIR, f"replication_{name}.json")
        if os.path.exists(path):  # pinned splits are never rewritten
            print(f"{name}: exists, skipped")
            continue
        cfgs = sample_replication(name)
        json.dump({"spec": REP_SPECS[name], "n": len(cfgs),
                   "arch_ids": [spaces.arch_id(c) for c in cfgs], "configs": cfgs},
                  open(path, "w"), indent=1)
        print(f"{name}: {len(cfgs)} archs -> {path}")


def _atest_configs() -> list[dict]:
    """The 50 sealed A-test archs (splits/a_pool_test.json), configs from their
    seed-0 ground-truth files, in A-test list order."""
    test_ids = json.load(open(os.path.join(SPLITS_DIR, "a_pool_test.json")))["test"]
    cfgs = {}
    for f in os.listdir(os.path.join(RESULTS_DIR, "gt_A")):
        if f.endswith("_s0.json"):
            r = json.load(open(os.path.join(RESULTS_DIR, "gt_A", f)))
            cfgs[r["arch_id"]] = r["config"]
    return [cfgs[a] for a in test_ids]


def train_rep(name: str, device: str, data_root: str = "data", seed: int = 0):
    """name in REP_SPECS -> results/gt_<name>/; name == "Atest" -> results/gt_A/
    (the A-test archs live in the released Space-A set). seed != 0 is the
    9b-seeds ceiling expansion: extra seeds never replace seed-0 ground truth."""
    if name == "Atest":
        configs, out_dir = _atest_configs(), os.path.join(RESULTS_DIR, "gt_A")
        # the arch's position in the released Space-A sample (== its sweep index)
        sample_pos = {spaces.arch_id(c): i for i, c in enumerate(spaces.sample_archs("A", 250))}
    else:
        split = json.load(open(os.path.join(SPLITS_DIR, f"replication_{name}.json")))
        configs, out_dir = split["configs"], os.path.join(RESULTS_DIR, f"gt_{name}")
    data = {s: load_split(data_root, s) for s in ["train", "val", "test"]}
    os.makedirs(out_dir, exist_ok=True)
    for i, cfg in enumerate(configs):
        aid = spaces.arch_id(cfg)
        out_path = os.path.join(out_dir, f"{aid}_s{seed}.json")
        if os.path.exists(out_path):
            print(f"[{i}] {aid} s{seed} skip (done)")
            continue
        res = train_one(cfg, data, device, seed=seed)
        res["sweep_index"] = sample_pos[aid] if name == "Atest" else i
        res["replication_set"] = name
        if seed != 0:
            res["seed_role"] = "ceiling-only (9b-seeds); ground truth stays seed 0"
        tmp = out_path + ".tmp"  # atomic write: a crash mid-write must not leave a 'done' file
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, out_path)
        print(f"[{i}] {aid} val={res['val_acc']:.4f} test={res['test_acc']:.4f} "
              f"{res['wall_clock_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--train", choices=list(REP_SPECS) + ["Atest"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(2, os.cpu_count() - 2))
    if args.sample:
        write_splits()
    elif args.train:
        train_rep(args.train, args.device, seed=args.seed)


if __name__ == "__main__":
    main()
