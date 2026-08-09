"""Statistics caches for GP evolution (PROTOCOL.md section 5).

FROZEN extraction protocol (cache-v1): one forward + one backward (CE loss)
per architecture on a fixed minibatch (batch 8, seed 0), CPU, train mode,
init_seed 0. Cached per layer:
  W_i / G_i   conv+linear weights and their gradients
  A_i         ReLU activation outputs
Speech spaces additionally (the interpretability bet — terminals that
vision-evolved proxies cannot exploit):
  AstdT_i         std of A_i across the time axis (last dim)
  An_i / Gn_i / AnstdT_i   same statistics on a matched noise minibatch
                           (gaussian, per-mel-bin mean/std matched, seed 0)
NB201 caches skip the speech-specific extras (no meaningful time axis).

One compressed .npz per arch under cache/<name>/; index json maps arch_id ->
ground-truth accuracy (test_acc for speech, NB-Suite-Zero val_accuracy for
NB201) so evolution reads fitness targets from the same place.

Usage:
  .venv/bin/python -m speech_zcp.cache_stats --target A
  .venv/bin/python -m speech_zcp.cache_stats --target B
  .venv/bin/python -m speech_zcp.cache_stats --target nb201
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import spaces
from .data_sc import load_split

PROTOCOL = {
    "batch_size": 8,
    "seed": 0,
    "device": "cpu",
    "loss": "cross_entropy",
    "init_seed": 0,
    "version": "cache-v1",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def fixed_speech_batch(root="data"):
    d = load_split(root, "train")  # deterministically shuffled at build time (seed 0)
    return d["x"][: PROTOCOL["batch_size"]].clone(), d["y"][: PROTOCOL["batch_size"]].clone()


def matched_noise_batch(x: torch.Tensor) -> torch.Tensor:
    """Gaussian noise matched to the real batch's per-mel-bin statistics."""
    g = torch.Generator().manual_seed(PROTOCOL["seed"])
    mu = x.mean(dim=(0, 3), keepdim=True)
    sd = x.std(dim=(0, 3), keepdim=True)
    return mu + sd * torch.randn(x.shape, generator=g)


def _forward_backward_stats(model: nn.Module, x, y):
    """-> (weights, grads, acts) as lists of numpy arrays, layer order fixed."""
    model.train()
    model.zero_grad()
    acts = []

    def hook(mod, inp, out):
        acts.append(out.detach().numpy().astype(np.float32))

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.ReLU)]
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    for h in handles:
        h.remove()
    ws, gs = [], []
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            ws.append(m.weight.detach().numpy().astype(np.float32))
            g = m.weight.grad
            gs.append((g if g is not None else torch.zeros_like(m.weight)).numpy().astype(np.float32))
    return ws, gs, acts


def _std_time(acts):
    """std across the last (time) axis; keeps (B, C, ...) leading dims."""
    return [a.std(axis=-1) if a.ndim >= 3 else a for a in acts]


def extract_speech(cfg: dict, x, y, xn) -> dict:
    model = spaces.build_model(cfg, init_seed=PROTOCOL["init_seed"])
    ws, gs, acts = _forward_backward_stats(model, x, y)
    model_n = spaces.build_model(cfg, init_seed=PROTOCOL["init_seed"])
    _, gns, actns = _forward_backward_stats(model_n, xn, y)
    arrs = {}
    for i, a in enumerate(ws):
        arrs[f"W_{i}"] = a
    for i, a in enumerate(gs):
        arrs[f"G_{i}"] = a
    for i, a in enumerate(acts):
        arrs[f"A_{i}"] = a
    for i, a in enumerate(gns):
        arrs[f"Gn_{i}"] = a
    for i, a in enumerate(actns):
        arrs[f"An_{i}"] = a
    for i, a in enumerate(_std_time(acts)):
        arrs[f"AstdT_{i}"] = a
    for i, a in enumerate(_std_time(actns)):
        arrs[f"AnstdT_{i}"] = a
    return arrs


def extract_nb201(arch_str: str, x, y) -> dict:
    from .bench_201 import build_nb201

    model = build_nb201(arch_str, init_seed=PROTOCOL["init_seed"])
    ws, gs, acts = _forward_backward_stats(model, x, y)
    arrs = {}
    for i, a in enumerate(ws):
        arrs[f"W_{i}"] = a
    for i, a in enumerate(gs):
        arrs[f"G_{i}"] = a
    for i, a in enumerate(acts):
        arrs[f"A_{i}"] = a
    return arrs


def build_speech(space: str):
    out_dir = os.path.join(CACHE_DIR, space)
    os.makedirs(out_dir, exist_ok=True)
    x, y = fixed_speech_batch()
    xn = matched_noise_batch(x)
    accs = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, f"gt_{space}", "*_s0.json")):
        r = json.load(open(f))
        accs[r["arch_id"]] = r["test_acc"]
    n_arch = 250 if space == "A" else 200
    archs = spaces.sample_archs(space, n_arch)
    for i, cfg in enumerate(archs):
        aid = spaces.arch_id(cfg)
        path = os.path.join(out_dir, f"{aid}.npz")
        if not os.path.exists(path):
            np.savez_compressed(path, **extract_speech(cfg, x, y, xn))
        if (i + 1) % 25 == 0:
            print(f"{space}: {i + 1}/{len(archs)}", flush=True)
    index = {spaces.arch_id(c): {"acc": accs[spaces.arch_id(c)], "i": i}
             for i, c in enumerate(archs)}
    json.dump({"protocol": PROTOCOL, "archs": index},
              open(os.path.join(CACHE_DIR, f"{space}_index.json"), "w"))
    print(f"{space}: done, {len(index)} archs indexed")


def build_nb201_cache(n_sample: int = 500):
    from .bench_201 import fixed_cifar_batch, op_indices_key_to_arch_str

    out_dir = os.path.join(CACHE_DIR, "nb201")
    os.makedirs(out_dir, exist_ok=True)
    zc = json.load(open("data/nbs_zero/zc_nasbench201.json"))["cifar10"]
    keys = sorted(zc.keys())
    sample = random.Random(0).sample(keys, n_sample)  # same sample as the gate
    xc, yc = fixed_cifar_batch()
    x, y = xc[: PROTOCOL["batch_size"]], yc[: PROTOCOL["batch_size"]]
    index = {}
    for i, key in enumerate(sample):
        arch = op_indices_key_to_arch_str(key)
        aid = f"nb201_{zc[key]['id']}"
        path = os.path.join(out_dir, f"{aid}.npz")
        if not os.path.exists(path):
            np.savez_compressed(path, **extract_nb201(arch, x, y))
        index[aid] = {"acc": zc[key]["val_accuracy"], "arch_str": arch, "i": i}
        if (i + 1) % 25 == 0:
            print(f"nb201: {i + 1}/{len(sample)}", flush=True)
    json.dump({"protocol": PROTOCOL, "archs": index},
              open(os.path.join(CACHE_DIR, "nb201_index.json"), "w"))
    print(f"nb201: done, {len(index)} archs indexed")


def load_arch(cache_name: str, arch_id: str) -> dict:
    with np.load(os.path.join(CACHE_DIR, cache_name, f"{arch_id}.npz")) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["A", "B", "nb201"])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.target == "nb201":
        build_nb201_cache()
    else:
        build_speech(args.target)


if __name__ == "__main__":
    main()
