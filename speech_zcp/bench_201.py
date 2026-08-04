"""NAS-Bench-201 models + the NB-Suite-Zero validation gate (PROTOCOL.md section 4).

Minimal reimplementation of the NB201 macro/cell (Dong & Yang, ICLR 2020):
stem Conv3x3->16 + BN; 3 stages of N=5 cells (16/32/64 channels) separated by
ResNet basic-block reductions; BN+ReLU, global avg pool, linear head.
Cell: 4 nodes, DAG, ops in {none, skip_connect, nor_conv_1x1, nor_conv_3x3,
avg_pool_3x3}, arch string like
"|nor_conv_3x3~0|+|none~0|skip_connect~1|+|none~0|none~1|nor_conv_3x3~2|".

GATE (run after downloading NB-Suite-Zero data, see download_instructions()):
on a >=500-arch NB201/CIFAR-10 sample, our proxy scores must Spearman-match
the precomputed scores per proxy: rho >= 0.99 data-free, >= 0.90 data-dependent.
NB-Suite-Zero provides scalar scores + accuracies only, NOT raw statistics
tensors (vision-side evolution needs our own cache, section 5).
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn

from .proxies import ALL_PROXIES, DATA_FREE, compute_proxy

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
N_CELLS_PER_STAGE = 5
STAGE_CHANNELS = [16, 32, 64]
CIFAR_CLASSES = 10


def op_module(name: str, c: int) -> nn.Module:
    if name == "none":
        return _Zero()
    if name == "skip_connect":
        return nn.Identity()
    if name == "avg_pool_3x3":
        return nn.AvgPool2d(3, stride=1, padding=1)
    k = 1 if name == "nor_conv_1x1" else 3
    return nn.Sequential(
        nn.ReLU(), nn.Conv2d(c, c, k, padding=k // 2, bias=False), nn.BatchNorm2d(c)
    )


class _Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def parse_arch_str(arch: str) -> list[list[tuple[str, int]]]:
    """-> per-node list of (op, from_node)."""
    nodes = []
    for node_str in arch.split("+"):
        edges = [e for e in node_str.split("|") if e]
        nodes.append([(e.split("~")[0], int(e.split("~")[1])) for e in edges])
    return nodes


class Cell(nn.Module):
    def __init__(self, arch: str, c: int):
        super().__init__()
        self.node_edges = parse_arch_str(arch)
        self.ops = nn.ModuleList(
            [nn.ModuleList([op_module(op, c) for op, _ in edges]) for edges in self.node_edges]
        )

    def forward(self, x):
        feats = [x]
        for edges, ops in zip(self.node_edges, self.ops):
            feats.append(sum(op(feats[src]) for (_, src), op in zip(edges, ops)))
        return feats[-1]


class ResNetBasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.a = nn.Sequential(
            nn.ReLU(), nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.b = nn.Sequential(
            nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        )
        self.down = nn.Sequential(
            nn.AvgPool2d(2, stride=2), nn.Conv2d(cin, cout, 1, bias=False)
        )

    def forward(self, x):
        return self.b(self.a(x)) + self.down(x)


class NB201Net(nn.Module):
    """Input: (B, 3, 32, 32) CIFAR-10."""

    def __init__(self, arch: str, num_classes: int = CIFAR_CLASSES):
        super().__init__()
        c1, c2, c3 = STAGE_CHANNELS
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1))
        layers = []
        for stage, c in enumerate(STAGE_CHANNELS):
            if stage > 0:
                layers.append(ResNetBasicBlock(STAGE_CHANNELS[stage - 1], c))
            layers += [Cell(arch, c) for _ in range(N_CELLS_PER_STAGE)]
        self.body = nn.Sequential(*layers)
        self.lastact = nn.Sequential(nn.BatchNorm2d(c3), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward_features(self, x):
        return self.lastact(self.body(self.stem(x)))

    def forward(self, x):
        return self.classifier(self.pool(self.forward_features(x)).flatten(1))


def build_nb201(arch: str, init_seed: int = 0) -> nn.Module:
    torch.manual_seed(init_seed)
    return NB201Net(arch)


def download_instructions() -> str:
    return (
        "NB-Suite-Zero data (Krishnakumar et al., NeurIPS 2022):\n"
        "  git clone -b zerocost https://github.com/automl/NASLib\n"
        "  bash NASLib/scripts/bash_scripts/download_nbs_zero.sh nb201\n"
        "  -> place zc_nasbench201.json under data/nbs_zero/\n"
        "The json maps arch strings -> per-proxy scores + val accuracy per dataset."
    )


def run_gate(zc_json_path: str, n_sample: int = 500, seed: int = 0, dataset: str = "cifar10"):
    """Spearman(our proxy, NB-Suite-Zero proxy) per proxy on a fixed NB201 sample."""
    import scipy.stats as st

    with open(zc_json_path) as f:
        zc = json.load(f)
    if dataset in zc:
        zc = zc[dataset]
    archs = sorted(zc.keys())
    rng = random.Random(seed)
    sample = rng.sample(archs, min(n_sample, len(archs)))

    g = torch.Generator().manual_seed(0)
    inputs = torch.randn(8, 3, 32, 32, generator=g)  # placeholder: swap for a fixed CIFAR batch
    targets = torch.randint(0, CIFAR_CLASSES, (8,), generator=g)

    theirs = {p: [] for p in ALL_PROXIES}
    ours = {p: [] for p in ALL_PROXIES}
    name_map = {"nwot": "nwot", "params": "params", "flops": "flops"}  # extend per json keys
    for i, arch in enumerate(sample):
        for p in ALL_PROXIES:
            key = name_map.get(p, p)
            if key not in zc[arch]:
                continue
            score = zc[arch][key]["score"] if isinstance(zc[arch][key], dict) else zc[arch][key]
            theirs[p].append(score)
            ours[p].append(compute_proxy(p, lambda: build_nb201(arch), inputs, targets))
        if (i + 1) % 50 == 0:
            print(f"gate: {i + 1}/{len(sample)} archs")

    print("\n=== NB201 VALIDATION GATE ===")
    for p in ALL_PROXIES:
        if not theirs[p]:
            print(f"{p:10s}  (not in NB-Suite-Zero json)")
            continue
        rho = st.spearmanr(theirs[p], ours[p]).statistic
        threshold = 0.99 if p in DATA_FREE else 0.90
        status = "PASS" if rho >= threshold else "FAIL <-- implementation bug until proven otherwise"
        print(f"{p:10s}  rho={rho:.4f}  (gate {threshold})  {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zc-json", default="data/nbs_zero/zc_nasbench201.json")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    if not os.path.exists(args.zc_json):
        print(f"missing {args.zc_json}\n\n{download_instructions()}")
    else:
        run_gate(args.zc_json, args.n)
