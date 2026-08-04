"""Space A (DS-CNN-style) and Space B (TC-ResNet-style) sampling grammars.

Seeded and deterministic; released with the benchmark (PROTOCOL.md section 3).
All sampling ranges below are PILOT-TUNABLE: if the pilot's accuracy spread is
< 15 percentage points, widen ranges (or move to 35-class) and re-pilot.
Any range change after the pilot gate invalidates the affected sweep.

Space A follows Zhang et al., "Hello Edge" (DS-CNN): 2D depthwise-separable
conv stacks over log-mel spectrograms.
Space B follows Choi et al., Interspeech 2019 (TC-ResNet): temporal-conv
residual stacks treating mel bins as channels of a 1D sequence.
"""

import hashlib
import json
import random

import torch
import torch.nn as nn

N_CLASSES = 12
N_MELS = 40
N_FRAMES = 101  # 1 s @ 16 kHz, 25 ms window / 10 ms hop, center=True

MASTER_SEED = 2027  # single source of truth for architecture lists

# ---------------------------------------------------------------------------
# Space A: DS-CNN
# ---------------------------------------------------------------------------
# grammar-v2 (2026-08-04): pilot on grammar-v1 gave only 8.1 points accuracy
# spread (gate: >=15). Widened with QUALITY-degrading knobs at similar size
# (kernel 1 = spatially blind depthwise; more stride-2 blocks and strided
# stems = temporal collapse; 1-block nets) plus modest width extension down
# to 8 channels. Deliberately avoids relying on tiny-params cripples alone:
# params-accuracy Spearman was already 0.732 and must stay < ~0.8.
A_CHANNELS = [8, 16, 24, 32, 48, 64, 96, 128, 172]
A_KERNELS = [1, 3, 5, 7]
A_POOLS = ["avg", "max"]
A_BLOCK_RANGE = (1, 6)
A_MAX_STRIDE2_BLOCKS = 4  # stride-2 blocks allowed beyond the stem
A_STEM_STRIDES = [[2, 2], [2, 1], [4, 2]]


def sample_space_a(rng: random.Random) -> dict:
    n_blocks = rng.randint(*A_BLOCK_RANGE)
    n_s2 = rng.randint(0, min(A_MAX_STRIDE2_BLOCKS, n_blocks))
    stride2_at = set(rng.sample(range(n_blocks), n_s2))
    return {
        "space": "A",
        "stem_channels": rng.choice(A_CHANNELS[:5]),
        "stem_stride": rng.choice(A_STEM_STRIDES),
        "blocks": [
            {
                "channels": rng.choice(A_CHANNELS),
                "kernel": rng.choice(A_KERNELS),
                "stride": 2 if i in stride2_at else 1,
            }
            for i in range(n_blocks)
        ],
        "pool": rng.choice(A_POOLS),
    }


class DSBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, padding=k // 2, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        return self.act2(self.bn2(self.pw(self.act1(self.bn1(self.dw(x))))))


class DSCNN(nn.Module):
    """Input: (B, 1, N_MELS, N_FRAMES)."""

    def __init__(self, cfg: dict):
        super().__init__()
        c0 = cfg["stem_channels"]
        stem_stride = tuple(cfg.get("stem_stride", (2, 2)))
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=(10, 4), stride=stem_stride, padding=(4, 1), bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(),
        )
        blocks, cin = [], c0
        for b in cfg["blocks"]:
            blocks.append(DSBlock(cin, b["channels"], b["kernel"], b["stride"]))
            cin = b["channels"]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1) if cfg["pool"] == "avg" else nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(cin, N_CLASSES)

    def forward_features(self, x):
        return self.blocks(self.stem(x))

    def forward(self, x):
        z = self.pool(self.forward_features(x)).flatten(1)
        return self.classifier(self.dropout(z))


# ---------------------------------------------------------------------------
# Space B: TC-ResNet
# ---------------------------------------------------------------------------
# grammar-v2 (2026-08-04): same spread-widening pass as Space A — kernel 1
# (no temporal context), dilation 4 (receptive-field overshoot), free per-block
# stride (up to full temporal collapse), 1-block nets, width 0.25.
B_BLOCK_RANGE = (1, 6)
B_WIDTHS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
B_KERNELS = [1, 3, 5, 7, 9, 11, 13, 15]
B_DILATIONS = [1, 2, 4]
B_BASE_CHANNELS = [24, 32, 48, 64, 96, 128]  # progression before width multiplier


def sample_space_b(rng: random.Random) -> dict:
    n_blocks = rng.randint(*B_BLOCK_RANGE)
    return {
        "space": "B",
        "width": rng.choice(B_WIDTHS),
        "blocks": [
            {
                "kernel": rng.choice(B_KERNELS),
                "dilation": rng.choice(B_DILATIONS),
                "stride": rng.choice([1, 2]),
            }
            for i in range(n_blocks)
        ],
    }


class TCResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int, dilation: int):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv1 = nn.Conv1d(cin, cout, k, stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, 1, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act1 = nn.ReLU()
        self.act_out = nn.ReLU()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act_out(h + self.shortcut(x))


class TCResNet(nn.Module):
    """Input: (B, 1, N_MELS, N_FRAMES); mel bins become conv1d channels."""

    def __init__(self, cfg: dict):
        super().__init__()
        w = cfg["width"]
        c0 = max(8, round(16 * w))
        self.stem = nn.Sequential(
            nn.Conv1d(N_MELS, c0, 3, padding=1, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(),
        )
        blocks, cin = [], c0
        for i, b in enumerate(cfg["blocks"]):
            cout = max(8, round(B_BASE_CHANNELS[i] * w))
            blocks.append(TCResBlock(cin, cout, b["kernel"], b["stride"], b["dilation"]))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(cin, N_CLASSES)

    def forward_features(self, x):
        if x.dim() == 4:  # (B, 1, mels, T) -> (B, mels, T)
            x = x.squeeze(1)
        return self.blocks(self.stem(x))

    def forward(self, x):
        z = self.pool(self.forward_features(x)).flatten(1)
        return self.classifier(self.dropout(z))


# ---------------------------------------------------------------------------
# Deterministic architecture lists
# ---------------------------------------------------------------------------
def arch_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


# grammar-v2.1: per-model compute cap. The uncapped grammar produced a 112M-MAC
# outlier that alone breached the pilot wall-clock gate (313s); capping bounds
# the sweep's worst case at ~1-2 min/model without narrowing the quality knobs.
FLOPS_CAP = 60e6


def _model_flops(cfg: dict) -> float:
    from .proxies import flops_count  # local import; proxies does not import spaces

    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)
    return flops_count(build_model(cfg), dummy)


def sample_archs(space: str, n: int, master_seed: int = MASTER_SEED) -> list[dict]:
    """Deterministic, deduplicated, FLOPs-capped list of n configs.

    Identical on every machine (fixed rng stream + deterministic rejection)."""
    rng = random.Random(f"{space}-{master_seed}")
    sampler = sample_space_a if space == "A" else sample_space_b
    seen, out = set(), []
    while len(out) < n:
        cfg = sampler(rng)
        aid = arch_id(cfg)
        if aid in seen:
            continue
        seen.add(aid)
        if _model_flops(cfg) > FLOPS_CAP:
            continue
        out.append(cfg)
    return out


def build_model(cfg: dict, init_seed: int = 0) -> nn.Module:
    """Fixed init seed: required for data-free proxy reproducibility (section 4)."""
    torch.manual_seed(init_seed)
    return DSCNN(cfg) if cfg["space"] == "A" else TCResNet(cfg)
