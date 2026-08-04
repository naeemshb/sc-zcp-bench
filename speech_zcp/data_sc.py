"""Speech Commands v2, 12-class protocol, cached 40-mel log-spectrograms.

Protocol (frozen; report exactly in the paper, PROTOCOL.md section 8):
- Splits: the official validation_list.txt / testing_list.txt partition,
  as exposed by torchaudio's SPEECHCOMMANDS subset argument.
- 12 classes: yes no up down left right on off stop go + "unknown"
  (all other words, subsampled to ~10% of each split) + "silence"
  (1 s crops of _background_noise_, ~10% of each split, seeded).
- Features: 40-mel log-spectrogram, 16 kHz, 25 ms window (n_fft=400),
  10 ms hop (160), log(mel + 1e-6), per-clip zero-pad/truncate to 1 s.
  Cached once to data/features_sc2_12/{train,val,test}.pt.

Usage: .venv/bin/python -m speech_zcp.data_sc --root data
"""

import argparse
import os
import random

import torch
import torchaudio

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = TARGET_WORDS + ["unknown", "silence"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
SAMPLE_RATE = 16000
UNKNOWN_FRAC = 0.10
SILENCE_FRAC = 0.10
FEATURE_SEED = 0

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160, n_mels=40
)


def wav_to_feature(wave: torch.Tensor) -> torch.Tensor:
    """(1, n_samples) -> (1, 40, 101) log-mel."""
    if wave.shape[-1] < SAMPLE_RATE:
        wave = torch.nn.functional.pad(wave, (0, SAMPLE_RATE - wave.shape[-1]))
    wave = wave[..., :SAMPLE_RATE]
    return torch.log(_MEL(wave) + 1e-6)


def _silence_crops(noise_waves: list[torch.Tensor], n: int, seed: int) -> list[torch.Tensor]:
    rng = random.Random(seed)
    crops = []
    for _ in range(n):
        w = noise_waves[rng.randrange(len(noise_waves))]
        start = rng.randrange(max(1, w.shape[-1] - SAMPLE_RATE))
        crops.append(w[..., start : start + SAMPLE_RATE])
    return crops


def build_split(root: str, subset: str, out_path: str) -> None:
    ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
    rng = random.Random(f"{subset}-{FEATURE_SEED}")

    feats, labels = [], []
    unknown_pool = []
    for i in range(len(ds)):
        wave, sr, label, *_ = ds[i]
        assert sr == SAMPLE_RATE
        if label in LABEL_TO_IDX:
            feats.append(wav_to_feature(wave))
            labels.append(LABEL_TO_IDX[label])
        else:
            unknown_pool.append(i)

    n_target = len(feats)
    n_unknown = max(1, int(UNKNOWN_FRAC * n_target))
    for i in rng.sample(unknown_pool, min(n_unknown, len(unknown_pool))):
        wave, _, _, *_ = ds[i]
        feats.append(wav_to_feature(wave))
        labels.append(LABEL_TO_IDX["unknown"])

    noise_dir = os.path.join(root, "SpeechCommands", "speech_commands_v0.02", "_background_noise_")
    noise_waves = [
        torchaudio.load(os.path.join(noise_dir, f))[0]
        for f in sorted(os.listdir(noise_dir))
        if f.endswith(".wav")
    ]
    n_silence = max(1, int(SILENCE_FRAC * n_target))
    for crop in _silence_crops(noise_waves, n_silence, seed=hash(f"{subset}-{FEATURE_SEED}") % 2**31):
        feats.append(wav_to_feature(crop))
        labels.append(LABEL_TO_IDX["silence"])

    x = torch.stack(feats)  # (N, 1, 40, 101)
    y = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(FEATURE_SEED))
    torch.save({"x": x[perm], "y": y[perm], "labels": LABELS}, out_path)
    print(f"{subset}: {len(y)} clips -> {out_path} ({x.element_size() * x.nelement() / 1e6:.0f} MB)")


def load_split(root: str, subset: str) -> dict:
    path = os.path.join(root, "features_sc2_12", f"{subset}.pt")
    return torch.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    out_dir = os.path.join(args.root, "features_sc2_12")
    os.makedirs(out_dir, exist_ok=True)
    for subset, name in [("training", "train"), ("validation", "val"), ("testing", "test")]:
        out = os.path.join(out_dir, f"{name}.pt")
        if os.path.exists(out):
            print(f"skip {name} (exists)")
            continue
        build_split(args.root, subset, out)


if __name__ == "__main__":
    main()
