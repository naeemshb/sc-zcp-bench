"""Speech Commands v2, 12-class protocol, cached 40-mel log-spectrograms.

Protocol (frozen; report exactly in the paper, PROTOCOL.md section 8):
- Splits: the official validation_list.txt / testing_list.txt partition;
  training = all remaining files (Warden 2018 protocol).
- 12 classes: yes no up down left right on off stop go + "unknown"
  (all other words, subsampled to ~10% of each split, seeded) + "silence"
  (1 s crops of _background_noise_, ~10% of each split, seeded).
- Features: 40-mel log-spectrogram, 16 kHz, 25 ms window (n_fft=400),
  10 ms hop (160), log(mel + 1e-6), per-clip zero-pad/truncate to 1 s.
  Cached once to data/features_sc2_12/{train,val,test}.pt.

WAVs are decoded with the stdlib wave module (SC v2 is 16-bit PCM mono
16 kHz throughout) — torchaudio is used only for the MelSpectrogram tensor
op, so no torchcodec/FFmpeg dependency.

Usage: .venv/bin/python -m speech_zcp.data_sc --root data
"""

import argparse
import os
import random
import wave

import numpy as np
import torch
import torchaudio

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = TARGET_WORDS + ["unknown", "silence"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
SAMPLE_RATE = 16000
UNKNOWN_FRAC = 0.10
SILENCE_FRAC = 0.10
FEATURE_SEED = 0

DATASET_SUBDIR = os.path.join("SpeechCommands", "speech_commands_v0.02")

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160, n_mels=40
)


def load_wav(path: str) -> torch.Tensor:
    """16-bit PCM mono wav -> (1, n_samples) float tensor in [-1, 1]."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, path
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(x).unsqueeze(0)


def wav_to_feature(wave_t: torch.Tensor) -> torch.Tensor:
    """(1, n_samples) -> (1, 40, 101) log-mel."""
    if wave_t.shape[-1] < SAMPLE_RATE:
        wave_t = torch.nn.functional.pad(wave_t, (0, SAMPLE_RATE - wave_t.shape[-1]))
    wave_t = wave_t[..., :SAMPLE_RATE]
    return torch.log(_MEL(wave_t) + 1e-6)


def official_split_lists(ds_dir: str) -> dict[str, set[str]]:
    out = {}
    for name, fname in [("val", "validation_list.txt"), ("test", "testing_list.txt")]:
        with open(os.path.join(ds_dir, fname)) as f:
            out[name] = {line.strip() for line in f if line.strip()}
    return out


def collect_files(ds_dir: str) -> dict[str, list[str]]:
    """split -> list of 'word/file.wav' relative paths."""
    held_out = official_split_lists(ds_dir)
    splits = {"train": [], "val": [], "test": []}
    for word in sorted(os.listdir(ds_dir)):
        word_dir = os.path.join(ds_dir, word)
        if not os.path.isdir(word_dir) or word == "_background_noise_":
            continue
        for f in sorted(os.listdir(word_dir)):
            if not f.endswith(".wav"):
                continue
            rel = f"{word}/{f}"
            if rel in held_out["test"]:
                splits["test"].append(rel)
            elif rel in held_out["val"]:
                splits["val"].append(rel)
            else:
                splits["train"].append(rel)
    return splits


def _silence_crops(noise_waves: list[torch.Tensor], n: int, seed: int) -> list[torch.Tensor]:
    rng = random.Random(seed)
    crops = []
    for _ in range(n):
        w = noise_waves[rng.randrange(len(noise_waves))]
        start = rng.randrange(max(1, w.shape[-1] - SAMPLE_RATE))
        crops.append(w[..., start : start + SAMPLE_RATE])
    return crops


def build_split(root: str, split: str, rel_files: list[str], out_path: str) -> None:
    ds_dir = os.path.join(root, DATASET_SUBDIR)
    rng = random.Random(f"{split}-{FEATURE_SEED}")

    feats, labels, unknown_pool = [], [], []
    for rel in rel_files:
        word = rel.split("/")[0]
        if word in LABEL_TO_IDX:
            feats.append(wav_to_feature(load_wav(os.path.join(ds_dir, rel))))
            labels.append(LABEL_TO_IDX[word])
        else:
            unknown_pool.append(rel)

    n_target = len(feats)
    n_unknown = max(1, int(UNKNOWN_FRAC * n_target))
    for rel in rng.sample(unknown_pool, min(n_unknown, len(unknown_pool))):
        feats.append(wav_to_feature(load_wav(os.path.join(ds_dir, rel))))
        labels.append(LABEL_TO_IDX["unknown"])

    noise_dir = os.path.join(ds_dir, "_background_noise_")
    noise_waves = [
        load_wav(os.path.join(noise_dir, f))
        for f in sorted(os.listdir(noise_dir))
        if f.endswith(".wav")
    ]
    n_silence = max(1, int(SILENCE_FRAC * n_target))
    silence_seed = int.from_bytes(f"{split}-{FEATURE_SEED}".encode(), "little") % 2**31
    for crop in _silence_crops(noise_waves, n_silence, seed=silence_seed):
        feats.append(wav_to_feature(crop))
        labels.append(LABEL_TO_IDX["silence"])

    x = torch.stack(feats)  # (N, 1, 40, 101)
    y = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(FEATURE_SEED))
    torch.save({"x": x[perm], "y": y[perm], "labels": LABELS}, out_path)
    print(
        f"{split}: {len(y)} clips ({n_target} target, {min(n_unknown, len(unknown_pool))} unknown, "
        f"{n_silence} silence) -> {out_path} ({x.element_size() * x.nelement() / 1e6:.0f} MB)"
    )


def load_split(root: str, split: str) -> dict:
    return torch.load(os.path.join(root, "features_sc2_12", f"{split}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    ds_dir = os.path.join(args.root, DATASET_SUBDIR)
    assert os.path.isdir(ds_dir), f"dataset not found at {ds_dir} (download step missing)"
    out_dir = os.path.join(args.root, "features_sc2_12")
    os.makedirs(out_dir, exist_ok=True)
    splits = collect_files(ds_dir)
    for split in ["train", "val", "test"]:
        out = os.path.join(out_dir, f"{split}.pt")
        if os.path.exists(out):
            print(f"skip {split} (exists)")
            continue
        build_split(args.root, split, splits[split], out)


if __name__ == "__main__":
    main()
