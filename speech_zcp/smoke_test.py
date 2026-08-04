"""End-to-end smoke test: grammars build valid models with real size spread,
every proxy returns a finite score on both speech spaces and on NB201.

Run: .venv/bin/python -m speech_zcp.smoke_test
"""

import math

import torch

from . import spaces
from .bench_201 import build_nb201
from .proxies import ALL_PROXIES, compute_all, params_count


def main():
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 1, spaces.N_MELS, spaces.N_FRAMES, generator=g)
    y = torch.randint(0, spaces.N_CLASSES, (8,), generator=g)

    for space in ["A", "B"]:
        archs = spaces.sample_archs(space, 8)
        sizes = []
        for cfg in archs:
            model = spaces.build_model(cfg)
            out = model(x)
            assert out.shape == (8, spaces.N_CLASSES), (space, out.shape)
            sizes.append(params_count(model))
        print(
            f"Space {space}: 8 archs OK, params [{min(sizes):,.0f} .. {max(sizes):,.0f}] "
            f"(ratio {max(sizes) / min(sizes):.1f}x)"
        )
        for cfg in archs[:2]:
            scores = compute_all(lambda: spaces.build_model(cfg), x, y)
            bad = [k for k, v in scores.items() if not math.isfinite(v)]
            assert not bad, f"non-finite proxies on {space}/{spaces.arch_id(cfg)}: {bad}"
            pretty = "  ".join(f"{k}={v:.3g}" for k, v in scores.items())
            print(f"  {spaces.arch_id(cfg)}: {pretty}")

    # determinism: data-free proxies must be bit-identical across calls (section 4)
    cfg = spaces.sample_archs("A", 1)[0]
    from .proxies import compute_proxy

    for p in ["params", "flops", "synflow"]:
        a = compute_proxy(p, lambda: spaces.build_model(cfg), x, y)
        b = compute_proxy(p, lambda: spaces.build_model(cfg), x, y)
        assert a == b, f"{p} not deterministic: {a} vs {b}"
    print("data-free determinism OK")

    arch = "|nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|nor_conv_3x3~2|"
    xc = torch.randn(8, 3, 32, 32, generator=g)
    yc = torch.randint(0, 10, (8,), generator=g)
    scores = compute_all(lambda: build_nb201(arch), xc, yc)
    bad = [k for k, v in scores.items() if not math.isfinite(v)]
    assert not bad, f"non-finite proxies on NB201: {bad}"
    print(f"NB201 cell OK ({params_count(build_nb201(arch)):,.0f} params): "
          + "  ".join(f"{k}={v:.3g}" for k, v in scores.items()))
    print(f"\nALL SMOKE TESTS PASSED ({len(ALL_PROXIES)} proxies)")


if __name__ == "__main__":
    main()
