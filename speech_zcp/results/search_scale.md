# 9d-search-scale: N=200 cold evolution, committed (30x15) vs big search (100x30) — fitting side only

| | committed 30x15 | big search 100x30 |
|---|---|---|
| weight-only size proxies (of 10) | 7 | 7 |
| elites using speech terminals (of 10) | 3 | 3 |
| distinct formulas | 5 | 4 |
| median generation found | 3.0 | 3.0 |
| modal formula | `asum(rl1(W))` ×6 | `asum(rl1(W))` ×5 |
| fit-ρ (first 80% of chain), mean ± sd | 0.808 ± 0.059 | 0.832 ± 0.015 |
| selection fitness (held-back 20%), mean | 0.822 | 0.827 |

## Big-search elites per seed

| seed | formula | terminals | fit-ρ | sel | gen |
|---|---|---|---|---|---|
| 1 | `asum(rl1(W))` | W | 0.807 | 0.895 | 2 |
| 2 | `asum(rl2(log(W)))` | W | 0.830 | 0.875 | 16 |
| 3 | `asum(rl2(sign(Gn)))` | Gn | 0.850 | 0.760 | 2 |
| 4 | `asum(rl1(W))` | W | 0.832 | 0.843 | 7 |
| 5 | `asum(rl1(W))` | W | 0.845 | 0.695 | 3 |
| 6 | `asum(rl2(log(W)))` | W | 0.842 | 0.826 | 3 |
| 7 | `asum(rl1(W))` | W | 0.812 | 0.882 | 0 |
| 8 | `asum(rl2(sign(Gn)))` | Gn | 0.849 | 0.819 | 5 |
| 9 | `asum(rl1(W))` | W | 0.838 | 0.793 | 3 |
| 10 | `asum(rl2(relu(sign(Gn))))` | Gn | 0.817 | 0.880 | 1 |

**Pinned reading:** (a) size attractor is search-scale-robust: >=6/10 big-search elites are weight-only size proxies

Fit-side ρ is not comparable with Space-B numbers; no sealed set was scored.
