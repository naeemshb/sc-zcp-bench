# Exploratory N=100 sweep: n100_pop50_gen20_depth3 (population 50, generations 20, max_depth 3, seeds 1-25)

NOT a sealed evaluation. fit = Spearman on the 80 fitted archs; sel = on the 20 selection archs; hold = on the 100 pool archs
outside the seed's chain (within-Space-A pseudo-holdout). 'committed Space B' = the already-sealed value of that exact formula,
if one exists (evaluation.json / n300.json / the identical standard proxy); '?' = never sealed-evaluated.

| seed | formula | class | gen | fit | sel | hold | committed Space B |
|---|---|---|---|---|---|---|---|
| 1 | `asum(rl2(AstdT))` | AstdT/AnstdT | 3 | 0.740 | 0.733 | 0.676 | 0.586 (sealed evolved artifact) |
| 2 | `asum(rl1(W))` | W-only | 4 | 0.770 | 0.836 | 0.855 | 0.741 (sealed evolved artifact) |
| 3 | `asum(rl1(W))` | W-only | 5 | 0.835 | 0.737 | 0.828 | 0.741 (sealed evolved artifact) |
| 4 | `asum(rl1(W))` | W-only | 0 | 0.875 | 0.717 | 0.821 | 0.741 (sealed evolved artifact) |
| 5 | `asum(rl1(W))` | W-only | 0 | 0.859 | 0.800 | 0.806 | 0.741 (sealed evolved artifact) |
| 6 | `amax(rmean(AstdT))` | AstdT/AnstdT | 0 | 0.593 | 0.777 | 0.601 | ? |
| 7 | `amean(rmean(AstdT))` | AstdT/AnstdT | 0 | 0.635 | 0.734 | 0.680 | 0.231 (sealed evolved artifact) |
| 8 | `asum(rl1(W))` | W-only | 4 | 0.873 | 0.729 | 0.809 | 0.741 (sealed evolved artifact) |
| 9 | `asum(rl1(W))` | W-only | 6 | 0.858 | 0.834 | 0.793 | 0.741 (sealed evolved artifact) |
| 10 | `asum(rl1(W))` | W-only | 1 | 0.720 | 0.900 | 0.870 | 0.741 (sealed evolved artifact) |
| 11 | `asum(rl1(AstdT))` | AstdT/AnstdT | 0 | 0.680 | 0.874 | 0.493 | ? |
| 12 | `asum(rl1(W))` | W-only | 0 | 0.821 | 0.869 | 0.818 | 0.741 (sealed evolved artifact) |
| 13 | `asum(rl1(W))` | W-only | 3 | 0.810 | 0.774 | 0.836 | 0.741 (sealed evolved artifact) |
| 14 | `amax(rmean(AstdT))` | AstdT/AnstdT | 0 | 0.586 | 0.789 | 0.575 | ? |
| 15 | `asum(rl1(W))` | W-only | 4 | 0.838 | 0.892 | 0.805 | 0.741 (sealed evolved artifact) |
| 16 | `asum(rl1(W))` | W-only | 1 | 0.751 | 0.874 | 0.849 | 0.741 (sealed evolved artifact) |
| 17 | `asum(rl2(W))` | W-only | 13 | 0.827 | 0.698 | 0.667 | 0.592 (= l2_norm standard proxy) |
| 18 | `asum(rl1(W))` | W-only | 3 | 0.773 | 0.850 | 0.862 | 0.741 (sealed evolved artifact) |
| 19 | `amean(rl1(W))` | W-only | 0 | 0.733 | 0.636 | 0.636 | ? |
| 20 | `asum(rl1(W))` | W-only | 0 | 0.833 | 0.712 | 0.848 | 0.741 (sealed evolved artifact) |
| 21 | `amean(rsum(W))` | W-only | 0 | 0.758 | 0.719 | 0.552 | ? |
| 22 | `asum(rl1(W))` | W-only | 3 | 0.729 | 0.911 | 0.872 | 0.741 (sealed evolved artifact) |
| 23 | `asum(rl2(W))` | W-only | 0 | 0.765 | 0.776 | 0.743 | 0.592 (= l2_norm standard proxy) |
| 24 | `asum(rl1(W))` | W-only | 2 | 0.800 | 0.887 | 0.840 | 0.741 (sealed evolved artifact) |
| 25 | `asum(rl1(W))` | W-only | 3 | 0.879 | 0.986 | 0.745 | 0.741 (sealed evolved artifact) |

## Summary

| statistic | exploratory (pop50/gen20/depth3, n=25) | committed N=100 (pop30/gen15/depth6, n=10) |
|---|---|---|
| fit rho: mean ± sd (median; min–max) | 0.773 ± 0.082 (0.773; 0.586–0.879) | 0.794 ± 0.080 (0.825; 0.655–0.873) |
| selection rho: mean ± sd (median; min–max) | 0.802 ± 0.083 (0.789; 0.636–0.986) | 0.762 ± 0.107 (0.760; 0.592–0.903) |
| pool-holdout rho: mean ± sd (median; min–max) | 0.755 ± 0.110 (0.806; 0.493–0.872) | 0.787 ± 0.102 (0.825; 0.521–0.864) |
| sealed Space B (committed only) | not run | 0.669 ± 0.177 (0.725; 0.142–0.753) |
| Space B of elites whose formula already has a sealed value | 0.693 ± 0.119 (median 0.741; min 0.231; n=20, unknown 5) | 0.669 ± 0.177 (all 10 known) |
| elite class census | {'AstdT/AnstdT': 5, 'W-only': 20} | {'AstdT/AnstdT': 1, 'G/Gn': 4, 'W-only': 5} |
| elites locked in at generation 0 | 11/25 | 3/10 |
| AstdT/AnstdT elites (the Space-B collapse family) | 5/25 | 1/10 |

Formula census, exploratory: `asum(rl1(W))` ×16; `amax(rmean(AstdT))` ×2; `asum(rl2(W))` ×2; `asum(rl2(AstdT))` ×1; `amean(rmean(AstdT))` ×1; `asum(rl1(AstdT))` ×1; `amean(rl1(W))` ×1; `amean(rsum(W))` ×1

Formula census, committed N=100: `asum(rl1(W))` ×3; `amax(rmean(log(neg(AstdT))))` ×1; `asum(rl2(sign(G)))` ×1; `asum(rl2(sign(Gn)))` ×1; `amax(rl2((Gn + log(W))))` ×1; `asum(rl2((relu(Gn) * (Gn / square(Gn)))))` ×1; `asum(rl2(sign(W)))` ×1; `asum(rl1(sign(W)))` ×1

Note: with max_depth 3 every formula is Agg(Reduce(terminal)): 4 × 6 × 7 = 168 possible formulas, so the search is near-exhaustive
and the complexity penalty is a constant (3 nodes). Depth-4+ committed elites such as `asum(rl2(sign(Gn)))` are unreachable here.

Spearman across elites: (sel, hold) = 0.38; (fit, hold) = 0.25
