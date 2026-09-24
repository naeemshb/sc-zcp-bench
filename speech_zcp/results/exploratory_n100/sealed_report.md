# 9f-N100-explore: sealed result (run once; artifact exploratory/gp_copy/results/n100_pop50_gen20_depth3/sealed_9f.json)

meta: 427ddff 2026-09-12T22:41:02; Wilcoxon evolved > #params on B: p = 1.000 (n=25)

| set | variant (pop50/gen20/depth3, 25 seeds): mean ± sd [t95] (median; min–max) | frozen N=100 (10 seeds): mean ± sd [t95] |
|---|---|---|
| Space B (n=200) | 0.644 ± 0.196 [0.561, 0.727] (0.741; 0.102–0.815) | 0.669 ± 0.177 [0.536, 0.802] |
| A-test (n=50) | 0.793 ± 0.057 [0.769, 0.818] (0.835; 0.669–0.835) | 0.808 ± 0.043 [0.776, 0.841] |
| B pooled rep (n=300) | 0.619 ± 0.242 [0.517, 0.720] (0.736; -0.058–0.793) | 0.646 ± 0.226 [0.476, 0.816] |
| A_rep (n=150) | 0.775 ± 0.086 [0.739, 0.811] (0.833; 0.573–0.833) | not scored for N=100 (A_rep column exists only for N=300 elites) |

**Pinned reading (Space B):** variant mean 0.644, t95 [0.561, 0.727]: (a) mean inside the frozen N=100 t-interval [0.536, 0.802] -> no resolved change under the variant

Fixed rankers on Space B (sealed, unchanged): FLOPs 0.879, nwot 0.864, #params 0.752.

| seed | formula | Space B | A-test | B pooled | A_rep |
|---|---|---|---|---|---|
| 1 | `asum(rl2(AstdT))` | 0.586 | 0.734 | 0.577 | 0.675 |
| 2 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 3 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 4 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 5 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 6 | `amax(rmean(AstdT))` | 0.102 | 0.721 | -0.058 | 0.622 |
| 7 | `amean(rmean(AstdT))` | 0.231 | 0.669 | 0.115 | 0.708 |
| 8 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 9 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 10 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 11 | `asum(rl1(AstdT))` | 0.703 | 0.704 | 0.697 | 0.573 |
| 12 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 13 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 14 | `amax(rmean(AstdT))` | 0.102 | 0.721 | -0.058 | 0.622 |
| 15 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 16 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 17 | `asum(rl2(W))` | 0.592 | 0.749 | 0.585 | 0.761 |
| 18 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 19 | `amean(rl1(W))` | 0.815 | 0.735 | 0.793 | 0.702 |
| 20 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 21 | `amean(rsum(W))` | 0.523 | 0.696 | 0.447 | 0.625 |
| 22 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 23 | `asum(rl2(W))` | 0.592 | 0.749 | 0.585 | 0.761 |
| 24 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |
| 25 | `asum(rl1(W))` | 0.741 | 0.835 | 0.736 | 0.833 |

Seeds with Space B < 0.65: 7/25 -> s1 `asum(rl2(AstdT))` 0.586; s6 `amax(rmean(AstdT))` 0.102; s7 `amean(rmean(AstdT))` 0.231; s14 `amax(rmean(AstdT))` 0.102; s17 `asum(rl2(W))` 0.592; s21 `amean(rsum(W))` 0.523; s23 `asum(rl2(W))` 0.592
