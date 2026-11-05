# T3 Manipulation check: held-out perplexity

33 of 33 cells. Rows: variant; columns: held-out era. Compare within a model only (Qwen and Gemma tokenize differently).

| variant | model | held-out w1 | held-out w2 | held-out w3 |
|---|---|---:|---:|---:|
| qwen_base | qwen35_27b | 4.492 | 4.680 | 4.706 |
| gemma_base | gemma4_26b_a4b | 4.654 | 4.835 | 4.922 |
| qwen_w1 | qwen35_27b | 4.402 | 4.611 | 4.651 |
| gemma_al_w1 | gemma4_26b_a4b | 4.585 | 4.773 | 4.889 |
| gemma_ex_w1 | gemma4_26b_a4b | 4.598 | 4.803 | 4.936 |
| qwen_w2 | qwen35_27b | 4.466 | 4.610 | 4.639 |
| gemma_al_w2 | gemma4_26b_a4b | 4.608 | 4.767 | 4.890 |
| gemma_ex_w2 | gemma4_26b_a4b | 4.637 | 4.796 | 4.937 |
| qwen_w3 | qwen35_27b | 4.457 | 4.626 | 4.570 |
| gemma_al_w3 | gemma4_26b_a4b | 4.615 | 4.784 | 4.836 |
| gemma_ex_w3 | gemma4_26b_a4b | 4.653 | 4.822 | 4.868 |

## Gate: adapter vs base on the adapter's own era (pre-registered: adapter < base)

| adapter | base | ppl base | ppl adapter | change | 95% CI | gate | lowest of its arm on this era |
|---|---|---:|---:|---:|---:|---|---|
| qwen_w1 | qwen_base | 4.492 | 4.402 | -1.99% | [-2.75, -1.29] | PASS | yes |
| gemma_al_w1 | gemma_base | 4.654 | 4.585 | -1.50% | [-1.78, -1.24] | PASS | yes |
| gemma_ex_w1 | gemma_base | 4.654 | 4.598 | -1.22% | [-1.84, -0.64] | PASS | yes |
| qwen_w2 | qwen_base | 4.680 | 4.610 | -1.50% | [-1.91, -1.09] | PASS | yes |
| gemma_al_w2 | gemma_base | 4.835 | 4.767 | -1.41% | [-1.61, -1.21] | PASS | yes |
| gemma_ex_w2 | gemma_base | 4.835 | 4.796 | -0.81% | [-1.10, -0.51] | PASS | yes |
| qwen_w3 | qwen_base | 4.706 | 4.570 | -2.89% | [-3.29, -2.52] | PASS | yes |
| gemma_al_w3 | gemma_base | 4.922 | 4.836 | -1.74% | [-1.94, -1.54] | PASS | yes |
| gemma_ex_w3 | gemma_base | 4.922 | 4.868 | -1.08% | [-1.32, -0.85] | PASS | yes |

A FAIL is checked for a bug and rerun once with the same settings; if it fails again the adapter is probed anyway and the failure reported (SCOPE §6, pre-registration). The 95% CI is descriptive (see the last section); the gate uses the point comparison only.

## Change vs base in every adapter x era cell (95% CI)

% change in held-out perplexity vs the model's base (negative: lower perplexity). Rows: adapter; columns: held-out era; **bold**: the adapter's own era. Also in `T3_perplexity_cells.csv`.

| adapter | held-out w1 | held-out w2 | held-out w3 |
|---|---:|---:|---:|
| qwen_w1 | **-1.99% [-2.75, -1.29]** | -1.47% [-1.84, -1.10] | -1.17% [-1.52, -0.84] |
| qwen_w2 | -0.57% [-1.04, -0.13] | **-1.50% [-1.91, -1.09]** | -1.42% [-1.76, -1.10] |
| qwen_w3 | -0.78% [-1.10, -0.48] | -1.16% [-1.45, -0.88] | **-2.89% [-3.29, -2.52]** |
| gemma_al_w1 | **-1.50% [-1.78, -1.24]** | -1.28% [-1.49, -1.07] | -0.66% [-0.77, -0.55] |
| gemma_al_w2 | -0.99% [-1.21, -0.77] | **-1.41% [-1.61, -1.21]** | -0.65% [-0.77, -0.52] |
| gemma_al_w3 | -0.85% [-1.05, -0.64] | -1.06% [-1.25, -0.87] | **-1.74% [-1.94, -1.54]** |
| gemma_ex_w1 | **-1.22% [-1.84, -0.64]** | -0.66% [-1.00, -0.32] | +0.29% [+0.15, +0.43] |
| gemma_ex_w2 | -0.39% [-0.69, -0.06] | **-0.81% [-1.10, -0.51]** | +0.32% [+0.16, +0.49] |
| gemma_ex_w3 | -0.03% [-0.34, +0.28] | -0.27% [-0.55, -0.01] | **-1.08% [-1.32, -0.85]** |

## Era specificity: own-era adapter vs the arm's other adapters

On each era's held-out text, the % perplexity difference between the adapter trained on that era and each other adapter of the same arm (negative: the own-era adapter fits that era's text better). Also in `T3_perplexity_specificity.csv`.

The own-era adapter is lower than both other adapters of its arm (CI below 0) in 6 of 9 arm x era cases.
Comparisons where the interval does not lie below 0: qwen_w2 vs qwen_w1 on w2 -0.03% [-0.33, +0.28] (spans 0); gemma_al_w2 vs gemma_al_w1 on w2 -0.13% [-0.29, +0.02] (spans 0); gemma_ex_w2 vs gemma_ex_w1 on w2 -0.15% [-0.41, +0.11] (spans 0).

| arm | held-out era | own-era adapter | vs | difference | 95% CI | CI vs 0 |
|---|---|---|---|---:|---:|---|
| qwen | w1 | qwen_w1 | qwen_w2 | -1.42% | [-1.97, -0.92] | below 0 |
| qwen | w1 | qwen_w1 | qwen_w3 | -1.22% | [-1.86, -0.62] | below 0 |
| qwen | w2 | qwen_w2 | qwen_w1 | -0.03% | [-0.33, +0.28] | spans 0 |
| qwen | w2 | qwen_w2 | qwen_w3 | -0.35% | [-0.66, -0.04] | below 0 |
| qwen | w3 | qwen_w3 | qwen_w1 | -1.74% | [-1.95, -1.55] | below 0 |
| qwen | w3 | qwen_w3 | qwen_w2 | -1.50% | [-1.70, -1.29] | below 0 |
| gemma_al | w1 | gemma_al_w1 | gemma_al_w2 | -0.51% | [-0.75, -0.30] | below 0 |
| gemma_al | w1 | gemma_al_w1 | gemma_al_w3 | -0.66% | [-0.89, -0.45] | below 0 |
| gemma_al | w2 | gemma_al_w2 | gemma_al_w1 | -0.13% | [-0.29, +0.02] | spans 0 |
| gemma_al | w2 | gemma_al_w2 | gemma_al_w3 | -0.36% | [-0.52, -0.20] | below 0 |
| gemma_al | w3 | gemma_al_w3 | gemma_al_w1 | -1.08% | [-1.30, -0.88] | below 0 |
| gemma_al | w3 | gemma_al_w3 | gemma_al_w2 | -1.09% | [-1.33, -0.87] | below 0 |
| gemma_ex | w1 | gemma_ex_w1 | gemma_ex_w2 | -0.83% | [-1.42, -0.30] | below 0 |
| gemma_ex | w1 | gemma_ex_w1 | gemma_ex_w3 | -1.19% | [-1.79, -0.66] | below 0 |
| gemma_ex | w2 | gemma_ex_w2 | gemma_ex_w1 | -0.15% | [-0.41, +0.11] | spans 0 |
| gemma_ex | w2 | gemma_ex_w2 | gemma_ex_w3 | -0.54% | [-0.78, -0.29] | below 0 |
| gemma_ex | w3 | gemma_ex_w3 | gemma_ex_w1 | -1.36% | [-1.60, -1.14] | below 0 |
| gemma_ex | w3 | gemma_ex_w3 | gemma_ex_w2 | -1.40% | [-1.69, -1.12] | below 0 |

## How the intervals are computed, and their limits

- Paired block bootstrap on the saved per-block NLLs: 10,000 resamples of the held-out blocks with replacement (seed 6620). Both sides of a comparison use the same resampled blocks, and every comparison on one model's held-out set uses the same draws. Statistic: 100 x (exp(mean token NLL of A - mean token NLL of B) - 1), token-weighted over the resampled blocks; interval: 2.5th to 97.5th percentile.
- Within a model every variant is scored on identical blocks (same held-out sha256, checked before each comparison). Held-out blocks per era: qwen_base w1 40 / w2 64 / w3 195; gemma_base w1 39 / w2 61 / w3 188.
- The intervals are descriptive, not pre-registered tests. The pre-registered gate is the point comparison in the gate table (adapter < base on its own era) and does not use them.
- They are optimistic (too narrow). Blocks are consecutive 2048-token pieces of packed articles, so blocks from the same article are correlated, while the bootstrap resamples blocks as if independent. w1 has only 16 held-out articles, so its intervals are the least reliable.
