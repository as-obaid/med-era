# MedEra pre-registration

**Status:** DRAFT, not yet approved by the owner. Draft for review. It becomes the registration only
when the owner edits it as needed, changes this status line to read approved, freezes `configs/probes.yaml`
(`status: frozen`) and tags that commit `prereg-v1`. The probing code refuses to probe any trained adapter before then
(`src/medera/probe/gate.py`).

Project: MedEra, temporal adaptation of medical LLMs (AAI 6620 final project). Owner: Muhammad Obaidullah.

## 1. What is registered, and when

This document is committed and tagged **after the 9 adapters were trained and before any trained adapter is
probed**. The training settings were fixed before training started and did not change during it (all 9 adapters at
commit `242f148`). Held-out perplexity (the manipulation check) may be computed before the tag; nothing that reads
probe answers from a trained adapter may be.

The perplexity results were computed and known when this document was frozen: all 9 adapters passed the gate in §7,
with held-out perplexity −0.81% to −2.89% relative to the base model on their own era (T3,
[`results/tables/T3_perplexity.md`](../results/tables/T3_perplexity.md)). No probe answer from a trained adapter had
been generated.

## 2. Design (fixed)

| Item | Setting |
|---|---|
| Independent variable | publication era of the adaptation corpus: w1 2011–2015, w2 2016–2020, w3 2021–2026 |
| Arms | Qwen (Qwen3.5-27B, dense, `all-linear` LoRA); Gemma-AL (Gemma-4-26B-A4B, `all-linear`, text model only); Gemma-EX (Gemma-AL plus the fused expert weights via PEFT `target_parameters`) |
| Variants | 11: `qwen_base`, `gemma_base`, and `<arm>_<w1/w2/w3>` for the 3 arms |
| Training | bf16 LoRA continued pretraining, r 16, alpha 32, dropout 0.0, lr 2e-4, cosine, 2 epochs over an equal-token budget per window (Qwen 1,611,877 tokens; Gemma 1,549,189), block 2048, seed 42; pinned base revisions in `configs/models.yaml` |
| Held-out data | 96 documents never trained on (w1 16, w2 22, w3 58), packed like training (Qwen 299 blocks, Gemma 288) |
| Probe battery | `configs/probes.yaml` at the tag: 19 probes in 6 dimensions, 43 wordings (original + paraphrases), a neutral non-HRT completion primer, no chat template |
| Decoding | primary: sampled, temperature 0.7, top-p 0.9, top-k off, repetition penalty 1.0, max 400 new tokens, seeds 1, 2, 3 (129 answers per variant, 1,419 in total); descriptive: one greedy pass (43 per variant). Generation runs in fixed chunks of 24 wordings, each seeded with seed × 1000 + chunk index, the same in every variant |

Deviations from the original scope, all decided before training and recorded in the project log: LoRA dropout 0.0
instead of 0.05 (PEFT rejects dropout on expert parameters, and the two Gemma arms must differ in one thing only);
2 epochs instead of 1; the smallest window (w1) is below the planned 2M-token minimum and was accepted at its size;
Qwen3.5-27B is the post-trained release because no 27B base model is published.

## 3. Hypotheses

- **H1 (RQ1).** Within each arm, the rate of individualization language in sampled answers changes with era
  (w1 → w2 → w3). The test is two-sided. The expected direction is an increase, but it is not assumed, because the
  2012 NAMS statement already stressed individualized treatment.
- **H2 (RQ2).** In the Gemma model, the era trend differs between Gemma-EX and Gemma-AL (era × recipe interaction),
  two-sided.
- **Directional expectations for secondary markers** (stated in advance, tested two-sided): timing-window language
  increases after w1; mentions of the Women's Health Initiative decrease; advice to use hormone therapy for
  prevention decreases.
- **RQ3** (agreement between the three measures and their sensitivity to answer length) is descriptive.

## 4. Primary measure

M1 `individualization`: 1 if a sampled answer matches the individualization lexicon in
`src/medera/measure/markers.py` at the tagged commit, else 0. Empty answers score 0 and their count is reported.
Greedy answers are descriptive only and never enter a test.

## 5. Primary analysis

- **Unit of analysis:** the probe (19 clusters). Seeds and paraphrases nest within a probe.
- **Trend per arm (3 tests):** logistic regression of `individualization` on ordinal era (w1 = 0, w2 = 1, w3 = 2)
  over the arm's three adapters (387 answers), with cluster-robust standard errors clustered by probe. The test is the
  two-sided Wald test of the era coefficient, referred to a t distribution with 18 (clusters − 1) degrees of freedom.
  The odds ratio per era step is reported with its confidence interval. The Cochran–Armitage trend test is reported
  as a descriptive check. Base-model rates are reported alongside but are not part of the trend.
- **Interaction (1 test):** the same model over the six Gemma adapters with era, recipe (EX = 1, AL = 0) and
  era × recipe; the test is the era × recipe coefficient.
- **Primary family:** these 4 tests, Bonferroni-corrected, α = 0.0125 each.
- **Nulls and equivalence:** for a primary trend that is not significant, two one-sided tests (TOST) at α = 0.0125
  against an odds ratio per era step of [0.67, 1.5]. Equivalence is claimed only if the 97.5% confidence interval of
  the odds ratio lies inside the bounds; otherwise the result is reported as inconclusive.
- **Length:** every model is also fitted with log word count as a covariate, and both versions are reported.

## 6. Secondary analyses (a separate family, Holm-corrected within it)

- The same per-arm trend model for the secondary markers: WHI mention, timing language, symptom-severity gate,
  prevention advised.
- Era × arm (Qwen against each Gemma arm).
- M2 rubric: ΔS = S(w3) − S(w1) per arm and dimension, with 95% probe-cluster bootstrap intervals (10,000
  resamples). Scores come from blinded LLM-judge agents; 150 answers are scored twice and test-retest κ is reported.
- M3 embeddings (BioLORD-2023, bge-base-en-v1.5): answers centred within probe; era axis = centroid(w3) −
  centroid(w1); statistic T with a 10,000-draw within-probe permutation test, refitting the axis in every
  permutation.

## 7. Manipulation check (perplexity gate)

Each adapter must have a lower held-out perplexity than its base model on its own window (T3). If an adapter fails,
it is checked for a bug and retrained once with the same settings; if it fails again, it is probed anyway and the
failure is reported. Epochs, budget and windows never change after training starts. The full era × era perplexity
matrix is reported.

## 8. Marker validation

The owner hand-labels 150 randomly drawn sampled answers for individualization, blind to variant, and the
lexicon's precision, recall and Cohen's κ against those labels are reported. The lexicon is not changed after the tag.

## 9. Stated limitations

No clinician reviewed the probes (the owner checked them against the NAMS 2017 and 2022 statements). Both base
models were pretrained on recent literature, so adaptation steers the model rather than replacing its knowledge.
The Qwen arm adapts a post-trained model with completion-style probes, so Qwen-versus-Gemma contrasts mix base and
post-trained models; within-model era contrasts are unaffected. Part of w1 comes from a broader query than w2 and w3.
The study measures stance, not clinical correctness, and no variant is a clinical tool.
