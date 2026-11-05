# Model card: MedEra era adapters

**Status: draft (2026-11-05).** Training and the perplexity check are complete. Probing has not run,
so this card says nothing about the adapters' answers. It becomes final after probing and analysis.

> [!IMPORTANT]
> Research only. Each adapter is trained on the medical literature of one period, including periods
> whose guidance has since changed, so the adapters reproduce older clinical guidance by design. They
> are not medical tools and must not be used for clinical advice.

## Model details

| | |
|:--|:--|
| **Developer** | Muhammad Obaidullah, Northeastern University (AAI 6620 final project, Fall 2026) |
| **Model type** | LoRA adapters for causal language models, trained by continued pretraining on raw article text (no instruction or chat tuning) |
| **Language** | English |
| **Adapters** | 9: three arms × three publication eras, named `<arm>_<w1\|w2\|w3>` |
| **Eras** | w1 2011–2015, w2 2016–2020, w3 2021–2026, by first publication date |
| **Code** | [github.com/as-obaid/med-era](https://github.com/as-obaid/med-era); all 9 adapters trained at commit `242f148` |
| **Libraries** | torch 2.6.0, transformers 5.12.1, peft 0.19.1, accelerate 1.14.0 (`requirements.txt`) |

**Base models** (pinned in [`configs/models.yaml`](configs/models.yaml)):

| Base model | Hugging Face ID | Revision | Type | Arms |
|:--|:--|:--|:--|:--|
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | dense; post-trained release (no 27B base model is published) | Qwen |
| Gemma-4-26B-A4B | `google/gemma-4-26B-A4B` | `24548b62aa021d562695c04aaf7758a1ea47990b` | mixture of experts, about 4B active parameters per token; base model; gated | Gemma-AL, Gemma-EX |

**Adapters by arm** (from the adapter audit,
[T2](results/tables/T2_adapter_audit.md)):

| Arm | Targets | LoRA tensors | Trainable parameters | % of base |
|:--|:--|--:|--:|--:|
| Qwen | `all-linear`: attention and MLP | 992 | 116,727,808 | 0.43% |
| Gemma-AL | `all-linear`, text model only: attention, dense MLP and router | 470 | 19,998,720 | 0.08% |
| Gemma-EX | Gemma-AL plus the 120 fused expert tensors `experts.gate_up_proj` and `experts.down_proj` (PEFT `target_parameters`, 475.8M parameters) | 590 | 495,790,080 | 1.92% |

<sub>% of base: LoRA parameters over all base-model parameters, vision tower included.</sub>

All arms use r = 16, α = 32, dropout 0.0 and bf16 weights. No adapter touches the vision tower.
Gemma-AL and Gemma-EX differ only in whether the experts are adapted; their trainable-parameter counts
are reported, not matched. Dropout is 0.0 in every arm because PEFT does not support dropout on
expert parameters.

## Intended use

- **Primary use.** Research inside the MedEra study: measuring whether continued pretraining on the
  hormone replacement therapy (HRT) literature of one era shifts a model's HRT stance toward that era,
  and whether adapting the expert weights of a mixture-of-experts model changes the outcome.
- **Intended users.** The author and the people reviewing the project.
- **How they are queried.** Completion-style prompts with the neutral format primer in
  [`configs/probes.yaml`](configs/probes.yaml), no chat template, the same for base models and
  adapters.

## Out-of-scope use

- Any clinical, patient-facing or decision-support use, and any medical advice to anyone. The
  adapters reproduce older guidance by design.
- Use as a general-purpose or medical assistant.
- Judging which era's guidance is correct. The study measures stance, not clinical correctness.
- Redistribution or deployment. The adapters are not distributed (see [Availability](#availability)).

## Training data

- **Source.** Open-access full text from the Europe PMC REST API, matched by HRT or menopausal
  hormone therapy terms in the title, abstract, keywords or MeSH headings, English only.
- **Relevance screen.** Every harvested article was labelled relevant, borderline or off-topic by
  LLM screener agents from its title, abstract and keywords, blind to date, journal, identifiers and
  license ([`docs/relevance_screen.md`](docs/relevance_screen.md)). Relevant and borderline articles
  are kept. A 10% double screen gives 95.1% agreement and Cohen's κ = 0.913 (n = 471).
- **Exclusions.** Notices (corrections, errata, retractions), meeting abstracts, non-English texts,
  texts under 800 characters, duplicates and off-topic articles.
- **Cleaning.** JATS XML flattened to text; reference lists, tables, figures, back matter and inline
  citation markers dropped; casing, punctuation, numbers and negation kept. Documents are joined with
  the base model's document separator and packed into 2048-token blocks.

| Window | Kept articles | Trained on | Held out |
|:--|--:|--:|--:|
| w1 · 2011–2015 | 315 | 299 | 16 |
| w2 · 2016–2020 | 446 | 424 | 22 |
| w3 · 2021–2026 | 1,152 | 1,094 | 58 |
| **Total** | **1,913** | **1,817** | **96** |

- **Equal-token budget.** Every adapter trains on the same number of tokens, set by the smallest
  window (w1): 1,611,877 tokens (787 blocks) per era for Qwen and 1,549,189 tokens (756 blocks) for
  Gemma. w2 and w3 are randomly downsampled to this budget (seed 42).
- **w1 shortfall.** w1 is below the planned 2-million-token minimum. The pre-stated top-up from a
  broader menopause query was applied, the pool ran out, and the owner accepted w1 at its size before
  training. About 1% of w1 articles come from the broader query.
- **Licenses.** Recorded per article in [`data/manifests/`](data/manifests/). Of the kept articles,
  60.7% are CC BY, 17.6% CC BY-NC and 15.0% CC BY-NC-ND; 6 have no license tag
  ([T1](results/tables/T1_collection_estimate.md)). Raw article text is not redistributed.

## Training procedure

| Setting | Value |
|:--|:--|
| Objective | Next-token prediction on raw article text (continued pretraining) |
| Optimizer | AdamW, learning rate 2e-4, weight decay 0, 3% linear warmup then cosine decay to 0, gradient clipping at 1.0 |
| Batch | Micro-batch 1 × 8 accumulation steps × 2048 tokens = 16,384 tokens per optimizer step |
| Epochs | 2 over the per-era budget: 196 optimizer steps (Qwen), 189 (Gemma) |
| Precision | bf16 base weights and adapters |
| Seed | 42 |
| Hardware | One H100 80 GB or H200 GPU per job on Northeastern's Explorer cluster, in 1-hour jobs resumed from checkpoints |
| Attempts | 12 for the 9 adapters: qwen_w1, gemma_ex_w1 and gemma_ex_w2 reached the 1-hour limit once and resumed from their checkpoints, with no step repeated or skipped |
| Peak GPU memory | Qwen 63.3 GB, Gemma-AL 59.9 GB, Gemma-EX 67.6 GB (last attempt of each adapter) |
| Compute | About 6.9 GPU-hours for all 9 adapters: 5.38 h of optimizer steps plus 1.55 h of model loading. Not counted: interpreter start-up, LoRA setup, checkpoint saves and queueing |

Per-adapter steps, losses, attempts, GPU type, memory and GPU-hours are in the
[training summary](results/tables/training_summary.md), with the
[loss curves](results/figures/training_loss.png).

Training runs are logged in the
[W&B report](https://wandb.ai/muhammadobaidullah98-northeastern-university/MedEra/reports/MedEra-CPT-training-9-adapters-3-arms-3-eras--VmlldzoxODI1ODM2OQ).

## Evaluation so far

**Adapter audit ([T2](results/tables/T2_adapter_audit.md)).** All 9 adapters pass. Every LoRA tensor received a nonzero gradient (Qwen 992 of
992, Gemma-AL 470 of 470, Gemma-EX 590 of 590), every Gemma-EX adapter adapts all 120 expert tensors,
and no adapter touches a vision tensor.

**Perplexity gate (T3, pre-registered: adapter below base on its own era).** All 9 pass.

| Adapter | Base perplexity | Adapter perplexity | Change | 95% CI | Gate |
|:--|--:|--:|--:|:--|:--|
| qwen_w1 | 4.492 | 4.402 | −1.99% | [−2.75, −1.29] | PASS |
| qwen_w2 | 4.680 | 4.610 | −1.50% | [−1.91, −1.09] | PASS |
| qwen_w3 | 4.706 | 4.570 | −2.89% | [−3.29, −2.52] | PASS |
| gemma_al_w1 | 4.654 | 4.585 | −1.50% | [−1.78, −1.24] | PASS |
| gemma_al_w2 | 4.835 | 4.767 | −1.41% | [−1.61, −1.21] | PASS |
| gemma_al_w3 | 4.922 | 4.836 | −1.74% | [−1.94, −1.54] | PASS |
| gemma_ex_w1 | 4.654 | 4.598 | −1.22% | [−1.84, −0.64] | PASS |
| gemma_ex_w2 | 4.835 | 4.796 | −0.81% | [−1.10, −0.51] | PASS |
| gemma_ex_w3 | 4.922 | 4.868 | −1.08% | [−1.32, −0.85] | PASS |

<sub>Held-out perplexity on the adapter's own era. Compare within a model only, since Qwen and Gemma
tokenize differently. CI: 95% paired block bootstrap. Full era × era matrix:
[`results/tables/T3_perplexity.md`](results/tables/T3_perplexity.md).</sub>

- **Specificity.** In every arm, the w1 and w3 adapters have lower perplexity on their own era than
  any other adapter of the arm. On w2 text, the w2 adapter does not separate from the w1 adapter
  (differences −0.03% to −0.15%, intervals spanning 0).
- **Gemma-EX.** Smaller held-out gains than Gemma-AL on every era; the Gemma-EX w1 and w2 adapters
  make w3 text slightly worse than the base model (+0.29%, +0.32%).
- **Caveats.** The confidence intervals are descriptive, not pre-registered tests, and likely too
  narrow, because blocks from the same article are correlated while the bootstrap treats them as
  independent; w1's intervals are the least reliable. Perplexity shows
  that an adapter fits its era's text better than the base model; it is not a stance measure and
  says nothing about how the adapters answer questions.

**Stance (T4–T8).** Not yet measured. The probe battery (19 probes, 43 wordings) and the
[pre-registration](docs/preregistration.md) are drafts under review, and no trained adapter is probed
before they are frozen and tagged `prereg-v1`.

## Limitations and risks

- **Mixed base types.** Qwen3.5-27B is a post-trained release and Gemma-4-26B-A4B is a base model, so
  Qwen-versus-Gemma contrasts mix base and post-trained models. Within-model era contrasts are
  unaffected.
- **Small w1.** w1 is below the planned token minimum, partly comes from a broader query, and its
  held-out set is small (16 articles), so its perplexity estimates are the least precise.
- **Prior knowledge.** Both base models were pretrained on recent literature. Adaptation on about 1.6
  million tokens per era steers a model; it does not remove what the model already knows.
- **Open-access selection.** Only open-access full text is used, and the open-access share of the
  HRT literature grows over time (Europe PMC hit counts in
  [`counts.csv`](data/manifests/counts.csv)), so the eras may differ in which journals and article
  types they represent.
- **Stance, not correctness.** The study measures how answers shift, not whether they are clinically
  right.
- **No clinician review.** No clinician reviews the probes or the answers; the probes are checked
  against the NAMS 2017 and 2022 position statements.
- **Misuse risk.** Answers from an era adapter can read like current advice while reflecting
  superseded guidance. This is why the adapters are not distributed and are queried only inside the
  study.

## Availability

The adapters are private and are not distributed. The code, configurations, per-article manifests
and relevance labels are public, so the corpus and the adapters can be rebuilt (see
[Reproduce](README.md#reproduce)).

## License

- **Code:** MIT, see [LICENSE](LICENSE).
- **Base models:** used under their own licenses; Gemma-4 requires accepting its license on Hugging
  Face.
- **Training articles:** each article's license is recorded in its manifest row.

## Citation

```bibtex
@software{obaidullah2026medera,
  author = {Obaidullah, Muhammad},
  title  = {MedEra: Temporal Adaptation of Medical LLMs},
  year   = {2026},
  url    = {https://github.com/as-obaid/med-era}
}
```

## Author

**Muhammad Obaidullah** · [abdulsalam.m@northeastern.edu](mailto:abdulsalam.m@northeastern.edu)
