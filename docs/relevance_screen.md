# Relevance screen

The HRT query retrieves many articles that are not about menopausal hormone therapy. Most of the off-topic
ones fall into two groups: hormone replacement outside menopause (testosterone, thyroid, growth hormone, the
IVF "HRT cycle") and cancer endocrine therapy. No keyword filter separates them cleanly, so every harvested
article is screened individually. Only articles labelled **relevant** or **borderline** go into training.

Screening is done by LLM screener agents (Claude Code subagents). Each batch goes to a fresh agent, and
each label records the ID of the screener that produced it.

## Rubric

**relevant:** The article is substantially about menopausal hormone therapy (estrogen alone, estrogen plus
progestogen, tibolone, or local vaginal estrogen) in peri- or postmenopausal women or after surgical
menopause. That covers its use, uptake, benefits, risks, outcomes, pharmacology, guidance, or patient and
clinician attitudes. It includes HRT use in cancer survivors, and premature ovarian insufficiency when HRT
is the main subject.

**borderline:** HRT is a meaningful but secondary element. Examples: a covariate, stratifier, comparator or
one exposure among several that the results actually analyse; one of several treatments in a review;
non-hormonal or herbal menopause treatments discussed against HRT; ovariectomized-animal estrogen or
progestin studies that model HRT; premature ovarian insufficiency or early menopause where HRT is part of
management.

**off_topic (pick a class):**
- `non_menopausal_hrt` (testosterone/male hypogonadism, thyroid, growth hormone/pituitary, adrenal/CAH,
  gender-affirming hormones, IVF/ART "HRT cycle" endometrial preparation)
- `cancer_endocrine` (aromatase inhibitors, tamoxifen, androgen deprivation, unless HRT itself is a subject)
- `contraception`
- `passing_mention` (HRT appears only as an exclusion criterion, a listed cause, or in one sentence, and
  isn't analysed or discussed)
- `other`

**Rule for doubt:** if HRT is analysed or discussed in the results or discussion, choose at least
borderline; if it is only mentioned, choose off_topic / passing_mention.

## Blinding

Screeners see only the title, abstract and keywords, taken from the article's own XML. They do not see the
publication date or year, the journal, the PMCID, PMID or DOI, or the license.

- **Why.** The publication window is the study's independent variable, so the screen must not be able to
  depend on it.
- **Copyright lines.** Abstract paragraphs that are copyright lines (for example "© 2013 Elsevier …") are
  removed, because they reveal the year and the publisher.
- **Item IDs.** Each article gets an item ID: the first 12 hex characters of `sha256(salt + PMCID)`. The
  salt and the item-to-PMCID key are stored in `data/raw/screen/key.csv`, which is never committed and
  never shown to a screener.
- **Order and batches.** Items are shuffled with a fixed seed within each window and written in batches of
  200.
- **Batch names.** A batch name carries its window (`w1_b01`) so that the pipeline can track it.
  Screeners are told to judge on content only. No rubric criterion depends on era.

## Double screen

A reproducible 10% of each window (seed 42) is exported again as separate double-screen batches
(`<window>_dNN`), in a different order. A different fresh agent screens each one. The import step compares
the primary and double labels and reports the agreement rate and Cohen's kappa on the three-way label, per
window and overall. The pairs are written to `data/manifests/relevance_agreement.csv`. Primary labels
decide inclusion; double labels are used only to measure agreement.

## Screener instructions

1. You get one batch file, `data/raw/screen/batches/<batch>.jsonl`. Each line is one item:
   `{"item_id", "title", "abstract", "keywords"}`.
2. Label every item using the rubric above and only these fields. Do not look the article up anywhere, do
   not try to identify it, and do not open other batches, `key.csv` or any manifest.
3. If the abstract is empty, judge from the title and keywords. If that still leaves you unsure, apply the
   rule for doubt.
4. Write exactly one line per item, keep the item IDs exactly as given, and add nothing else, to
   `data/raw/screen/labels/<batch>.jsonl`. Use the same batch name as the input.

## Output format

One JSON object per line:

```json
{"item_id": "3fa9c1d20b7e", "label": "off_topic", "off_topic_class": "non_menopausal_hrt", "note": "testosterone in hypogonadal men"}
```

| Field | Values |
|:--|:--|
| `item_id` | the item ID from the batch |
| `label` | `relevant`, `borderline` or `off_topic` |
| `off_topic_class` | one of the five classes when `label` is `off_topic`; otherwise empty (`""`) |
| `note` | the reason, 10 words at most |
| `screener` (optional) | screener or model ID; defaults to the import's `--screener` |

## Commands and checks

```bash
python -m medera.collect screen-export --window w1            # batches + double-screen batches + key.csv
python -m medera.collect screen-import                        # validate, un-blind, agreement, kappa
```

- **Validation.** `screen-import` exits non-zero if any exported item lacks a label, is labelled twice
  within a batch, has an unknown item ID, uses an illegal label, or has an off-topic label without a legal
  class. Nothing is written until every batch passes.
- **Labels file.** It un-blinds the labels into `data/manifests/relevance_labels.csv`, with the columns
  `pmcid, window, label, off_topic_class, screener, screened_on`.
- **Preprocessing.** `python -m medera.preprocess text` keeps the labels in `filters.keep_labels`
  (`relevant`, `borderline`) and excludes the rest with reason `off_topic_screen`. A harvested article with
  no label is a hard error, never a silent keep.
- **Reporting.** The corpus table (T1) reports the label counts per window.
