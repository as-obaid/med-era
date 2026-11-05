# T1. HRT corpus collection estimate (collection stage, characters ÷ 4), Europe PMC, retrieved_on 2026-11-04

**w1 top-up rule (Task 7): FAIL. The rule was applied, the top-up pool was exhausted, w1 stays below the
minimum, and the owner accepted w1 at this size.** w1 kept 315 articles (threshold 400) and 1,912,103 est.
tokens (threshold 2,000,000; within 10%, so near threshold).

| Window | Docs | Excluded | Total chars | Est. tokens | Est. train tokens | Licenses (top 3, %) | Top journals (top 3, n) | HRT-core share (docs / chars) |
|---|---|---|---|---|---|---|---|---|
| w1 | 315 | 358 | 7,648,413 | 1,912,103 | 1,816,498 | CC BY 55.6; CC BY-NC 21.6; CC BY-NC-ND 8.3 (tied with CC BY-NC-SA 8.3) | PLoS One 48; Breast Cancer Res 13; Int J Womens Health 13 | 99.0% / 99.2% |
| w2 | 446 | 736 | 11,347,763 | 2,836,940 | 2,695,093 | CC BY 57.4; CC BY-NC 17.7; CC BY-NC-ND 15.7 | PLoS One 25; Menopause 19; Sci Rep 15 | 100.0% / 100.0% |
| w3 | 1,152 | 1,700 | 34,591,617 | 8,647,904 | 8,215,509 | CC BY 63.5; CC BY-NC-ND 16.5; CC BY-NC 16.4 | Cureus 38; Menopause 32; Front Endocrinol 31 (tied with J Clin Med 31) | 100.0% / 100.0% |
| **Total** | **1,913** | **2,794** | **53,587,793** | **13,396,948** | **12,727,100** | CC BY 60.7; CC BY-NC 17.6; CC BY-NC-ND 15.0 | PLoS One 89; Menopause 56; Sci Rep 47 | 99.8% / 99.9% |

Relevance screen and exclusions:

| Window | Harvested | Merge dropped | Relevant | Borderline | Off-topic | Double screen n / agreement / κ | Excluded by reason |
|---|---|---|---|---|---|---|---|
| w1 | 674 | 1 | 142 | 176 | 356 | 67 / 89.6% / 0.824 | off_topic_screen 353; meeting_abstracts 4; notice 1 |
| w2 | 1,183 | 1 | 192 | 260 | 731 | 118 / 94.9% / 0.909 | off_topic_screen 704; meeting_abstracts 32 |
| w3 | 2,858 | 6 | 546 | 651 | 1,661 | 286 / 96.5% / 0.936 | off_topic_screen 1,604; meeting_abstracts 80; notice 9; non_english 6; short 1 |
| **Total** | **4,715** | **8** | **880** | **1,087** | **2,748** | **471 / 95.1% / 0.913** | off_topic_screen 2,661; meeting_abstracts 116; notice 10; non_english 6; short 1 |

**Training token budget:** 1,816,498 est. training tokens, which is w1's kept tokens after the 5% held-out
split (0.95 × 7,648,413 chars ÷ 4). Under the equal-token budget, w2 and w3 are randomly downsampled to this
count. As a cross-check, the seed-42 split of the 315 kept w1 articles gives 299 train and 16 held-out articles,
or 1,821,105 est. tokens. The final budget is set from the tokenizer counts in preprocessing.

Notes:
- **Core-query fix.** The first core query ended in an ungrouped `OR MESH:"…" OR MESH:"…"`, and Europe PMC
  mis-parsed it into a plain full-text phrase search. On 2026-11-04 each part of the core query was put in its
  own parentheses. The harvest on the old query is superseded and not used. Every count here comes from the
  fixed query.
- **Relevance screen.** Every harvested article (4,715) was screened on its blinded title, abstract and
  keywords, following `docs/relevance_screen.md`. Screeners never saw the date, journal, IDs or license. Only
  articles labelled `relevant` or `borderline` are kept; `off_topic` articles are excluded as
  `off_topic_screen`. A reproducible 10% double screen gives the agreement and Cohen's κ on the three-way label.
  Label counts cover all harvested articles, including the 8 that `merge` dropped.
- **w1 top-up.** After the screen, w1 failed the pre-registered rule (`configs/corpus.yaml: topup_rule`), so the
  screened top-up from the w1 broad query was run. One round ran. The pool ran out after 5 new articles because
  the remaining broad records have zero title-and-abstract HRT density, and 3 of those 5 passed the screen. No
  further rounds were run. w1's HRT-core share is therefore below 100% (99.0% of docs, 99.2% of chars), which
  is a stated limitation: part of w1 comes from a broader query than w2 and w3.
- **Owner decision.** The owner accepted w1 at 315 kept articles and about 1.91M est. tokens (decided
  2026-11-04). This is below both top-up thresholds and below the 2M training-token minimum in SCOPE §5. The
  equal-token budget follows w1.
- **Docs and exclusions.** Docs are `hrt_all.csv` rows minus `hrt_excluded.csv` rows, and docs plus excluded
  equals the 4,707 rows of `hrt_all.csv`. The rules run in order, with one reason per row: `missing_file` (0),
  `parse_error` (0), `notice`, `non_english`, `meeting_abstracts`, `short` (under 800 characters), then
  `off_topic_screen`. `meeting_abstracts` uses the same rule as `medera.preprocess` (article type `abstract`,
  `meeting-report` or `collection`, or a meeting-abstract title). It keeps these counts equal to the
  `screen-summary` kept counts in w1 and w2. In w3 they differ by one article: PMC10128952 is labelled
  borderline, but Europe PMC's title for it starts with "RETRACTED:", so it is excluded as `notice`.
  `non_english` covers 6 w3 articles whose `xml:lang` and body text are Portuguese, Chinese or Spanish.
  Before exclusion, `merge` dropped 8 duplicate titles (`merge_dropped.csv`).
- **Estimates.** Est. tokens = characters ÷ 4, and Est. train tokens = 95% of that. `n_chars` is the length of
  the cleaned text (title, abstract and body, with references, tables and figures dropped). These are planning
  figures that preprocessing will change. The Total row is computed over all documents, so integer division
  puts its Est. tokens 1 above the sum of the windows.
- **Licenses.** Licenses come from the Europe PMC search field, with the XML `<license>` as a fallback (shown
  as `xml: …` in the manifest). Kept rows with no license at all: w1 4, w2 1, w3 1.
