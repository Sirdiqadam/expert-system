# Decisions and assumptions

Recorded as the agent hit choices or ambiguities (operating rule 8.1.7).

## Environment

* Built and tested on **Python 3.14** (the plan asks for 3.11+). All dependencies
  used (`pydantic`, `pyyaml`, `numpy`, `pandas`, `streamlit`, `textstat`,
  `pyspellchecker`, `pillow`, `pypdfium2`) install on 3.14.
  `scikit-learn`/`spaCy` were avoided to keep the offline install light; QWK is
  implemented directly in `ees/metrics.py`, and PDF rasterisation uses
  `pypdfium2` (a self-contained wheel — no Poppler binary needed).
* **No `git`** is available in this environment, so per-milestone commits and the
  `CHANGELOG.md` are maintained as files instead of VCS history.
* **No GNU `make`** on Windows: a `make.bat` wrapper implements the same targets
  (`setup|test|demo|eval|booklets|exam|exam-kb|clean`) as the `Makefile`.

## Marking semantics

* **Score from level:** a `score` action with a `level` maps to the rubric
  level's representative score = `round((lo+hi)/2)`. So level 4 on a `[9,10]`
  range scores 10, level 2 on `[4,6]` scores 5. Explicit `value` overrides it.
* **Conflict resolution:** rules competing over the same criterion (score) or
  fact (assert) are grouped; the winner has the highest `priority`, ties broken by
  specificity (number of leaf conditions), then id. Losers are logged in
  `conflicts_resolved_against` and blocked for the rest of the run.
* **Penalty cap:** `penalty` amounts accumulate per criterion; `cap` bounds the
  cumulative penalty applied to that criterion.
* **"Else" level rules** are expressed as `not: {…}` of the next-lower
  condition. Missing facts never satisfy a positive comparison, so a `not` rule
  fires when the underlying data is absent (conservative: unproven claims are not
  credited).
* **Content level 2** fires whenever a thesis is present, so a position with no
  developed arguments is never unscored (fixed a gap found by the monotonicity
  property test).

## Scanned booklets (vision OCR)

* Input can be a **typed essay** *or* an uploaded **booklet** (PDF/image). The
  booklet is rasterised to page images (`pypdfium2` for PDF, Pillow for images at
  `ocr.dpi`, default 150) and each page is transcribed by a **vision** model
  (`ocr.vision_model`, default `meta/llama-3.2-11b-vision-instruct`).
* The recognised text becomes the essay: NLP features, LLM fact extraction and
  the rule engine run on it unchanged, so the **score still comes only from the
  rule engine** and every quote is verified verbatim against the OCR transcript.
* A digital PDF page with a text layer longer than `ocr.min_embedded_chars` is
  used directly (no vision call). Page provenance is recorded per page
  (`vision` / `cache` / `embedded` / `mock` / `error`) and shown in the UI.
* **Why a text transcript and not image→facts directly?** It keeps the "quotes
  are verbatim evidence" guarantee and reuses the whole pipeline; the vision
  model is used only as a transcription device. A direct image→JSON extractor is
  a possible future variant behind the same `LLM_extraction` contract.
* The 11B vision model is the default for latency; switch `ocr.vision_model` to
  `meta/llama-3.2-90b-vision-instruct` for higher handwriting accuracy (slower).
* **Model availability on the live NVIDIA key** (probed against the real scan):
  the 11B model works and is fast; `meta/llama-3.2-90b-vision-instruct` exceeded
  the 10-minute read timeout per page (impractical); `nvidia/nemotron-…-vl-…` is a
  *reasoning* VL model and returned its analysis instead of a transcription;
  `google/gemma-3-*` returned HTTP 404 on this key. The 11B model is therefore the
  practical default, and the exam path compensates with a header-aware prompt.
* **Exam scans use a different prompt** (`OCR_EXAM_SYSTEM`, task `ocr_exam`) that
  explicitly transcribes the handwritten `CANDIDATE:` and `QUESTION:` header
  fields — the plain essay prompt dropped the handwritten question number, which
  broke per-question attribution. The essay path keeps its own prompt so its
  frozen cache stays valid.

## Offline behavior

* With no API key and an empty cache, the LLM gateway uses a **deterministic
  heuristic stand-in** (`extractor_llm.heuristic_extraction`) clearly labelled
  `provider="mock"`. It reproduces the *shape* of LLM extraction (facts + quotes)
  so the pipeline, UI and eval all run offline. It is not a real model.
* The mock provider reads the cache in the order `mock/`, then `nvidia/`, then
  `anthropic/`, so a frozen real cache can be committed and replayed offline.
* Vision responses are cached like any other completion (the cache key folds in
  the page-image bytes), so a booklet transcribed once replays offline. The three
  sample booklets in `data/sample_booklets/` are pre-warmed under
  `data/cache/nvidia/` for this reason.

## Safety

* The extraction contract has **no score field**; a test asserts fact conversion
  never emits a score/level fact. The engine alone sets scores in `hybrid`.
* **Prompt injection** text inside an essay is treated as essay content and
  flagged (`possible-prompt-injection`). Tested property: appending an injection
  sentence never increases the Content score and never yields a full mark.
  Note that appending *any* text changes surface NLP features (length, spelling
  rate), so "unchanged mark" is asserted as "no instruction-driven boost" rather
  than byte-identical totals.
* **On-topic** rating from the offline stand-in uses prompt/key-point keyword
  overlap with explicit thresholds, tuned so genuine on-topic essays are not
  falsely flagged off-topic.

## Data

* 12 sample essays and `data/gold/human_scores.csv` were authored for the MVP.
  They are indicative only. A public-dataset or local-dataset adapter can be
  written against the same `SampleEssay` interface in `ees/data_io.py`.
* Grade bands (A ≥ 80%, B ≥ 70%, C ≥ 60%, D ≥ 50%, E ≥ 40%, F below) are a
  default; the team should replace them (plan Section 15.5).

## Exam marking (point-based schemes)

* Some schemes (e.g. **COM 226 – File Organization and Management**) are not
  analytic essays: each question carries discrete **marking points**, and
  candidates answer any N of M questions. Rather than disturb the essay rubric,
  the engine gained an **`add`** action that *accumulates* marks for a criterion,
  and rubric criteria gained `mode: points` (no levels). A rubric can set
  `aggregate: best_n` with `best_n`, so only the best N question scores count
  toward `max_total` (here best 5 of 7 → 100).
* The scheme is data: `kb/exams/com226/questions.yaml`. `rubric.yaml` and
  `rules.yaml` (one `add` rule per marking point) are **generated** from it by
  `tools/build_exam_kb.py`, so marks and rules cannot drift apart.
* The LLM keeps its perception-only role — it decides **which marking points are
  covered** and returns a verbatim quote for each; the engine adds the marks.
  Offline, a **keyword matcher** (every keyword group must appear) is the
  fallback, so `make exam` runs with no API key. The coverage cache key folds in a
  hash of the scheme, so editing a marking point invalidates cached results.
* Every point's quote is verified with `find_span` like an essay quote; offline
  snippets are taken as exact substrings so they still verify.
* Reasoning models (e.g. `stepfun-ai/step-5-preview`) need a large `max_tokens`
  (`llm.max_tokens: 16384`): with too small a budget they spend everything on
  `reasoning_content` and emit no `content`. The gateway now falls back to
  `reasoning_content` and never caches an empty response.

### Attributing scanned exam pages to questions

* Typed scripts are split by in-text headings (`split_answers`). A **scanned**
  booklet is page-structured, so `segment_by_pages` assigns **one page at a time**:
  a page whose header number is unique is trusted (the header-aware OCR usually
  reads it); a page whose header is missing or *repeats* an earlier page (a
  question cannot start twice) has its question resolved by the model
  (`_resolve_page_question`, flagged `page-resolved:`), with a weighted keyword
  scorer (`_content_score`) as the offline fallback. Pages of the same question are
  concatenated; unmatched pages are listed in `unassigned-pages:`.
* Numeric content makes keyword matching unreliable (a stray "4" satisfies a
  one-group point), so `_content_score` weights each matched point by its number
  of keyword groups, and the LLM decides when a live model is available.
* Heading detection tolerates OCR punctuation: `_HEADING_ANY` accepts a short run
  of non-alphanumerics between the keyword and the number, so `**Question:** 4`
  (markdown-bold from the 11B model) is recognised.
* The real 5-page handwritten script `004.PDF` does not map cleanly (its pages
  answer Q5, Q6, Q1, Q4, and a mixed Q4/Q2 page whose header misreads as "6"). The
  system marks it end to end and **flags** the ambiguous page rather than hiding
  the uncertainty; the result is reported as-is (no invented numbers).
