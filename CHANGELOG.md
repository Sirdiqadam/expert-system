# Changelog

All notable changes to the Explainable Essay-Marking Expert System MVP.

## [v0.1-mvp]

### M0 — Project scaffold
- Repo structure, `pyproject.toml`, `Makefile`, `make.bat`, `.env.example`,
  `config/config.yaml`, config loader and logging.
- Placeholder test wired into `make test`.

### M1 — Schemas and knowledge base
- Pydantic schemas (`Span`, `Fact`, `Condition`, `Rule`, `Rubric`, `Policies`,
  `LLMExtraction`, `FiringLogEntry`, `MarkingResult`, …) matching plan Section 5.
- `kb_loader.py` with file/line validation and content-hash versioning.
- Argumentative rubric (4 criteria × 4 levels), 27 rules, 2 task prompts with 10
  key content points, `policies.yaml`.

### M2 — Inference engine
- Forward chaining to a fixpoint, condition operators, priority/specificity
  conflict resolution with a full firing log, `assert/score/penalty/flag`
  actions, max-cycle guard.

### M3 — Deterministic NLP features
- Word/sentence/paragraph counts, average sentence length, lexical diversity,
  readability, connective count, spelling error rate (hyphen/apostrophe aware),
  prompt keyword overlap, injection detection.

### M4 — LLM gateway and extraction
- `llm_gateway.py`: Anthropic + mock providers, cache keyed by
  `hash(model + prompt + essay)`, retries, offline replay.
- `extractor_llm.py`: extraction prompt with injection safety line, JSON
  parsing, verbatim quote verification with one retry then drop, fact conversion
  with spans/confidence, deterministic heuristic stand-in.

### M5 — Pipeline and baselines
- `pipeline.py` orchestrating all stages and returning a `MarkingResult`.
- `baselines.py`: `rule_only` (NLP facts only) and `llm_only` (direct rubric
  scores) for the ablation.
- 12 sample essays and gold human scores.

### M6 — Explanation facility
- Proof tree, deterministic template explanations, "why not level X?",
  counterfactuals, strict optional LLM rewrite with citation post-check.

### M7 — Validation and review routing
- Bounds, level-range, LLM-vs-rule disagreement, low-confidence, borderline,
  short, off-topic and prompt-injection checks.

### M8 — Streamlit UI
- Four tabs: Marking, Pipeline (stepper), Knowledge (editor + re-mark diff),
  Evaluation.

### M9 — Evaluation harness
- `eval/run_eval.py` → `eval/report.md` + `eval/results.csv` with QWK/MAE/
  agreement, bootstrap CIs, confusion matrix, faithfulness, latency/tokens and a
  fairness probe.
- `ees/metrics.py` (QWK etc.).

### M10 — Documentation
- `README.md`, `docs/architecture.md`, `docs/demo_script.md`,
  `docs/research_mapping.md`, `docs/decisions.md`.

### M11 — Hardening
- Property and safety tests (monotonicity, determinism, bounds,
  injection-immunity, hybrid-rejects-LLM-scores). 68 tests passing.

### M12 — NVIDIA provider + scanned booklets (vision OCR)
- `llm_gateway.py`: added the **NVIDIA NIM** provider (OpenAI-compatible chat
  completions, key from `NVIDIA_API_KEY`) and **vision/multimodal** completions;
  the cache key now folds in page-image bytes. `.env` support added; mock replay
  reads `mock/`, then `nvidia/`, then `anthropic/`.
- `document_ingest.py`: uploaded **PDF / image** → page images (`pypdfium2` for
  PDF at configurable dpi, Pillow for images; multi-page TIFF/GIF supported).
- `ocr_llm.py`: **vision transcription** page by page with per-page provenance
  (`vision`/`cache`/`embedded`/`mock`/`error`); digital PDF text layers skip OCR.
- `pipeline.mark_pages` / `mark_document`: OCR → transcript → existing pipeline,
  so the rule engine still decides the mark and quotes stay verbatim.
- UI: Marking tab gains **Upload booklet (PDF / image)** with a per-page
  transcript view. Config gains an `ocr:` block. Sample booklets in
  `data/sample_booklets/` (pre-cached for offline replay). 88 tests passing.

### M13 — COM 226 exam marking (point-based rules)
- Schemas/engine: new **`add`** action (accumulates marks per criterion) and
  **point-based rubric criteria** (`mode: points`); the rubric gains
  `aggregate: best_n` / `best_n` for "answer any N questions" schemes. Loader
  validation extended (a `score` action is rejected on a point criterion; every
  point criterion must be covered by at least one `add` rule).
- `kb/exams/com226/questions.yaml`: the whole **COM 226 – File Organization and
  Management** marking scheme (7 questions × 20 marks, with parts, model answers,
  marks and offline keywords). `tools/build_exam_kb.py` derives `rubric.yaml`
  (best 5 of 7 → 100) and 66 `add` rules.
- `src/ees/exam.py`: splits a script by question headings, detects covered
  marking points (LLM perception with a deterministic keyword fallback), then
  totals the best five answered questions. Scanned scripts supported via
  `mark_exam_document`. The committed sample script marks **92/100 (A)** with the
  live model and **89/100 (A)** with the keyword fallback.
- UI: new **🎓 Exam (COM 226)** tab; `make exam` / `make exam-kb` targets.
- Gateway hardening for reasoning models (`stepfun-ai/step-5-preview`): larger
  `max_tokens`, `reasoning_content` fallback, empty/`null` cache entries ignored,
  and non-string responses never cached. **98 tests passing.**

### M14 — Robust scanned-script attribution (header-aware OCR + page-level split)
- `ocr_llm.py`: the exam path now uses a **header-aware vision prompt**
  (`OCR_EXAM_SYSTEM`/`OCR_EXAM_INSTRUCTION`, separate `ocr_exam` cache task) that
  explicitly transcribes the handwritten `CANDIDATE:` and `QUESTION:` header
  fields. On the real scan this captured `QUESTION: 5`, `QUESTION: 6`, …, which
  the previous prompt dropped. Essay booklets keep their own prompt/cache.
- `exam.py`: question headings now tolerate OCR punctuation variants such as
  `**Question:** 4` (`_HEADING_ANY` allows a short run of non-alphanumerics).
- `exam.py`: multi-page scans are split **page by page** (`segment_by_pages`): a
  page with a unique header number is trusted; a missing or repeated header is
  assigned by the model (`_resolve_page_question`, `page-resolved:` flag) with the
  weighted-keyword scorer (`_content_score`) as the offline fallback. Pages of the
  same question are concatenated; unmatched pages are reported.
- Real script `004.PDF` (5 pages, handwritten, no text layer) now marks end to
  end: page OCR → facts → rules → **15/100 (F)** on a genuinely weak script, with
  per-question attribution and every credited point quoting the candidate verbatim.
- **105 tests passing.**
