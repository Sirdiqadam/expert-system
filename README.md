# Explainable Hybrid Expert System for Marking Student Essays

A small, runnable Python/Streamlit MVP of a **hybrid LLM + rule-based expert
system** that marks student essays with **explainable, traceable** scores.

> **Core design principle:** the LLM *perceives* (extracts evidence from the
> text) and *verbalizes* (turns the rule trace into readable feedback). The
> **rule-based inference engine decides** (scores and applies marking policy).
> The LLM never produces the final score in `hybrid` mode. Every mark is
> traceable to a fired rule and a verbatim evidence quote.

Working title: *Knowledge-Based Expert System for Automated Assessment and
Explainable Marking of Students' Writing Examinations.*

---

## Quickstart (< 10 minutes)

Requires **Python 3.11+**. The demo runs **fully offline** with no API key
(cached/heuristic mock mode).

```bash
# Linux/macOS
make setup && make demo

# Windows (no GNU Make needed)
make.bat setup
make.bat demo

# or manually
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,ui,nlp]"
.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

Then open the Streamlit URL and use the five tabs: **Marking**, **Pipeline**,
**Knowledge**, **Evaluation**, **Exam (COM 226)**.

To run the test suite or the evaluation harness:

```bash
make test      # pytest
make eval      # writes eval/report.md + eval/results.csv
```

### Using a real LLM (optional)

The default live provider is **NVIDIA NIM** (OpenAI-compatible). Copy
`.env.example` to `.env` and set `NVIDIA_API_KEY`. Provider/model/endpoint live in
`config/config.yaml` (`llm.provider`, `llm.model`, `llm.base_url`); override with
`EES_LLM_PROVIDER` / `EES_LLM_MODEL`. Anthropic is also supported
(`llm.provider: anthropic`, `ANTHROPIC_API_KEY`).

All calls go through `llm_gateway.py` at temperature 0 and are cached by
`hash(model + prompt + essay [+ page images])`, so subsequent runs are offline.
With no key present the app falls back to the offline mock automatically.

---

## Scanned booklets (PDF / images)

Besides typing/pasting an essay you can **upload a scanned student booklet** — a
PDF (one or many pages) or an image. Each page is rasterised to an image, read by
a **vision model** (`ocr.vision_model`, default `meta/llama-3.2-11b-vision-instruct`)
and transcribed to text; that transcript is then marked exactly like a typed
essay. The score is still produced **only** by the rule engine, and every evidence
quote is verified verbatim against the recognised text.

In the **Marking** tab choose *📄 Upload booklet (PDF / image)*, select a task
prompt, and click **Read pages & mark**. The UI shows the per-page transcript and
its provenance (`vision` / `cache` / `embedded` / `mock` / `error`).

Three ready-made booklets live in `data/sample_booklets/` and are pre-cached, so
the scanned flow also works **offline**. Regenerate them with:

```bash
python tools/make_sample_booklets.py
```

```python
# programmatic use
from ees.pipeline import mark_document
result = mark_document(pdf_bytes, "booklet.pdf", kb, config, gateway, prompt_id="school_uniforms")
result.total, result.ocr_text, [(p.index, p.source) for p in result.pages]
```

Tune OCR in `config/config.yaml` under `ocr:` (`vision_model`, `dpi`,
`max_pages`, `min_embedded_chars`). Digital PDFs with a real text layer skip the
vision call entirely.

---

## Exam marking: COM 226 (point-based scheme)

The same engine also marks **question-and-answer exams**, where marks are awarded
per *marking point* rather than per rubric level. The first scheme shipped is
**COM 226 – File Organization and Management** (ND, Federal Polytechnic
Damaturu): candidates answer any **FIVE** of seven questions, 20 marks each,
total **100**.

* Scheme source of truth: `kb/exams/com226/questions.yaml` — each question → parts
  → marking points, with the model answer, marks and offline keywords.
* Derived machine KB: `rubric.yaml` (one point-based criterion per question,
  `aggregate: best_n`, `best_n: 5`) and `rules.yaml` (one `add` rule per point).
  Regenerate with `make exam-kb` (`python tools/build_exam_kb.py com226`).
* Marker: `src/ees/exam.py`. The LLM **only detects which points are covered**
  (with a verbatim quote per point); the inference engine adds the marks through
  `add` rules. Offline it falls back to a deterministic keyword matcher.

```bash
make exam      # mark the sample script (offline unless an API key is set)
make exam-kb   # regenerate rubric.yaml + rules.yaml from questions.yaml
```

```python
from ees.exam import load_exam_paper, mark_exam
paper = load_exam_paper("kb/exams/com226/questions.yaml")
result = mark_exam(script_text, paper, exam_kb, config, gateway, mode="hybrid")
result.total, result.grade, [(q.number, q.raw_marks, q.counts) for q in result.questions]
```

Scanned answer scripts are supported too (`mark_exam_document`), and there is a
**🎓 Exam (COM 226)** tab in the app. The committed sample script
(`data/exam/com226/sample_student_script.txt`) marks to **92/100 (A)** with the
live model and **89/100 (A)** with the deterministic keyword fallback (slightly
stricter: it does not credit the buffer *overlap* function or the GFS *rotation*
step the model accepted).

**Scanned scripts.** An uploaded booklet is read page by page by a vision model
using a **header-aware prompt** that also transcribes the handwritten
`CANDIDATE:` and `QUESTION:` fields. Because scans are page-structured,
`segment_by_pages` assigns **one page at a time**: a page with a unique header
number is trusted; a missing or repeated header is resolved by the model (offline:
a weighted-keyword match), and the choice is recorded as a `page-resolved:` flag.
This means a real 5-page handwritten script (`004.PDF`) marks end to end —
page images → vision transcript → facts → `add` rules → **15/100 (F)** — with each
credited point quoting the candidate verbatim and the deliberately ambiguous page
flagged for review. Switching the vision model is a config change
(`ocr.vision_model`).

---

## Architecture

Nine classic expert-system components, each a separate module:

| # | Component | Module |
|---|-----------|--------|
| 1 | Knowledge Base | `kb/*.yaml` + `kb_loader.py` |
| 2 | Knowledge Acquisition | Knowledge tab in `app/streamlit_app.py` |
| 3 | Fact Extraction (Perception) | `features_nlp.py` (NLP) + `extractor_llm.py` (LLM) |
| 4 | Working Memory (Fact Base) | `working_memory.py` |
| 5 | Inference Engine | `inference_engine.py` (forward chaining) |
| 6 | Explanation Facility | `explainer.py` |
| 7 | Validation & Consistency Checker | `validator.py` |
| 8 | User Interface | `app/streamlit_app.py` (Streamlit) |
| 9 | LLM Gateway | `llm_gateway.py` |

Scanned-booklet input adds a **vision-OCR pre-stage** (`document_ingest.py` +
`ocr_llm.py`) in front of Fact Extraction; it changes nothing downstream.

### Data flow

```
Essay text  ──┐            Booklet (PDF / image)
              │                    │
              │                    ▼
              │        [Vision OCR: page image → transcript]
              │                    │
              └──────────┬─────────┘
                         ▼
[Fact Extraction] ─► NLP features ─┐
        │                           ├─► [Working Memory: facts + evidence spans]
        └─► LLM structured extract ─┘              │
                                                   ▼
[Knowledge Base: rubric, rules] ──► [Inference Engine: forward chaining]
                                                   │ firing log
                                                   ▼
                          [Scores per criterion + total + grade]
                                                   │
                          [Validation / Consistency Checker] ─► review flags
                                                   │
                          [Explanation Facility: proof tree → feedback]
                                                   ▼
                        UI: mark, evidence-highlighted essay, why-trace
```

### The hybrid principle

Facts, not scores, come from the LLM:

```
thesis_present = true  (quote: "…")
num_supporting_arguments = 3
counterargument_rebutted = false
```

Rules turn facts into marks:

```
IF thesis_present AND num_supporting_arguments >= 3 AND counterargument_rebutted
THEN content_level = 4  (score content within level 4)
```

The explanation is generated from the firing log, so the reasons given are
exactly the reasons used.

---

## Repository layout

```
essay-expert-system/
├─ README.md, CHANGELOG.md, pyproject.toml, Makefile, make.bat, .env.example
├─ config/config.yaml
├─ kb/                      # editable knowledge (rubric, rules, content points, policies)
├─ kb/exams/com226/         # COM 226 scheme: questions.yaml + derived rubric/rules
├─ src/ees/                 # the 9 components + schemas, config, metrics, data_io
├─ src/ees/document_ingest.py  # PDF/image → page images
├─ src/ees/ocr_llm.py          # page → transcript (vision)
├─ src/ees/exam.py             # point-based exam marking (COM 226)
├─ app/streamlit_app.py     # the 5-tab UI
├─ data/sample_essays/      # 12 sample essays across quality levels
├─ data/sample_booklets/    # sample scanned booklets (PDF) for the OCR demo
├─ data/exam/com226/        # sample exam answer script
├─ data/gold/human_scores.csv
├─ data/cache/              # LLM response cache (offline demo)
├─ data/runs/               # JSON marking traces (generated)
├─ eval/run_eval.py         # evaluation harness -> eval/report.md
├─ tools/                   # helper scripts (make_sample_booklets.py, build_exam_kb.py)
├─ tests/                   # unit + integration + golden + safety tests
└─ docs/                    # architecture, demo script, research mapping, decisions
```

---

## 5-minute demo script

1. Show the architecture diagram and name each component.
2. Mark a mid-quality essay in **hybrid** mode; open the **Pipeline** tab and walk
   through the six stages.
3. Switch the Marking tab to **📄 Upload booklet**, upload
   `data/sample_booklets/essay_04_booklet.pdf`, choose `school_uniforms`, and mark.
   Show the **per-page transcript** and that the mark is identical in structure to
   the typed path — the LLM read the page, the rules decided the mark.
4. Step through the **inference engine** and show a conflict being resolved.
5. Ask **"Why not level 4 for Content?"** and show the counterfactual.
6. In the **Knowledge** tab, change one threshold (e.g. arguments needed for
   level 4 from 3 to 2), re-mark, and show the mark and trace change.
7. Paste an essay containing a **prompt-injection** sentence and show it is
   flagged and ignored.
8. Open the **Evaluation** tab and show the three-mode comparison.

---

## Evaluation

`make eval` computes, per mode (`rule_only`, `llm_only`, `hybrid`): **QWK**
(primary), **MAE**, exact and adjacent agreement (overall and per criterion),
bootstrap confidence intervals, a confusion matrix, explanation faithfulness
rate, and latency/token cost. Results are written to `eval/report.md` and
`eval/results.csv`. Human inter-rater agreement is reported as the ceiling
reference.

The report is produced by running on the real sample set in the repo — **no
numbers are fabricated**.

---

## Testing strategy

* Unit tests: engine operators, conflict resolution, KB validation, quote
  verification, feature functions, metrics.
* Golden tests: fixed essays with fixed extraction produce fixed scores/traces.
* Property/safety tests: monotonicity (an extra supporting argument never lowers
  the Content score), determinism, scores always in bounds, prompt injection
  ignored, hybrid rejects LLM-provided scores, explanations cannot cite unknown
  rule ids.
* Integration tests: full pipeline in mock mode across all sample essays.
* Scanned-booklet tests: PDF/multi-page-TIFF ingestion, page limits, headerless
  sniffing, vision error handling, OCR cache replay, and end-to-end
  `mark_document` on a generated booklet.
* Exam tests: point-based scheme/`add` action, best-N aggregation, the question
  splitter, the offline keyword matcher, and scanned-**page** attribution
  (header trust, duplicate-header re-resolution, content fallback, unassigned
  pages).
* Coverage target: ≥ 80% on `src/ees/`.

---

## Risks, ethics and limitations

* **LLM variability** — mitigated by temperature 0, caching, JSON-schema
  validation and verbatim quote verification; residual variance can be checked by
  re-running extraction several times.
* **Bias and fairness** — language-background and dialect effects are real; the
  fairness probe compares score distributions across essay-length bands and
  flagged/borderline scripts always go to human review.
* **Human in the loop** — the system supports marking and does **not** replace the
  examiner; borderline and flagged scripts always go to review.
* **Privacy** — no student-identifying data should be put in prompts; anonymize
  before upload; essay text is sent to an external LLM API in `hybrid` and
  `llm_only` modes (not in `rule_only`).
* **Overclaiming** — results on the small sample set are indicative only; the real
  evaluation requires a larger local dataset.

---

## Possible extensions after the MVP

Retrieval of model answers/exemplars (RAG) feeding the fact layer; learning rules
from lecturers' past marks; additional genres and question types; uncertainty
calibration and active learning; LMS integration with an audit log.
