# Architecture

This document describes the components of the Explainable Hybrid Expert System
for Marking Student Essays, the data that flows between them, and how each maps
to a classic expert-system role.

## Design principle

The LLM **perceives** (extracts facts and quotes) and **verbalizes** (rewrites the
rule trace into feedback). The **rule-based inference engine decides** the score.
This separation is what makes every mark explainable: the reasons shown to the
user are exactly the rules that fired, backed by verbatim evidence.

## Components

| # | Component | Module | Responsibility |
|---|-----------|--------|----------------|
| 1 | Knowledge Base | `kb/*.yaml`, `kb_loader.py` | Rubric, rules, content points, policies as editable data; schema + consistency validation with file/line errors; content-hash versioning |
| 2 | Knowledge Acquisition | `app/streamlit_app.py` (Knowledge tab) | Form/YAML editor, validation messages, save new version, re-mark to see the diff |
| 3 | Fact Extraction | `features_nlp.py` (deterministic), `extractor_llm.py` (LLM) | NLP: counts, readability, connectives, spelling rate, injection detection. LLM: thesis, arguments, counterargument, coherence, on-topic, key points — each with a verbatim quote |
| 4 | Working Memory | `working_memory.py` | Typed fact base for one essay; every fact carries `source`, `confidence`, `evidence` spans |
| 5 | Inference Engine | `inference_engine.py` | Forward chaining to a fixpoint; `all/any/not` and comparison operators; priority conflict resolution (ties by specificity), full firing log; actions `assert/score/penalty/flag`; max-cycle guard |
| 6 | Explanation Facility | `explainer.py` | Proof tree (score ← rules ← facts ← evidence), deterministic template explanations, "why not level X?", counterfactuals, optional strict LLM rewrite with a citation post-check |
| 7 | Validation & Consistency Checker | `validator.py` | Bounds, level-range consistency, LLM-vs-rule disagreement, low confidence, borderline, short, off-topic, prompt injection; routes to human review |
| 8 | User Interface | `app/streamlit_app.py` | Marking, Pipeline (stepper), Knowledge, Evaluation tabs |
| 9 | LLM Gateway | `llm_gateway.py` | Single wrapper; live providers (`nvidia`, `anthropic`); retries with backoff; multimodal vision calls for scanned pages |

## Data flow

```
Essay + Task prompt
   │
   ▼
[Fact Extraction] ─► NLP features ─┐
   │                               ├─► [Working Memory: facts + evidence spans]
   └─► LLM structured extraction ─┘                │
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

## Schemas (the contract)

Defined in `schemas.py` and implemented exactly:

* `Fact { name, value, source ∈ {nlp,llm,rule}, confidence?, evidence: [Span] }`
* `Condition` — a leaf `{fact, op, value}` or a group `{all|any|not}`
* `Rule { id, criterion?, priority, if, then: [RuleAction], explain }`
* `Rubric { rubric_id, max_total, criteria: [RubricCriterion{id,name,max_score,levels:[RubricLevel{level,range,descriptor}]}] }`
* `LLMExtraction` — the JSON the LLM must return (facts only, never scores)
* `FiringLogEntry { step, rule_id, matched_facts, asserted, conflicts_resolved_against, timestamp }`
* `MarkingResult` — everything the UI/eval needs for one run

## Marking modes

* `hybrid` (default) — NLP facts + LLM facts → inference engine → validation →
  explanation. The LLM never sets a score.
* `rule_only` — deterministic NLP facts only. Content-level rules that need the
  LLM's `thesis_present`/argument facts cannot fire, so this baseline is
  deliberately weak; it shows what pure rules can do.
* `llm_only` — the LLM returns rubric scores directly. Baseline only.

## Determinism

* Engine iterates rules sorted by id and fires to a fixpoint; identical facts
  produce an identical log (property-tested).
* Temperature 0, JSON-schema validation and verbatim quote checks bound LLM
  variability.
* There is no mock or offline mode: the gateway always calls a live provider, and
  a missing API key is an error.
