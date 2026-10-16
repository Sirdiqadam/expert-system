# Research mapping

Links each system component to the research questions the MVP must answer and to
the evaluation that supports them.

## RQ1 — Does hybrid agree with human markers as well as/​better than baselines?

**Metric:** Quadratic Weighted Kappa (primary), MAE, exact/adjacent agreement.

**Supported by**

* `eval/run_eval.py` runs all three modes (`rule_only`, `llm_only`, `hybrid`) on
  the sample set and reports QWK/MAE/agreement overall and per criterion, with
  bootstrap confidence intervals.
* Human inter-rater agreement is reported as the ceiling reference.
* `data/gold/human_scores.csv` stores two human marks and the adjudicated mark.

**Design features that help**

* The rule layer constrains scores to rubric level ranges (`inference_engine`),
  reducing the LLM's tendency to score holistically.
* Penalties and flags apply marking policy consistently.

## RQ2 — Are the explanations faithful?

**Metric:** automated faithfulness pass rate; human ratings of clarity/usefulness.

**Supported by**

* `explainer.build_proof_tree` constructs score ← rules ← facts ← evidence from
  the firing log; explanations are generated from that tree, not free-form.
* `explainer.verify_feedback` checks that every `[R-...]` citation exists in the
  firing log. The eval harness reports the faithfulness rate per mode.
* `tests/test_explainer.py` proves explanations never cite a fact absent from
  working memory or an unknown rule id.
* The optional LLM rewrite is a strict transformation of the tree and falls back
  to the deterministic template if the post-check fails.

## RQ3 — Is the system auditable and editable by non-programmers?

**Supported by**

* Knowledge is data (`kb/*.yaml`) validated by `kb_loader` with errors naming the
  file and line.
* The Knowledge tab lets a lecturer edit rules/rubric and re-mark to see the mark
  and trace change, with the changed rule shown as the one that fired.
* KB content-hash versioning records which knowledge produced a result.

## Evaluation plan coverage

| Plan item | Where |
|---|---|
| QWK / MAE / exact / adjacent, per criterion and overall | `eval/run_eval.py`, `ees/metrics.py` |
| Bootstrap CIs, confusion matrix | `ees/metrics.py`, `eval/report.md` |
| Faithfulness rate | `eval/run_eval.py` |
| Latency & token cost | `eval/report.md` (per-mode median latency, tokens) |
| Fairness probe (length bands) | `eval/report.md` |
| Ablations rule_only / llm_only / hybrid | three modes throughout |
| Public dataset adapter | `ees/data_io.py` (`load_sample_essays` shows the interface; adapt a loader to the same `SampleEssay` shape) |
| Local 100+ essay set | not included (needs ethics approval); load via the same adapter |

## Limitations of the current evidence

The sample set is 12 hand-made essays scored by the team. It is enough to
exercise every component and produce an indicative report, but **not** enough to
claim general marker agreement. The real evaluation requires the larger,
ethically-approved local dataset described in the plan (Section 11).
