"""Streamlit UI (MVP plan component 8, milestone M8).

Five tabs: Marking, Pipeline, Knowledge, Evaluation, Exam (COM 226). Requires a
live LLM provider and an API key.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Allow running via `streamlit run app/streamlit_app.py` from a checkout.
SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parent
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ees import explainer  # noqa: E402
from ees.config import PROJECT_ROOT, load_config  # noqa: E402
from ees.data_io import load_sample_essays  # noqa: E402
from ees.document_ingest import IngestionError  # noqa: E402
from ees.exam import load_exam_paper, mark_exam, mark_exam_document  # noqa: E402
from ees.kb_loader import KnowledgeError, load_knowledge  # noqa: E402
from ees.llm_gateway import LLMGateway  # noqa: E402
from ees.pipeline import mark, mark_document  # noqa: E402
from ees.schemas import Fact, ProofNode  # noqa: E402

st.set_page_config(page_title="Explainable Essay-Marking Expert System", layout="wide")

SOURCE_BADGE = {"nlp": "🟦 NLP", "llm": "🟪 LLM", "rule": "🟨 RULE"}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_gateway(
    provider: str, model: str, base_url: str | None, api_key: str | None
):
    return LLMGateway(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def load_kb():
    """Load the KB fresh each run so Knowledge-tab edits take effect."""

    return load_knowledge(PROJECT_ROOT / "kb")


def _gateway_for(config):
    provider = config.llm.provider
    if not config.llm.api_key:
        st.error(
            f"Provider '{provider}' requires an API key. Set "
            f"`{config.llm.api_key_env}` (e.g. in the repo `.env`) and reload."
        )
        st.stop()
    return get_gateway(
        provider,
        config.llm.model,
        config.llm.base_url,
        config.llm.api_key,
    )


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def highlighted_essay(essay: str, facts: list[Fact]) -> str:
    spans: list[tuple[int, int, str]] = []
    for fact in facts:
        for span in fact.evidence:
            spans.append((span.start, span.end, fact.name))
    spans.sort()
    out: list[str] = []
    cursor = 0
    for start, end, label in spans:
        if start < cursor:
            continue
        out.append(html.escape(essay[cursor:start]))
        out.append(
            f'<mark title="{html.escape(label)}" style="background:#ffe08a">'
            f"{html.escape(essay[start:end])}</mark>"
        )
        cursor = end
    out.append(html.escape(essay[cursor:]))
    return (
        '<div style="white-space:pre-wrap;line-height:1.6;font-size:1.02rem">'
        + "".join(out)
        + "</div>"
    )


def proof_tree_markdown(nodes: list[ProofNode], depth: int = 0) -> str:
    lines: list[str] = []
    icon = {"score": "🎯", "rule": "📜", "fact": "🔹", "evidence": "❝"}
    for node in nodes:
        val = "" if node.value is None else f" = `{node.value}`"
        detail = f" — {node.detail}" if node.detail else ""
        lines.append(f"{'  ' * depth}- {icon.get(node.kind, '•')} **{node.label}**{val}{detail}")
        if node.children:
            lines.append(proof_tree_markdown(node.children, depth + 1))
    return "\n".join(lines)


def facts_dataframe(facts: list[Fact]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fact": f.name,
                "value": f.value,
                "source": SOURCE_BADGE.get(f.source, f.source),
                "confidence": f.confidence,
                "evidence": f.evidence[0].quote if f.evidence else "",
            }
            for f in facts
        ]
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
config = get_config()
kb = load_kb()

st.title("Explainable Hybrid Expert System for Marking Essays")
st.caption(
    "LLM extracts evidence; rule-based inference engine decides the mark. "
    "Every mark is traceable to a fired rule and a verbatim quote."
)

tab_mark, tab_pipe, tab_kb, tab_eval, tab_exam = st.tabs(
    ["✍️ Marking", "🔬 Pipeline", "📚 Knowledge", "📊 Evaluation", "🎓 Exam (COM 226)"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Marking
# --------------------------------------------------------------------------- #
with tab_mark:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sample = st.selectbox(
            "Task prompt", ["(custom)"] + [p.id for p in kb.prompts], index=0
        )
    with col2:
        mode = st.selectbox("Mode", ["hybrid", "rule_only", "llm_only"], index=0)
    with col3:
        explain_mode = st.toggle("LLM rewrite of feedback", value=False)

    loaded_prompt_id = None
    if sample != "(custom)":
        loaded_prompt_id = sample

    input_kind = st.radio(
        "Input type",
        ["✍️ Typed essay", "📄 Upload booklet (PDF / image)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_kind.startswith("✍️"):
        if sample != "(custom)" and st.button("Load a sample essay for this prompt"):
            essays = load_sample_essays(config)
            match = next((e for e in essays if e.prompt_id == sample), None)
            if match:
                st.session_state["essay_text"] = match.text

        essay_text = st.text_area(
            "Essay text", value=st.session_state.get("essay_text", ""), height=260
        )

        if st.button("✍️ Mark essay", type="primary"):
            if not essay_text.strip():
                st.warning("Please paste an essay first.")
            else:
                with st.spinner("Marking…"):
                    result = mark(
                        essay_text,
                        kb,
                        config,
                        _gateway_for(config),
                        mode=mode,
                        prompt_id=loaded_prompt_id,
                        explain_with_llm=explain_mode,
                    )
                st.session_state["result"] = result
                st.session_state["last"] = {
                    "essay": essay_text,
                    "prompt_id": loaded_prompt_id,
                    "mode": mode,
                    "explain": explain_mode,
                }
    else:
        if not config.ocr.enabled:
            st.warning("Scanned-booklet input is disabled in config (`ocr.enabled: false`).")
        uploaded = st.file_uploader(
            "Upload a scanned student booklet",
            type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"],
        )
        if uploaded is not None:
            st.caption(
                f"**{uploaded.name}** — {len(uploaded.getvalue()) / 1024:.0f} KB. "
                f"Will render at {config.ocr.dpi} dpi (max {config.ocr.max_pages} pages) "
                f"and read with `{config.ocr.vision_model}`."
            )
        if st.button("📄 Read pages & mark", type="primary"):
            if uploaded is None:
                st.warning("Please upload a PDF or image booklet first.")
            else:
                with st.spinner("Reading each page with the vision model…"):
                    try:
                        result = mark_document(
                            uploaded.getvalue(),
                            uploaded.name,
                            kb,
                            config,
                            _gateway_for(config),
                            mode=mode,
                            prompt_id=loaded_prompt_id,
                            explain_with_llm=explain_mode,
                        )
                    except IngestionError as exc:
                        result = None
                        st.error(f"Could not read the booklet: {exc}")
                if result is not None:
                    st.session_state["result"] = result
                    st.session_state["last"] = {
                        "essay": result.ocr_text,
                        "prompt_id": loaded_prompt_id,
                        "mode": mode,
                        "explain": explain_mode,
                        "input_type": "scan",
                    }

    result = st.session_state.get("result")
    if result is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{result.total:g} / {result.max_total:g}")
        c2.metric("Grade", result.grade)
        c3.metric("Mode", result.mode)
        if result.flags:
            st.warning("Review flags: " + ", ".join(result.flags))

        if result.input_type == "scan":
            st.info(
                f"Marked from a scanned booklet — {len(result.pages)} page(s) read "
                f"by `{result.vision_model}`. The mark is still produced only by the "
                "rule engine, from facts extracted out of the recognised text."
            )
            with st.expander("Recognised text (per page)", expanded=False):
                for pg in result.pages:
                    src = {"vision": "vision model", "embedded": "PDF text layer",
                           "error": "ERROR"}.get(pg.source, pg.source)
                    st.markdown(f"**Page {pg.index + 1}** · _{src}_")
                    st.text(pg.text or "(no text recognised)")

        st.subheader("Per-criterion scores")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "criterion": cs.name,
                        "level": cs.level,
                        "score": cs.score,
                        "max": cs.max_score,
                        "fired rules": ", ".join(cs.fired_rules),
                    }
                    for cs in result.criterion_scores
                ]
            ),
            use_container_width=True,
        )

        st.subheader("Essay with evidence highlighted")
        st.markdown(highlighted_essay(st.session_state["last"]["essay"], result.facts),
                    unsafe_allow_html=True)

        st.subheader("Student feedback")
        st.markdown(result.feedback or result.explanations and "\n".join(result.explanations) or "_none_")

        if result.counterfactuals:
            st.subheader("What would raise the mark")
            for tip in result.counterfactuals:
                st.markdown(f"- {tip}")

# --------------------------------------------------------------------------- #
# Tab 2: Pipeline
# --------------------------------------------------------------------------- #
with tab_pipe:
    result = st.session_state.get("result")
    if result is None:
        st.info("Mark an essay in the Marking tab to explore the pipeline.")
    else:
        st.markdown("**Components:** " + " → ".join(
            ["Knowledge Base", "Fact Extraction", "Working Memory", "Inference Engine",
             "Explanation", "Validation"]
        ))
        stages = st.selectbox(
            "Stage",
            (
                (["0. Page reading (OCR)"] if result.input_type == "scan" else [])
                + [
                    "1. Knowledge Base",
                    "2. Fact extraction",
                    "3. Working memory",
                    "4. Inference engine",
                    "5. Explanation",
                    "6. Validation",
                ]
            ),
        )

        if stages.startswith("0"):
            st.markdown(
                f"**Scanned booklet** — {len(result.pages)} page(s) read by "
                f"`{result.vision_model}` at {config.ocr.dpi} dpi."
            )
            for pg in result.pages:
                st.markdown(f"**Page {pg.index + 1}** · `{pg.source}`")
                st.text(pg.text or "(no text recognised)")
            st.markdown("**Assembled transcript** (this becomes the essay)")
            st.text(result.ocr_text)
        elif stages.startswith("1"):
            st.markdown(f"Rubric **{kb.rubric.rubric_id}** with {len(kb.rubric.criteria)} "
                        f"criteria; {len(kb.rules)} rules; KB version `{kb.version}`.")
            st.json(kb.rubric.model_dump())
        elif stages.startswith("2"):
            nlp = [f for f in result.facts if f.source == "nlp"]
            st.markdown("**Deterministic NLP features**")
            st.dataframe(facts_dataframe(nlp), use_container_width=True)
            st.markdown("**LLM structured extraction (JSON)**")
            st.json(result.extraction or {})
        elif stages.startswith("3"):
            st.markdown("**Working memory: all facts**")
            st.dataframe(facts_dataframe(result.facts), use_container_width=True)
        elif stages.startswith("4"):
            st.markdown(f"**Firing log** ({len(result.firing_log)} steps). "
                        "Step through the rule firings:")
            if result.firing_log:
                idx = st.slider("step", 1, len(result.firing_log),
                                value=st.session_state.get("step_idx", 1))
                st.session_state["step_idx"] = idx
                entry = result.firing_log[idx - 1]
                st.markdown(f"### Step {entry.step}: `{entry.rule_id}`")
                st.write("**Matched facts:**", ", ".join(entry.matched_facts) or "—")
                st.write("**Asserted:**", entry.asserted or "—")
                if entry.conflicts_resolved_against:
                    st.write("**Conflicts resolved against:**",
                             ", ".join(entry.conflicts_resolved_against))
                st.info(entry.explanation or "(no template)")
                st.markdown("**All steps**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"step": e.step, "rule": e.rule_id, "criterion": e.criterion,
                             "asserted": e.asserted, "losers": ", ".join(e.conflicts_resolved_against)}
                            for e in result.firing_log
                        ]
                    ),
                    use_container_width=True,
                )
        elif stages.startswith("5"):
            st.markdown("**Proof tree** (score ← rules ← facts ← evidence)")
            st.markdown(proof_tree_markdown(result.proof_tree))
            st.markdown("**Why not a higher level?**")
            q1, q2 = st.columns(2)
            crit = q1.selectbox("criterion", [c.id for c in kb.rubric.criteria])
            level = q2.number_input("level", min_value=1, max_value=4, value=4)
            st.info(explainer.why_not_level(kb, crit, int(level), result.facts))
        else:
            st.markdown("**Validation / consistency checks**")
            if result.flags:
                st.error("Flags: " + ", ".join(result.flags))
            else:
                st.success("All checks passed.")
            st.write("LLM-vs-rule disagreement:", result.disagreement)

# --------------------------------------------------------------------------- #
# Tab 3: Knowledge
# --------------------------------------------------------------------------- #
with tab_kb:
    st.markdown("Edit the knowledge base as data. Schema validation runs on save.")
    which = st.selectbox("File", ["rubric.yaml", "rules.yaml", "content_points.yaml", "policies.yaml"])
    path = PROJECT_ROOT / "kb" / which
    text = st.text_area("YAML", value=path.read_text(encoding="utf-8"), height=380, key=f"kb_{which}")

    c1, c2 = st.columns(2)
    if c1.button("💾 Save & validate"):
        path.write_text(text, encoding="utf-8")
        try:
            load_knowledge(PROJECT_ROOT / "kb")
            st.success("Saved and validated. New KB version applied.")
        except KnowledgeError as exc:
            st.error(f"Validation failed: {exc}")
    if c2.button("🔁 Re-mark last essay with this change"):
        last = st.session_state.get("last")
        if not last:
            st.warning("Mark an essay first.")
        else:
            before = st.session_state.get("result")
            after = mark(
                last["essay"], kb, config, _gateway_for(config),
                mode=last["mode"], prompt_id=last["prompt_id"],
            )
            st.session_state["result"] = after
            delta = after.total - (before.total if before else 0)
            st.success(f"New total: {after.total:g} (change {delta:+g}). "
                       "See the Pipeline tab for the updated trace.")

# --------------------------------------------------------------------------- #
# Tab 4: Evaluation
# --------------------------------------------------------------------------- #
with tab_eval:
    st.markdown("Results from `make eval`. Regenerate with the button below.")
    results_csv = PROJECT_ROOT / "eval" / "results.csv"
    report_md = PROJECT_ROOT / "eval" / "report.md"

    if st.button("▶️ Run evaluation now (may take ~1 min)"):
        with st.spinner("Running evaluation across the sample set…"):
            from importlib import import_module

            eval_mod = import_module("eval.run_eval")
            eval_mod.run()

    if results_csv.exists():
        df = pd.read_csv(results_csv)
        modes = ["rule_only", "llm_only", "hybrid"]
        gold = df[df["mode"] == "hybrid"]["gold_total"].tolist()
        rows = []
        for m in modes:
            sub = df[df["mode"] == m]
            from ees.metrics import mae, quadratic_weighted_kappa

            rows.append(
                {
                    "mode": m,
                    "QWK": round(quadratic_weighted_kappa(gold, sub["pred_total"].tolist(), 0, 40), 3),
                    "MAE": round(mae(gold, sub["pred_total"].tolist()), 3),
                }
            )
        st.subheader("Three-mode comparison")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.bar_chart(pd.DataFrame(rows).set_index("mode")[["QWK", "MAE"]])

        st.subheader("Flagged / disagreeing cases (hybrid)")
        hybrid = df[df["mode"] == "hybrid"]
        st.dataframe(
            hybrid[["essay_id", "gold_total", "pred_total", "n_flags"]].sort_values("n_flags", ascending=False),
            use_container_width=True,
        )
        if report_md.exists():
            with st.expander("Full report (eval/report.md)"):
                st.markdown(report_md.read_text(encoding="utf-8"))
    else:
        st.info("No results yet. Click 'Run evaluation now'.")

# --------------------------------------------------------------------------- #
# Tab 5: Exam marking (COM 226 point-based scheme)
# --------------------------------------------------------------------------- #
with tab_exam:
    st.markdown(
        "Mark an exam answer script against the **COM 226 – File Organization and "
        "Management** scheme. Each covered marking point adds its marks; the best "
        "**five** answered questions are totalled out of **100**. The model only "
        "detects which points are covered (with quotes) — the rule engine adds the marks."
    )
    exam_dir = PROJECT_ROOT / "kb" / "exams" / "com226"
    try:
        exam_paper = load_exam_paper(exam_dir / "questions.yaml")
        exam_kb = load_knowledge(exam_dir)
    except (KnowledgeError, FileNotFoundError) as exc:  # pragma: no cover - defensive
        exam_paper = exam_kb = None
        st.error(f"Could not load the COM 226 scheme: {exc}")

    if exam_paper is not None:
        e1, e2 = st.columns([3, 1])
        with e2:
            exam_mode = st.selectbox("Mode", ["hybrid", "rule_only", "llm_only"], key="exam_mode")

        e_kind = st.radio(
            "Exam input",
            ["⌨️ Typed answers", "📄 Upload booklet (PDF / image)"],
            horizontal=True,
            label_visibility="collapsed",
            key="exam_kind",
        )

        if e_kind.startswith("⌨️"):
            if st.button("Load the sample COM 226 script"):
                st.session_state["exam_script"] = (
                    PROJECT_ROOT / "data" / "exam" / "com226" / "sample_student_script.txt"
                ).read_text(encoding="utf-8")
            script_text = st.text_area(
                "Candidate answers (label each as QUESTION ONE, Q2, 3, …)",
                height=260,
                key="exam_script",
            )
            if st.button("🎓 Mark exam", type="primary"):
                if not script_text.strip():
                    st.warning("Please paste the candidate's answers first.")
                else:
                    with st.spinner("Marking exam…"):
                        st.session_state["exam_result"] = mark_exam(
                            script_text, exam_paper, exam_kb, config,
                            _gateway_for(config), mode=exam_mode,
                        )
        else:
            ex_upload = st.file_uploader(
                "Upload a scanned answer script",
                type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"],
                key="exam_upload",
            )
            if st.button("📄 Read pages & mark exam", type="primary"):
                if ex_upload is None:
                    st.warning("Please upload a PDF or image first.")
                else:
                    with st.spinner("Reading each page with the vision model…"):
                        try:
                            st.session_state["exam_result"] = mark_exam_document(
                                ex_upload.getvalue(), ex_upload.name, exam_paper,
                                exam_kb, config, _gateway_for(config), mode=exam_mode,
                            )
                        except IngestionError as exc:
                            st.error(f"Could not read the booklet: {exc}")

        exam_result = st.session_state.get("exam_result")
        if exam_result is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", f"{exam_result.total:g} / {exam_result.max_total:g}")
            c2.metric("Grade", exam_result.grade)
            c3.metric("Answered", f"{exam_result.answer_count} (best {exam_result.best_n})")
            if exam_result.flags:
                st.info("Flags: " + ", ".join(exam_result.flags))
            if exam_result.input_type == "scan":
                st.info(
                    f"Marked from {len(exam_result.pages)} scanned page(s) read by "
                    f"`{exam_result.vision_model}`."
                )

            st.subheader("Per-question result")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "question": f"Q{q.number}",
                            "title": q.title,
                            "marks": q.raw_marks,
                            "max": q.max_marks,
                            "answered": q.answered,
                            "counted": q.counts,
                        }
                        for q in exam_result.questions
                    ]
                ),
                use_container_width=True,
            )

            st.subheader("Marking-point breakdown")
            for q in exam_result.questions:
                if not q.answered:
                    continue
                with st.expander(
                    f"Q{q.number}. {q.title} — {q.raw_marks:g}/{q.max_marks:g}", expanded=False
                ):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "point": p.point_id,
                                    "awarded": p.marks_awarded,
                                    "of": p.marks_available,
                                    "covered": p.covered,
                                    "method": p.method,
                                    "evidence": p.quote,
                                }
                                for p in q.points
                            ]
                        ),
                        use_container_width=True,
                    )
            st.caption(
                "Marks come from fired rules (`add` actions) in the inference engine; the "
                "model only detected coverage and quoted the script."
            )
