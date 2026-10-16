"""Generate the machine KB (rubric.yaml, rules.yaml, ...) for an exam paper.

The single source of truth is ``kb/exams/<id>/questions.yaml`` (hand-authored from
the marking scheme). This script derives:

* ``rubric.yaml``  - one point-based criterion per question, aggregate = best_n
* ``rules.yaml``   - one ``add`` rule per marking point (fires on ``covered_<point>``)
* ``content_points.yaml`` / ``policies.yaml`` - exam defaults

Usage:
    python tools/build_exam_kb.py com226
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ees.schemas import ExamPaper  # noqa: E402


def _dump(obj: object) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=100)


def build(exam_id: str = "com226") -> None:
    kb_dir = ROOT / "kb" / "exams" / exam_id
    paper = ExamPaper.model_validate(
        yaml.safe_load((kb_dir / "questions.yaml").read_text(encoding="utf-8"))
    )

    rubric = {
        "rubric_id": f"{exam_id}_exam_v1",
        "max_total": int(paper.total_marks),
        "aggregate": "best_n",
        "best_n": paper.answer_count,
        "criteria": [
            {
                "id": q.id,
                "name": f"Q{q.number}. {q.title}",
                "max_score": int(q.max_marks),
                "mode": "points",
            }
            for q in paper.questions
        ],
    }
    (kb_dir / "rubric.yaml").write_text(_dump(rubric), encoding="utf-8")

    rules = []
    for q in paper.questions:
        for part, pt in q.all_points():
            label = f"{part.label or ''}".strip()
            rules.append(
                {
                    "id": f"R-{pt.id.upper().replace('_', '-')}",
                    "criterion": q.id,
                    "description": f"{label} {pt.answer}".strip(),
                    "if": {"fact": f"covered_{pt.id}", "op": "==", "value": True},
                    "then": [{"action": "add", "criterion": q.id, "amount": pt.marks}],
                    "explain": f"Q{q.number}{label}: {pt.answer} (+{pt.marks:g})".strip(),
                }
            )
    (kb_dir / "rules.yaml").write_text(_dump(rules), encoding="utf-8")

    (kb_dir / "content_points.yaml").write_text(
        _dump({"prompts": [], "content_points": []}), encoding="utf-8"
    )
    (kb_dir / "policies.yaml").write_text(
        _dump(
            {
                "late_penalty_enabled": False,
                "late_penalty_per_day": 0.0,
                "min_words": 0,
                "max_words": None,
                "borderline_margin": 2,
            }
        ),
        encoding="utf-8",
    )
    print(
        f"{exam_id}: {len(rules)} point-rules across {len(paper.questions)} questions "
        f"(best {paper.answer_count} of {len(paper.questions)} -> {paper.total_marks:g})"
    )


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "com226")
