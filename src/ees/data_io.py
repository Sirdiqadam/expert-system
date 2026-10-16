"""Loading sample essays and human gold scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AppConfig


@dataclass
class SampleEssay:
    """One sample essay with its task prompt and human marks."""

    essay_id: str
    prompt_id: str
    quality: str
    text: str
    adjudicated_total: float
    human_totals: list[float]
    criterion_scores: dict[str, float]


def load_sample_essays(config: AppConfig) -> list[SampleEssay]:
    """Load every ``data/sample_essays/*.txt`` with its gold row."""

    gold = pd.read_csv(config.gold_path)
    rows = {str(r["essay_id"]): r for _, r in gold.iterrows()}
    essays: list[SampleEssay] = []
    for path in sorted(config.sample_path.glob("*.txt")):
        essay_id = path.stem
        row = rows.get(essay_id)
        if row is None:
            continue
        criterion = {
            c: float(row[c])
            for c in ("content", "organization", "language", "mechanics")
            if c in row
        }
        essays.append(
            SampleEssay(
                essay_id=essay_id,
                prompt_id=str(row["prompt_id"]),
                quality=str(row.get("quality", "")),
                text=path.read_text(encoding="utf-8").strip(),
                adjudicated_total=float(row["adjudicated_total"]),
                human_totals=[float(row["human1_total"]), float(row["human2_total"])],
                criterion_scores=criterion,
            )
        )
    return essays


def load_gold_frame(config: AppConfig) -> pd.DataFrame:
    return pd.read_csv(config.gold_path)
