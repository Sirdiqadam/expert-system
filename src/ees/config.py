"""Configuration loading for the essay expert system.

Reads ``config/config.yaml`` into a validated :class:`AppConfig` and resolves
all documented paths relative to the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# src/ees/config.py -> parents: [ees, src, <repo root>]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class LLMConfig(BaseModel):
    """Settings for the LLM gateway."""

    provider: str = "nvidia"  # nvidia | anthropic
    model: str = "stepfun-ai/step-5-preview"
    temperature: float = 0.0
    max_retries: int = 3
    max_tokens: int = 8192
    api_key_env: str = "NVIDIA_API_KEY"
    base_url: str | None = "https://integrate.api.nvidia.com/v1"

    @property
    def api_key(self) -> str | None:
        """Return the API key from the environment, or ``None`` if unset."""

        key = os.environ.get(self.api_key_env, "").strip()
        return key or None


class OCRConfig(BaseModel):
    """Settings for scanned-booklet ingestion and vision transcription."""

    enabled: bool = True
    vision_model: str = "meta/llama-3.2-11b-vision-instruct"
    dpi: int = 150
    max_pages: int = 30
    min_embedded_chars: int = 40


class PathsConfig(BaseModel):
    """Filesystem locations used across the app (relative to repo root)."""

    kb_dir: str = "kb"
    runs_dir: str = "data/runs"
    sample_dir: str = "data/sample_essays"
    gold_csv: str = "data/gold/human_scores.csv"


class ThresholdsConfig(BaseModel):
    """Policy thresholds used by the validator."""

    borderline_margin: int = 1
    disagreement_threshold: float = 5
    low_confidence: float = 0.5
    min_words: int = 100


class AppConfig(BaseModel):
    """Top-level application configuration."""

    mode: str = "hybrid"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve *relative* against the repository root."""

        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def kb_path(self) -> Path:
        return self.resolve(self.paths.kb_dir)

    @property
    def runs_path(self) -> Path:
        return self.resolve(self.paths.runs_dir)

    @property
    def sample_path(self) -> Path:
        return self.resolve(self.paths.sample_dir)

    @property
    def gold_path(self) -> Path:
        return self.resolve(self.paths.gold_csv)


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: set os.environ from KEY=VALUE lines (no override)."""

    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate configuration from *path* (default ``config/config.yaml``).

    A ``.env`` file at the repo root is read first (without overriding real env
    vars). Environment variables ``EES_MODE``, ``EES_LLM_PROVIDER`` and
    ``EES_LLM_MODEL`` override their YAML counterparts when present.
    """

    _load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    if os.environ.get("EES_MODE"):
        raw["mode"] = os.environ["EES_MODE"]
    llm = raw.setdefault("llm", {})
    if os.environ.get("EES_LLM_PROVIDER"):
        llm["provider"] = os.environ["EES_LLM_PROVIDER"]
    if os.environ.get("EES_LLM_MODEL"):
        llm["model"] = os.environ["EES_LLM_MODEL"]

    return AppConfig.model_validate(raw)
