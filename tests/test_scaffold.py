"""M0 placeholder test: proves the test runner and package import work."""

from __future__ import annotations

import ees


def test_version() -> None:
    assert ees.__version__


def test_config_loads() -> None:
    from ees.config import load_config

    cfg = load_config()
    assert cfg.mode in {"hybrid", "rule_only", "llm_only"}
    assert cfg.kb_path.name == "kb"
