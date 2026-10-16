"""Shared test fixtures.

There is no mock or offline mode: tests that exercise the LLM use the configured
live provider and are skipped when no API key is present. Pure-unit tests stub the
HTTP boundary (``ees.llm_gateway._http_post_json``) via monkeypatch and need no key.
"""

from __future__ import annotations

import pytest

from ees.config import load_config
from ees.llm_gateway import LLMGateway


@pytest.fixture(scope="session")
def live_gateway() -> LLMGateway:
    """The configured live provider, or skip when no API key is available."""

    config = load_config()
    if not config.llm.api_key:
        pytest.skip(
            f"live provider '{config.llm.provider}' requires {config.llm.api_key_env}"
        )
    return LLMGateway(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )
