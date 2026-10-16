"""Tests for the LLM gateway (caching, providers, offline behavior)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ees.llm_gateway import LLMGateway, LLMGatewayError


def test_mock_generates_and_caches(tmp_path: Path) -> None:
    gw = LLMGateway(provider="mock", cache_dir=tmp_path)
    r1 = gw.complete(task="t", system="sys", user="hello", mock=lambda: '{"ok": true}')
    assert r1.from_cache is False
    assert r1.provider == "mock"
    assert r1.text == '{"ok": true}'

    r2 = gw.complete(task="t", system="sys", user="hello", mock=lambda: '{"ok": false}')
    assert r2.from_cache is True
    assert r2.text == '{"ok": true}'
    assert (tmp_path / "mock").exists()


def test_cache_key_is_stable_across_instances(tmp_path: Path) -> None:
    a = LLMGateway(provider="mock", cache_dir=tmp_path)
    a.complete(task="t", system="s", user="u", mock=lambda: "resp")
    b = LLMGateway(provider="mock", cache_dir=tmp_path)
    r = b.complete(task="t", system="s", user="u", mock=lambda: "different")
    assert r.from_cache and r.text == "resp"


def test_different_task_does_not_collide(tmp_path: Path) -> None:
    gw = LLMGateway(provider="mock", cache_dir=tmp_path)
    gw.complete(task="a", system="s", user="u", mock=lambda: "A")
    r = gw.complete(task="b", system="s", user="u", mock=lambda: "B")
    assert r.text == "B"


def test_anthropic_without_key_raises(tmp_path: Path) -> None:
    gw = LLMGateway(provider="anthropic", cache_dir=tmp_path, api_key=None)
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")


def test_no_cache_no_mock_raises(tmp_path: Path) -> None:
    gw = LLMGateway(provider="mock", cache_dir=tmp_path)
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")


def test_nvidia_provider_builds_request_and_caches(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, headers, payload, timeout=120):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "hello from nvidia"}}]}

    monkeypatch.setattr("ees.llm_gateway._http_post_json", fake_post)
    gw = LLMGateway(
        provider="nvidia",
        model="stepfun-ai/step-5-preview",
        cache_dir=tmp_path,
        api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    r = gw.complete(task="t", system="sys", user="u")
    assert r.text == "hello from nvidia"
    assert r.provider == "nvidia" and r.from_cache is False
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "stepfun-ai/step-5-preview"
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["payload"]["stream"] is False

    # Second call is served from the nvidia cache (no second POST).
    captured.clear()
    r2 = gw.complete(task="t", system="sys", user="u")
    assert r2.from_cache is True and r2.text == "hello from nvidia"
    assert captured == {}


def test_nvidia_without_key_raises(tmp_path: Path) -> None:
    gw = LLMGateway(provider="nvidia", cache_dir=tmp_path, api_key=None)
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")


def test_mock_replays_nvidia_cache(tmp_path: Path) -> None:
    # A mock gateway should be able to replay a real nvidia response offline.
    nvidia_dir = tmp_path / "nvidia"
    nvidia_dir.mkdir(parents=True)
    gw_live = LLMGateway(
        provider="nvidia", cache_dir=tmp_path, api_key="k",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    key = gw_live._key("t", "s", "u")
    (nvidia_dir / f"{key}.json").write_text('{"text": "cached nvidia"}', encoding="utf-8")
    mock = LLMGateway(provider="mock", cache_dir=tmp_path)
    r = mock.complete(task="t", system="s", user="u")
    assert r.from_cache and r.text == "cached nvidia" and r.provider == "nvidia"


def test_nvidia_vision_builds_multimodal_request(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, headers, payload, timeout=120):
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "transcribed"}}]}

    monkeypatch.setattr("ees.llm_gateway._http_post_json", fake_post)
    gw = LLMGateway(
        provider="nvidia",
        model="meta/llama-3.2-11b-vision-instruct",
        cache_dir=tmp_path,
        api_key="k",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    r = gw.complete(
        task="ocr", system="sys", user="transcribe this", images=[b"\x89PNGfake"], max_tokens=512
    )
    assert r.text == "transcribed"
    content = captured["payload"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "transcribe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["payload"]["max_tokens"] == 512


def test_vision_cache_key_depends_on_image_bytes(tmp_path: Path) -> None:
    gw = LLMGateway(provider="mock", cache_dir=tmp_path)
    assert gw._key("ocr", "s", "u", [b"a"]) != gw._key("ocr", "s", "u", [b"b"])
    assert gw._key("ocr", "s", "u", [b"a"]) == gw._key("ocr", "s", "u", [b"a"])
