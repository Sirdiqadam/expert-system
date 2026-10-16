"""Tests for the LLM gateway (providers, request building, errors).

The gateway has no mock or offline mode. Request-building tests stub the HTTP
boundary; everything else would require live credentials.
"""

from __future__ import annotations

import pytest

from ees.llm_gateway import LLMGateway, LLMGatewayError


def test_unknown_provider_raises() -> None:
    gw = LLMGateway(provider="mock", api_key="k")
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")


def test_nvidia_without_key_raises() -> None:
    gw = LLMGateway(provider="nvidia", api_key=None)
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")


def test_anthropic_without_key_raises() -> None:
    gw = LLMGateway(provider="anthropic", api_key=None)
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")


def test_nvidia_provider_builds_request(monkeypatch) -> None:
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
        api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    r = gw.complete(task="t", system="sys", user="u")
    assert r.text == "hello from nvidia"
    assert r.provider == "nvidia"
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "stepfun-ai/step-5-preview"
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["payload"]["stream"] is False


def test_nvidia_vision_builds_multimodal_request(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, headers, payload, timeout=120):
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "transcribed"}}]}

    monkeypatch.setattr("ees.llm_gateway._http_post_json", fake_post)
    gw = LLMGateway(
        provider="nvidia",
        model="meta/llama-3.2-11b-vision-instruct",
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


def test_reasoning_content_is_used_when_content_is_empty(monkeypatch) -> None:
    def fake_post(url, headers, payload, timeout=120):
        return {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "step-5 working"},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr("ees.llm_gateway._http_post_json", fake_post)
    gw = LLMGateway(provider="nvidia", model="stepfun-ai/step-5-preview", api_key="k")
    r = gw.complete(task="t", system="s", user="u")
    assert r.text == "step-5 working"


def test_empty_response_raises(monkeypatch) -> None:
    def fake_post(url, headers, payload, timeout=120):
        return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}

    monkeypatch.setattr("ees.llm_gateway._http_post_json", fake_post)
    gw = LLMGateway(provider="nvidia", model="stepfun-ai/step-5-preview", api_key="k")
    with pytest.raises(LLMGatewayError):
        gw.complete(task="t", system="s", user="u")
