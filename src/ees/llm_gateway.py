"""LLM Gateway (MVP plan component 9): the single door to any LLM.

Responsibilities:
* one wrapper class with pluggable live providers (``nvidia`` and ``anthropic``),
* retries with backoff for the live provider,
* multimodal (vision) completions for reading scanned pages.

There is no mock or offline mode: every completion is a live provider call, and a
missing API key is an error. No other module may talk to a provider SDK directly.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

from .logging_setup import get_logger

log = get_logger("ees.llm")


class LLMGatewayError(RuntimeError):
    """Raised when a completion cannot be produced or validated."""


@dataclass
class LLMResponse:
    """A completion plus provenance used by the UI and the cost report."""

    text: str
    provider: str
    tokens: int


def _http_post_json(url: str, headers: dict[str, str], payload: dict, timeout: int = 120) -> dict:
    """POST JSON via the standard library and return the parsed response."""

    import json as _json
    import urllib.request

    body = _json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return _json.loads(response.read().decode("utf-8"))


def _openai_message_text(data: dict) -> str:
    """Extract text from an OpenAI-compatible response, tolerating reasoning models.

    Some models (e.g. ``stepfun-ai/step-5-preview``) put their working in
    ``message.reasoning_content`` and can leave ``content`` empty when the token
    budget is exhausted. Prefer ``content``; fall back to ``reasoning_content``;
    otherwise raise with the ``finish_reason`` so the cause is obvious.
    """

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content:
        return content
    reasoning = message.get("reasoning_content")
    if reasoning:
        log.warning(
            "provider returned empty content (finish_reason=%s); using reasoning_content",
            choice.get("finish_reason"),
        )
        return reasoning
    raise LLMGatewayError(
        f"provider returned no text (finish_reason={choice.get('finish_reason')}); "
        "the token budget may be too small for a reasoning model - raise max_tokens"
    )


class LLMGateway:
    """A retrying, provider-agnostic completion gateway."""

    def __init__(
        self,
        provider: str = "nvidia",
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_key = api_key
        self.base_url = base_url

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete(
        self,
        *,
        task: str,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 8192,
        timeout: int = 180,
    ) -> LLMResponse:
        """Return a live completion from the configured provider.

        ``task`` is a short label (``extraction`` / ``explanation`` /
        ``llm_only`` / ``ocr`` ...) used for logging. ``images`` (list of PNG
        bytes) turns the call into a vision request.
        """

        if self.provider not in {"anthropic", "nvidia"}:
            raise LLMGatewayError(
                f"unknown provider '{self.provider}' (expected 'nvidia' or 'anthropic')"
            )
        if not self.api_key:
            raise LLMGatewayError(
                f"provider '{self.provider}' requires an API key (set {self._key_env_hint()})"
            )
        log.debug("live completion via %s for task=%s", self.provider, task)
        if self.provider == "anthropic":
            text = self._call_anthropic(system, user, images, max_tokens)
        else:
            text = self._call_nvidia(system, user, images, max_tokens, timeout)
        return LLMResponse(text, provider=self.provider, tokens=self._tokens(text))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _key_env_hint(self) -> str:
        return {
            "anthropic": "ANTHROPIC_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
        }.get(self.provider, "API key")

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _call_nvidia(
        self,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 8192,
        timeout: int = 180,
    ) -> str:
        """Call an NVIDIA NIM endpoint (OpenAI-compatible chat completions)."""

        base = (self.base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        if images:
            content: list[dict] = [{"type": "text", "text": user}]
            for img in images:
                b64 = base64.b64encode(img).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            messages: list[dict] = [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": self.temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = _http_post_json(url, headers, payload, timeout=timeout)
                return _openai_message_text(data)
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                log.warning("nvidia call failed (attempt %s): %s", attempt + 1, exc)
                time.sleep(2**attempt)
        raise LLMGatewayError(f"nvidia call failed after retries: {last_error}")

    def _call_anthropic(
        self,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        try:
            import anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise LLMGatewayError(
                "the 'anthropic' package is not installed; run `pip install anthropic`"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        if images:
            blocks: list[dict] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(img).decode("ascii"),
                    },
                }
                for img in images
            ]
            blocks.append({"type": "text", "text": user})
            content: object = blocks
        else:
            content = user
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                )
                return "".join(
                    block.text for block in message.content if getattr(block, "type", "") == "text"
                )
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                backoff = 2**attempt
                log.warning("anthropic call failed (attempt %s): %s", attempt + 1, exc)
                time.sleep(backoff)
        raise LLMGatewayError(f"anthropic call failed after retries: {last_error}")
