"""LLM Gateway (MVP plan component 9): the single door to any LLM.

Responsibilities:
* one wrapper class with pluggable providers (``nvidia``, ``anthropic`` and
  ``mock``),
* a response cache keyed by ``hash(model + system + user [+ image bytes])`` so
  the demo can run fully offline by replaying cached responses,
* retries with backoff for the live provider,
* multimodal (vision) completions for reading scanned pages.

No other module may talk to a provider SDK directly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .logging_setup import get_logger

log = get_logger("ees.llm")


class LLMGatewayError(RuntimeError):
    """Raised when a completion cannot be produced or validated."""


@dataclass
class LLMResponse:
    """A completion plus provenance used by the UI and the cost report."""

    text: str
    provider: str
    from_cache: bool
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
    """A caching, retrying, provider-agnostic completion gateway."""

    def __init__(
        self,
        provider: str = "mock",
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.0,
        max_retries: int = 3,
        cache_dir: str | Path = "data/cache",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
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
        mock: Callable[[], str] | None = None,
        images: list[bytes] | None = None,
        max_tokens: int = 8192,
        timeout: int = 180,
    ) -> LLMResponse:
        """Return a completion, preferring cache, then the live provider.

        ``task`` is a short label (``extraction`` / ``explanation`` /
        ``llm_only``) mixed into the cache key so different prompts never collide.
        ``mock`` is a deterministic factory used when no cache entry exists and
        no API key is available (offline demo). ``images`` (list of PNG bytes)
        turns the call into a vision request; the cache key folds in the image
        bytes so a transcribed page is replayed offline.
        """

        key = self._key(task, system, user, images)

        for sub in self._read_dirs():
            cached = self.cache_dir / sub / f"{key}.json"
            if cached.exists():
                text = json.loads(cached.read_text(encoding="utf-8")).get("text")
                if not isinstance(text, str) or not text:
                    # Skip a poisoned/empty entry rather than returning `None`.
                    log.warning("ignoring empty cache entry (%s) for task=%s", sub, task)
                    continue
                log.debug("cache hit (%s) for task=%s", sub, task)
                return LLMResponse(text, provider=sub, from_cache=True, tokens=self._tokens(text))

        if self.provider in {"anthropic", "nvidia"}:
            if not self.api_key:
                raise LLMGatewayError(
                    f"provider '{self.provider}' requires an API key (set "
                    f"{self._key_env_hint()}) or use provider 'mock'"
                )
            if self.provider == "anthropic":
                text = self._call_anthropic(system, user, images, max_tokens)
            else:
                text = self._call_nvidia(system, user, images, max_tokens, timeout)
            self._store(self.provider, key, text)
            return LLMResponse(
                text, provider=self.provider, from_cache=False, tokens=self._tokens(text)
            )

        # mock provider with an empty cache: use the deterministic factory.
        if mock is None:
            raise LLMGatewayError(
                f"no cached response for task={task} and no mock factory supplied"
            )
        text = mock()
        self._store("mock", key, text)
        return LLMResponse(text, provider="mock", from_cache=False, tokens=self._tokens(text))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _read_dirs(self) -> list[str]:
        # Prefer provider-exact cache; the mock can replay any real provider's
        # cached responses for a frozen offline demo.
        if self.provider == "mock":
            return ["mock", "nvidia", "anthropic"]
        return [self.provider]

    def _key_env_hint(self) -> str:
        return {
            "anthropic": "ANTHROPIC_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
        }.get(self.provider, "API key")

    def _key(
        self, task: str, system: str, user: str, images: list[bytes] | None = None
    ) -> str:
        payload = "\n\x1e\n".join([task, self.model, system.strip(), user.strip()])
        if images:
            digest = hashlib.sha256()
            for img in images:
                digest.update(hashlib.sha256(img).digest())
            payload += "\n\x1e\n" + digest.hexdigest()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _store(self, sub: str, key: str, text: str) -> None:
        if not isinstance(text, str) or not text:
            log.warning("not caching empty response for provider=%s", sub)
            return
        folder = self.cache_dir / sub
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"provider": sub, "model": self.model, "text": text}
        (folder / f"{key}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

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
                "the 'anthropic' package is not installed; run "
                "`pip install anthropic` or use provider 'mock'"
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
