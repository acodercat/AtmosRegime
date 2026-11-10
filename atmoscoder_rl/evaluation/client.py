"""The model client: one OpenAI-compatible chat endpoint, configured by a `models.toml` section.

Every model call in this repository goes through `ChatModel.generate` — a local vLLM server
(`model_id = "local"`) or a hosted endpoint alike. A section of `models.toml` holds:

| field | meaning |
|---|---|
| `model_id` | the served model name (required) |
| `base_url` | the endpoint's `/v1` URL (required) |
| `api_key` / `api_key_env` | a literal key (only the placeholder `EMPTY` belongs in the file) or the environment variable holding the key |
| `temperature`, `max_tokens`, `timeout` | sampling and transport |
| `extra_body` | passed through verbatim (e.g. `chat_template_kwargs.enable_thinking`) |
| `thinking_echo_field` | the streamed delta field carrying the reasoning trace (`reasoning` on vLLM >= 0.24, `reasoning_content` elsewhere) |
| `retry_status` | HTTP statuses this endpoint also returns transiently, retried like 429/5xx (e.g. `[404]` for a flaky gateway) |

Unknown fields are rejected, so a typo cannot silently fall back to a default.

Responses are streamed, so a long generation cannot trip a gateway idle timeout; the
accumulated text equals a non-streamed completion. Transient failures (connection errors,
408/409/429/5xx) are retried with exponential backoff, honouring `Retry-After`. What cannot be
retried surfaces as `PromptTooLongError` (deterministic) or `ModelError`.
"""

from __future__ import annotations

import logging
import os
import random
import time
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import openai

from ..paths import MODELS_TOML

DEFAULT_MAX_TOKENS = 16384
FIELDS = {"model_id", "base_url", "api_key", "api_key_env", "temperature", "max_tokens",
          "timeout", "extra_body", "thinking_echo_field", "retry_status"}

MAX_RETRIES = 5
BASE_DELAY = 0.5   # seconds
MAX_DELAY = 32.0   # seconds
RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
_TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, ConnectionError, TimeoutError)

logger = logging.getLogger(__name__)
T = TypeVar("T")

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")


class ModelError(Exception):
    """A model call that failed for good. Recorded as an excluded error, not the model's miss."""


class PromptTooLongError(ModelError):
    """The request exceeded the context window. Deterministic, never retried."""


@dataclass(frozen=True)
class Completion:
    """One response. `content` is the answer channel exactly as streamed; `text` is what the
    evaluation grades — `content`, or the reasoning trace when a thinking server left content
    empty. `reasoning` is the trace, if the endpoint streams one in `thinking_echo_field`.
    Token counts are as the provider reported them; `reasoning_tokens` is the thinking share
    of `completion_tokens` (0 if not reported)."""
    text: str
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str | None = None
    reasoning_tokens: int = 0

    @property
    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens}


def load_config(name: str, path: Path = MODELS_TOML) -> dict[str, Any]:
    """One section of `models.toml`, validated, with `api_key_env` resolved to `api_key`."""
    with open(path, "rb") as handle:
        registry = tomllib.load(handle)
    if name not in registry:
        raise ValueError(f"model {name!r} not in {path}; have {sorted(registry)}")
    config = dict(registry[name])
    unknown = set(config) - FIELDS
    if unknown:
        raise ValueError(f"model {name!r}: unknown field(s) {sorted(unknown)}; allowed {sorted(FIELDS)}")
    for required in ("model_id", "base_url"):
        if not config.get(required):
            raise ValueError(f"model {name!r}: missing {required!r}")
    if "api_key_env" in config:
        variable = config.pop("api_key_env")
        key = os.environ.get(variable)
        if not key:
            raise ValueError(f"model {name!r}: environment variable {variable} (its key) is not set")
        config["api_key"] = key
    if not config.get("api_key"):
        raise ValueError(f"model {name!r}: missing 'api_key' (or 'api_key_env')")
    return config


def build_model(name: str) -> ChatModel:
    """The client for one `models.toml` section."""
    config = load_config(name)
    return ChatModel(config["model_id"], api_key=config["api_key"], base_url=config["base_url"],
                     timeout=config.get("timeout", 120.0), temperature=config.get("temperature"),
                     max_tokens=config.get("max_tokens", DEFAULT_MAX_TOKENS),
                     extra_body=config.get("extra_body"),
                     thinking_echo_field=config.get("thinking_echo_field"),
                     retry_status=config.get("retry_status", ()))


def _backoff(attempt: int) -> float:
    """0.5, 1, 2, ... 32 s (capped), plus 0-25 % jitter, for a 0-indexed attempt."""
    base = min(BASE_DELAY * 2 ** attempt, MAX_DELAY)
    return base + random.uniform(0, 0.25 * base)


def _retry_delay(attempt: int, error: Exception) -> float:
    retry_after = (getattr(error, "headers", None) or {}).get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), MAX_DELAY)
        except (TypeError, ValueError):
            pass
    return _backoff(attempt)


def _with_retry(operation: Callable[[], T], retryable_status: set[int]) -> T:
    def retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        return status in retryable_status or isinstance(error, _TRANSIENT)

    for attempt in range(MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == MAX_RETRIES or not retryable(error):
                raise
            delay = _retry_delay(attempt, error)
            logger.warning("model call failed (attempt %d/%d), retrying in %.1fs: %s",
                           attempt + 1, MAX_RETRIES + 1, delay, error)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _is_context_length(error: Exception) -> bool:
    if getattr(error, "code", None) == "context_length_exceeded":
        return True
    message = str(error).lower()
    return "maximum context length" in message or "context window" in message


def _field(obj: Any, name: str) -> Any:
    """A field of an SDK object, including non-standard ones kept in `model_extra`."""
    value = getattr(obj, name, None)
    if value is None and getattr(obj, "model_extra", None):
        value = obj.model_extra.get(name)
    return value


def _reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    count = getattr(details, "reasoning_tokens", None) if details else None
    return (count if count is not None else _field(usage, "reasoning_tokens")) or 0


class ChatModel:
    """A single-turn chat client for one OpenAI-compatible model."""

    def __init__(self, model_id: str, *, api_key: str, base_url: str, timeout: float = 120.0,
                 temperature: float | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
                 extra_body: dict | None = None, thinking_echo_field: str | None = None,
                 retry_status: Iterable[int] = ()) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or None
        self.thinking_echo_field = thinking_echo_field
        self.retryable_status = RETRYABLE_HTTP_STATUS | set(retry_status)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _request(self, prompt: str, system: str | None) -> dict[str, Any]:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        request: dict[str, Any] = {"model": self.model_id, "messages": messages,
                                   "max_tokens": self.max_tokens}
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.extra_body:
            extra = self.extra_body
            if os.environ.get("ATMOS_NO_CHAT_TEMPLATE_KWARGS") == "1":   # Mistral-tokenizer vLLM servers reject it (Ministral 3; explore/x_data, x2_passk do the same)
                extra = {k: v for k, v in extra.items() if k != "chat_template_kwargs"}
            if extra:
                request["extra_body"] = extra
        return request

    def _stream(self, request: dict[str, Any]) -> Completion:
        content: list[str] = []
        reasoning: list[str] = []
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        with self.client.chat.completions.create(
                **request, stream=True, stream_options={"include_usage": True}) as stream:
            for chunk in stream:
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                    reasoning_tokens = _reasoning_tokens(chunk.usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content.append(delta.content)
                if self.thinking_echo_field:
                    piece = _field(delta, self.thinking_echo_field)
                    if piece:
                        reasoning.append(piece)
        answer, reasoning_text = "".join(content), "".join(reasoning)
        # A thinking server can put everything in the reasoning field and leave content empty.
        return Completion(text=answer or reasoning_text, content=answer, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens, reasoning=reasoning_text or None,
                          reasoning_tokens=reasoning_tokens)

    def generate(self, prompt: str, system: str | None = None) -> Completion:
        request = self._request(prompt, system)
        try:
            return _with_retry(lambda: self._stream(request), self.retryable_status)
        except openai.BadRequestError as error:
            raise (PromptTooLongError if _is_context_length(error) else ModelError)(str(error)) from error
        except Exception as error:
            raise ModelError(str(error)) from error
