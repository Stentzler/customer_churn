"""Replaceable LLM provider boundary for constrained planning."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class LlmProviderError(RuntimeError):
    """Raised when an LLM provider cannot return a usable response."""


class LlmProvider(Protocol):
    """Small interface implemented by any future LLM backend.

    The rest of the project depends on this protocol instead of a concrete SDK.
    That keeps GitHub Models, local fake providers, or a future provider
    replaceable without changing deterministic training code.
    """

    def generate_experiment_plan(self, prompt: str) -> str:
        """Return raw model output for later Pydantic and policy validation."""
        ...


class DisabledLlmProvider:
    """Provider used when LLM planning is intentionally disabled."""

    def generate_experiment_plan(self, prompt: str) -> str:
        """Fail explicitly so the planner can choose deterministic fallback."""

        raise LlmProviderError("LLM planning is disabled")


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProviderSettings:
    """Configuration for OpenAI-compatible chat-completions APIs."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int


class OpenAICompatibleChatProvider:
    """Chat-completions provider using only the standard library."""

    def __init__(self, settings: OpenAICompatibleProviderSettings) -> None:
        self._settings = settings

    def generate_experiment_plan(self, prompt: str) -> str:
        """Request a JSON experiment plan from the configured provider."""

        payload = {
            "max_tokens": self._settings.max_tokens,
            "messages": [
                {
                    "content": (
                        "You are a constrained MLOps planner. Return only valid "
                        "JSON matching the requested schema."
                    ),
                    "role": "system",
                },
                {
                    "content": prompt,
                    "role": "user",
                },
            ],
            "model": self._settings.model,
            "response_format": {"type": "json_object"},
            "temperature": self._settings.temperature,
        }
        request = urllib.request.Request(
            url=f"{self._settings.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "customer-churn-agentic-mlops/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._settings.timeout_seconds,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise LlmProviderError(
                f"LLM provider request failed status={error.code}"
            ) from error
        except (
            TimeoutError,
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as error:
            raise LlmProviderError("LLM provider request failed") from error

        return _extract_chat_message_content(response_payload)


def _extract_chat_message_content(payload: object) -> str:
    if not isinstance(payload, dict):
        raise LlmProviderError("LLM provider response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmProviderError("LLM provider response does not contain choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LlmProviderError("LLM provider response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LlmProviderError(
            "LLM provider response choice does not contain a message"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LlmProviderError("LLM provider response message content is empty")
    return content
