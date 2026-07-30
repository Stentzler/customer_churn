import json
import urllib.error

import pytest
from src.agent import llm
from src.agent.llm import (
    LlmProviderError,
    OpenAICompatibleChatProvider,
    OpenAICompatibleProviderSettings,
)


def test_openai_compatible_provider_extracts_chat_completion_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request = {}

    def fake_urlopen(request, timeout: float):
        captured_request["url"] = request.full_url
        captured_request["timeout"] = timeout
        captured_request["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"schema_version": "1.0"}',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleChatProvider(_settings())

    content = provider.generate_experiment_plan("planner prompt")

    assert content == '{"schema_version": "1.0"}'
    assert captured_request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured_request["timeout"] == 30.0
    assert captured_request["body"]["model"] == "openai/gpt-oss-20b"
    assert captured_request["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_provider_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(_request, timeout: float):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleChatProvider(_settings())

    with pytest.raises(LlmProviderError, match="LLM provider request failed"):
        provider.generate_experiment_plan("planner prompt")


def test_openai_compatible_provider_reports_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(_request, timeout: float):
        raise urllib.error.HTTPError(
            url="https://api.example.test",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleChatProvider(_settings())

    with pytest.raises(LlmProviderError, match="status=429"):
        provider.generate_experiment_plan("planner prompt")


def _settings() -> OpenAICompatibleProviderSettings:
    return OpenAICompatibleProviderSettings(
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        max_tokens=1200,
        model="openai/gpt-oss-20b",
        temperature=0.0,
        timeout_seconds=30.0,
    )


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")
