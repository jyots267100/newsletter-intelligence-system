"""Gemini client error-mapping tests.

No real network calls are made: `_client.models.generate_content` is
monkeypatched on an already-constructed GeminiClient. This exercises the
exact error-translation logic without depending on live Gemini credentials
or quota - the project rule is that normal tests never touch the real API.
"""
from unittest.mock import Mock

import pytest
from google.genai import errors as genai_errors

from app.core.exceptions import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.gemini import GeminiClient


def _make_api_error(cls, code: int, message: str):
    # Bypass the real __init__ (whose exact signature varies across SDK
    # versions) and set only what GeminiClient._map_client_error reads.
    exc = cls.__new__(cls)
    exc.code = code
    exc.args = (message,)
    return exc


@pytest.fixture
def client():
    return GeminiClient(api_key="fake-key", model="gemini-2.5-flash", max_retries=1, timeout_seconds=5)


def test_missing_api_key_raises_configuration_error():
    with pytest.raises(LLMConfigurationError):
        GeminiClient(api_key="", model="gemini-2.5-flash")


def test_successful_generation_returns_text(client, monkeypatch):
    fake_response = Mock(text="Hello from Gemini")
    monkeypatch.setattr(
        client._client.models, "generate_content", Mock(return_value=fake_response)
    )
    assert client.generate("system", "user") == "Hello from Gemini"


def test_invalid_api_key_maps_to_configuration_error(client, monkeypatch):
    error = _make_api_error(genai_errors.ClientError, 401, "UNAUTHENTICATED: API key not valid")
    monkeypatch.setattr(
        client._client.models, "generate_content", Mock(side_effect=error)
    )
    with pytest.raises(LLMConfigurationError):
        client.generate("system", "user")


def test_rate_limit_is_retried_then_raises_rate_limit_error(client, monkeypatch):
    error = _make_api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED: quota exceeded")
    mock_call = Mock(side_effect=error)
    monkeypatch.setattr(client._client.models, "generate_content", mock_call)

    with pytest.raises(LLMRateLimitError):
        client.generate("system", "user")

    # max_retries=1 -> initial attempt + 1 retry = 2 calls
    assert mock_call.call_count == 2


def test_server_error_is_retried_then_raises_unavailable_error(client, monkeypatch):
    error = _make_api_error(genai_errors.ServerError, 503, "UNAVAILABLE: overloaded")
    mock_call = Mock(side_effect=error)
    monkeypatch.setattr(client._client.models, "generate_content", mock_call)

    with pytest.raises(LLMUnavailableError):
        client.generate("system", "user")
    assert mock_call.call_count == 2


def test_timeout_message_maps_to_timeout_error(client, monkeypatch):
    mock_call = Mock(side_effect=TimeoutError("request timed out"))
    monkeypatch.setattr(client._client.models, "generate_content", mock_call)

    with pytest.raises(LLMTimeoutError):
        client.generate("system", "user")


def test_empty_response_raises_unavailable_error(client, monkeypatch):
    fake_response = Mock(text="")
    monkeypatch.setattr(
        client._client.models, "generate_content", Mock(return_value=fake_response)
    )
    with pytest.raises(LLMUnavailableError):
        client.generate("system", "user")


def test_transient_success_after_one_retry(client, monkeypatch):
    error = _make_api_error(genai_errors.ServerError, 503, "UNAVAILABLE")
    success = Mock(text="recovered")
    mock_call = Mock(side_effect=[error, success])
    monkeypatch.setattr(client._client.models, "generate_content", mock_call)

    result = client.generate("system", "user")
    assert result == "recovered"
    assert mock_call.call_count == 2
