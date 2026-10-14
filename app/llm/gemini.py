"""Gemini LLM client, using Google's official unified `google-genai` SDK.

Error handling covers every failure mode called out in the project spec:
invalid API key, quota exhaustion, rate limiting, timeouts, transient
server errors, and unavailable models. Only transient errors (timeout,
429, 5xx) are retried, with a small bounded number of attempts and
exponential backoff - a bad API key or an invalid model name fails fast
instead of retrying uselessly.
"""
from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logging import get_logger
from app.llm.base import LLMClient

logger = get_logger(__name__)


class _TransientLLMError(Exception):
    """Internal marker used to trigger tenacity retries."""


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 2,
        timeout_seconds: int = 30,
    ):
        if not api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self._model = model
        self._max_retries = max_retries
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout_seconds * 1000),
        )

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        try:
            return self._generate_with_retry(system_instruction, user_prompt)
        except _TransientLLMError as exc:
            raise self._last_mapped_error from exc

    def _generate_with_retry(self, system_instruction: str, user_prompt: str) -> str:
        self._last_mapped_error: Exception | None = None

        @retry(
            reraise=True,
            stop=stop_after_attempt(max(1, self._max_retries + 1)),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(_TransientLLMError),
        )
        def _call() -> str:
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    ),
                )
            except genai_errors.ClientError as exc:
                mapped = self._map_client_error(exc)
                if isinstance(mapped, (LLMRateLimitError,)):
                    self._last_mapped_error = mapped
                    raise _TransientLLMError(str(exc)) from exc
                raise mapped from exc
            except genai_errors.ServerError as exc:
                self._last_mapped_error = LLMUnavailableError(
                    f"Gemini server error: {exc}"
                )
                raise _TransientLLMError(str(exc)) from exc
            except TimeoutError as exc:
                self._last_mapped_error = LLMTimeoutError(
                    f"Gemini request timed out: {exc}"
                )
                raise _TransientLLMError(str(exc)) from exc
            except Exception as exc:
                message = str(exc).lower()
                if "timeout" in message or "timed out" in message:
                    self._last_mapped_error = LLMTimeoutError(str(exc))
                    raise _TransientLLMError(str(exc)) from exc
                if "unavailable" in message or "503" in message:
                    self._last_mapped_error = LLMUnavailableError(str(exc))
                    raise _TransientLLMError(str(exc)) from exc
                # Unknown/non-transient failure: do not retry.
                logger.exception("Unexpected Gemini failure")
                raise LLMUnavailableError(f"Unexpected Gemini failure: {exc}") from exc

            text = getattr(response, "text", None)
            if not text:
                raise LLMUnavailableError("Gemini returned an empty response.")
            return text

        return _call()

    @staticmethod
    def _map_client_error(exc: genai_errors.ClientError) -> Exception:
        code = getattr(exc, "code", None)
        message = str(exc)
        if code == 401 or "API key" in message or "UNAUTHENTICATED" in message:
            return LLMConfigurationError(f"Invalid Gemini API key: {message}")
        if code == 429 or "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower():
            return LLMRateLimitError(f"Gemini rate limit / quota exceeded: {message}")
        if code == 404:
            return LLMConfigurationError(f"Gemini model not found/unavailable: {message}")
        return LLMUnavailableError(f"Gemini client error: {message}")
