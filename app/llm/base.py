"""Abstract LLM client interface.

Keeping this as a small abstract interface (rather than calling the Gemini
SDK directly from services) means:
  1. Tests can mock a single method instead of mocking the whole SDK.
  2. Swapping providers later only requires a new adapter class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """Returns the raw text response from the model.

        Implementations must translate provider-specific errors into the
        typed exceptions in `app.core.exceptions` (LLMTimeoutError,
        LLMRateLimitError, LLMUnavailableError, LLMConfigurationError) and
        must apply bounded retries only for transient failures.
        """
        raise NotImplementedError
