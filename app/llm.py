"""Thin wrapper around the OpenAI chat API, so every LLM call goes through one place."""
from __future__ import annotations

import json
import re

from app.config import Settings
from app.errors import ServiceError, UpstreamError


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _get_client(self):
        if self._client is None:
            if not self.settings.openai_api_key:
                raise ServiceError(503, "MODEL_NOT_READY", "OPENAI_API_KEY is not set")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.openai_api_key,
                                  timeout=self.settings.llm_timeout_s, max_retries=2)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 400, json_mode: bool = False) -> str:
        import openai

        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._get_client().chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise UpstreamError(429, "RATE_LIMITED", f"LLM rate limit: {e}") from e
        except openai.APITimeoutError as e:
            raise UpstreamError(504, "UPSTREAM_TIMEOUT", "LLM API timed out") from e
        except openai.AuthenticationError as e:
            raise UpstreamError(502, "UPSTREAM_AUTH_ERROR", "LLM API rejected the API key") from e
        except openai.APIError as e:
            raise UpstreamError(502, "UPSTREAM_ERROR", f"LLM API error: {e}") from e
        return (resp.choices[0].message.content or "").strip()

    def complete_json(self, system: str, user: str, max_tokens: int = 200) -> dict:
        text = self.complete(system, user, max_tokens=max_tokens, json_mode=True)
        return parse_json(text)


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences."""
    text = text.strip()
    fenced = re.search(r"\{.*\}", text, re.S)
    if fenced:
        text = fenced.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
