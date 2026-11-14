"""Thin wrapper around an OpenAI-compatible chat API (Groq or OpenAI), so every LLM call goes through one place."""
from __future__ import annotations

import json
import re

from app.config import Settings
from app.errors import ServiceError, UpstreamError


PROVIDERS = {
    # provider: (base_url, default model, settings field holding the key, env var name)
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant", "groq_api_key", "GROQ_API_KEY"),
    "openai": (None, "gpt-4o-mini", "openai_api_key", "OPENAI_API_KEY"),
}


class LLMClient:
    """Both providers speak the OpenAI chat-completions protocol, so one client covers them."""

    def __init__(self, settings: Settings):
        provider = (settings.llm_provider or "groq").lower()
        if provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}")
        base_url, default_model, key_field, self.key_env = PROVIDERS[provider]
        self.settings = settings
        self.provider = provider
        self.base_url = settings.llm_base_url or base_url
        self.model = settings.llm_model or default_model
        self.api_key = getattr(settings, key_field)
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ServiceError(503, "MODEL_NOT_READY", f"{self.key_env} is not set")
            from openai import OpenAI

            # Free tiers have per-minute token limits: retry 429s with the server's back-off.
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.settings.llm_timeout_s,
                                  max_retries=8 if self.provider == "groq" else 3)
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
