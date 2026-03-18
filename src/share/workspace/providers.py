"""Small LLM provider abstraction for the workspace agent."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderConfig:
    """Runtime configuration for one model backend."""

    provider_type: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: int = 60
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderResponse:
    """Structured provider output consumed by the runtime."""

    message: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str | None = None


class LLMProvider:
    """Interface for bounded chat+tool planning calls."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> ProviderResponse:
        raise NotImplementedError

    def _parse_response_text(self, text: str) -> ProviderResponse:
        cleaned = text.strip()
        if not cleaned:
            return ProviderResponse(message="", tool_calls=[], raw_text=text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return ProviderResponse(message=cleaned, tool_calls=[], raw_text=text)
        if isinstance(payload, dict):
            message = str(payload.get("message", "")).strip()
            tool_calls = payload.get("tool_calls", []) or []
            if not isinstance(tool_calls, list):
                tool_calls = []
            return ProviderResponse(message=message, tool_calls=tool_calls, raw_text=text)
        return ProviderResponse(message=cleaned, tool_calls=[], raw_text=text)


class OpenAICompatibleProvider(LLMProvider):
    """Call an OpenAI-compatible `/v1/chat/completions` endpoint using stdlib HTTP."""

    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> ProviderResponse:
        url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        system_instruction = (
            "You are a bounded robotics workflow agent. "
            "Always respond with strict JSON: "
            '{"message": "<assistant reply>", "tool_calls": [{"name": "<tool>", "arguments": {...}}]}. '
            "Use an empty tool_calls list when no tool is needed."
        )
        api_key = os.environ.get(self.config.api_key_env or "", "") if self.config.api_key_env else ""
        request_payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_instruction},
                *messages,
                {"role": "system", "content": json.dumps({"available_tools": tools}, indent=2)},
            ],
        }
        body = json.dumps(request_payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Provider request failed: {exc}") from exc
        text = payload["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return self._parse_response_text(str(text))


class OllamaProvider(LLMProvider):
    """Call Ollama's local chat endpoint using stdlib HTTP."""

    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> ProviderResponse:
        url = (self.config.base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a bounded robotics workflow agent. "
                    "Always respond with strict JSON: "
                    '{"message": "<assistant reply>", "tool_calls": [{"name": "<tool>", "arguments": {...}}]}. '
                    "Use an empty tool_calls list when no tool is needed."
                ),
            },
            *messages,
            {"role": "system", "content": json.dumps({"available_tools": tools}, indent=2)},
        ]
        request_payload = {
            "model": self.config.model,
            "stream": False,
            "options": {"temperature": self.config.temperature},
            "messages": prompt_messages,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self.config.extra_headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Provider request failed: {exc}") from exc
        text = payload["message"]["content"]
        return self._parse_response_text(str(text))

