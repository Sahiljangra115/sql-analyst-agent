"""Pluggable LLM backends: Anthropic, OpenAI-compatible (incl. Ollama), deterministic fake."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLM(Protocol):
    name: str
    price_in: float  # USD per 1M input tokens
    price_out: float

    def complete(self, system: str, user: str) -> LLMResult: ...


def _price(prefix_default_in=0.0, prefix_default_out=0.0):
    return (float(os.getenv("SQLAGENT_PRICE_IN", prefix_default_in)),
            float(os.getenv("SQLAGENT_PRICE_OUT", prefix_default_out)))


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("SQLAGENT_MODEL", "claude-haiku-4-5")
        self.name = f"anthropic:{self.model}"
        # Prices are not hardcoded (they change). Set SQLAGENT_PRICE_IN/OUT per 1M tokens.
        self.price_in, self.price_out = _price()

    def complete(self, system, user):
        r = self.client.messages.create(model=self.model, max_tokens=1024, temperature=0,
                                        system=system, messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResult(text, r.usage.input_tokens, r.usage.output_tokens)


class OpenAICompatLLM:
    """Any /v1/chat/completions server: Ollama, vLLM, llama.cpp, OpenAI."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 300):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.name = f"openai-compat:{model}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout)
        self.price_in, self.price_out = _price()  # local models: 0 unless overridden

    def complete(self, system, user):
        body = {"model": self.model, "temperature": 0, "max_tokens": 1024,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        effort = os.getenv("SQLAGENT_REASONING_EFFORT")  # e.g. "none" turns off thinking on Ollama reasoning models
        if effort:
            body["reasoning_effort"] = effort
        r = self.client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=body)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        return LLMResult(j["choices"][0]["message"]["content"] or "",
                         u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class ApiLLM(OpenAICompatLLM):
    """Hosted model behind an OpenAI-compatible endpoint. The key is read from an env var only
    (SQLAGENT_API_KEY, or the var named in SQLAGENT_API_KEY_ENV); it is never read from a file.
    Cost per question needs SQLAGENT_PRICE_IN / SQLAGENT_PRICE_OUT (USD per 1M tokens), set from the
    provider price page. If unset, cost reports 0 and a warning is printed once."""

    def __init__(self, base_url: str, model: str, key_env: str = "SQLAGENT_API_KEY"):
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(f"api backend needs the key in env var {key_env} (not set)")
        super().__init__(base_url, model, key, timeout=120)
        self.name = f"api:{model}"
        if not (os.getenv("SQLAGENT_PRICE_IN") and os.getenv("SQLAGENT_PRICE_OUT")):
            print("warning: SQLAGENT_PRICE_IN/OUT not set, cost per question will read 0", file=__import__("sys").stderr)


class OllamaLLM:
    """Native Ollama /api/chat. Needed because the OpenAI-compatible route cannot set num_ctx
    (default 4096 silently truncates full-schema prompts) or disable thinking reliably."""

    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 12288, timeout: float = 600):
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx
        self.name = f"ollama:{model}"
        self.client = httpx.Client(timeout=timeout)
        self.price_in, self.price_out = _price()

    def complete(self, system, user):
        r = self.client.post(f"{self.host}/api/chat", json={
            "model": self.model, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 1024},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        return LLMResult(j["message"]["content"] or "", j.get("prompt_eval_count", 0), j.get("eval_count", 0))


class FakeLLM:
    """Deterministic. `script` is a list of replies (cycled) or a fn(system, user) -> str."""
    price_in = price_out = 0.0

    def __init__(self, script: list[str] | Callable[[str, str], str]):
        self.script, self.i, self.name, self.calls = script, 0, "fake", []

    def complete(self, system, user):
        self.calls.append((system, user))
        if callable(self.script):
            text = self.script(system, user)
        else:
            text = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
        return LLMResult(text, len(system + user) // 4, len(text) // 4)


def _ollama_up(url: str) -> bool:
    try:
        return httpx.get(url.replace("/v1", "") + "/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _need(name: str):
    raise RuntimeError(f"set {name}")


def make_llm(backend: str | None = None) -> LLM:
    """backend: anthropic | api | openai | ollama | fake. Default: auto-detect from env names."""
    backend = backend or os.getenv("SQLAGENT_BACKEND")
    base = os.getenv("SQLAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not backend:
        if os.getenv("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        elif base:
            backend = "openai"
        elif _ollama_up("http://localhost:11434/v1"):
            backend = "ollama"
        else:
            raise RuntimeError("no LLM reachable: set ANTHROPIC_API_KEY, SQLAGENT_BASE_URL, or run Ollama")
    if backend == "anthropic":
        return AnthropicLLM()
    if backend == "ollama":
        return OllamaLLM(os.getenv("SQLAGENT_MODEL", "gemma4:e2b"), num_ctx=int(os.getenv("SQLAGENT_NUM_CTX", 12288)))
    if backend == "openai":
        return OpenAICompatLLM(base or "https://api.openai.com/v1", os.getenv("SQLAGENT_MODEL", "gpt-4o-mini"),
                               os.getenv("OPENAI_API_KEY") or os.getenv("SQLAGENT_API_KEY"))
    if backend == "api":
        if not base:
            raise RuntimeError("api backend needs SQLAGENT_BASE_URL (OpenAI-compatible, ends in /v1)")
        return ApiLLM(base, os.getenv("SQLAGENT_MODEL") or _need("SQLAGENT_MODEL"),
                      os.getenv("SQLAGENT_API_KEY_ENV", "SQLAGENT_API_KEY"))
    if backend == "fake":
        return FakeLLM(["SELECT 1"])
    raise ValueError(f"unknown backend {backend}")
