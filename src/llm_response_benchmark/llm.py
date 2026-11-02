"""Pluggable LLM backends.

`get_backend(name)` returns an object with
    complete(prompt: str, system: str | None = None, temperature: float = 0.0) -> str

Backends:
  mock       deterministic heuristic, no network (default)
  ollama     local Ollama server at $OLLAMA_HOST (default http://localhost:11434), model $OLLAMA_MODEL
  anthropic  Anthropic Messages API, key $ANTHROPIC_API_KEY, model $ANTHROPIC_MODEL
  openai     OpenAI Chat Completions API, key $OPENAI_API_KEY, model $OPENAI_MODEL
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Protocol

import httpx

from .textutils import check_constraints, overlap, shorten, tokens


class BackendError(RuntimeError):
    """Raised when a backend is misconfigured or unreachable."""


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------
_TAG = re.compile(r"<(?P<tag>[a-z_]+)>\n?(?P<body>.*?)\n?</(?P=tag)>", re.S)
_CHATTY = ("great question", "i hope this helps", "feel free", "absolutely", "sure, here", "let me know")


class MockBackend:
    """Deterministic stand-in for an LLM.

    For judge prompts (containing <response>...</response> and friends) it scores the response
    with simple heuristics: token overlap against key facts and known false claims, constraint
    checks, and phrase lists for padding. A hash of the prompt seeds a small amount of
    jitter so the mock behaves like a slightly imperfect judge, but the same input always
    produces the same output. Any other prompt gets a short canned completion.
    """

    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _rng_int(self, text: str, salt: str, mod: int) -> int:
        h = hashlib.sha256(f"{self.seed}|{salt}|{text}".encode()).digest()
        return int.from_bytes(h[:8], "big") % mod

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        sections = {m["tag"]: m["body"].strip() for m in _TAG.finditer(prompt)}
        if "response" in sections and "key_facts" in sections:
            return json.dumps(self._judge(sections, prompt))
        first = shorten(prompt.strip().splitlines()[0] if prompt.strip() else "", 80)
        return f"[mock completion] {first}"

    def _judge(self, sec: dict[str, str], raw: str) -> dict:
        from .schemas import Constraints

        response = sec["response"]
        rtoks = set(tokens(response))
        facts = [ln.lstrip("- ").strip() for ln in sec["key_facts"].splitlines() if ln.strip()]
        falses = [ln.lstrip("- ").strip() for ln in sec.get("false_claims", "").splitlines() if ln.strip()]
        try:
            constraints = Constraints.model_validate_json(sec.get("constraints") or "{}")
        except Exception:
            constraints = Constraints()

        covered = [f for f in facts if overlap(f, rtoks) >= 0.6]
        missing = [f for f in facts if f not in covered]
        found_false = [f for f in falses if overlap(f, rtoks) >= 0.75]
        violations = check_constraints(response, constraints)
        low = response.lower()
        chatty = sum(p in low for p in _CHATTY)

        context = set(tokens(sec.get("prompt", "")))
        for f in facts + falses:
            context |= set(tokens(f))
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", response) if len(tokens(s)) >= 4]
        offtopic = [s for s in sentences if overlap(s, context) < 0.25 and not any(p in s.lower() for p in _CHATTY)]

        frac = len(covered) / max(len(facts), 1)
        scores = {
            "factual_accuracy": 2 if found_false else (5 if frac >= 0.6 else 4),
            "relevance": max(1, 5 - 2 * min(len(offtopic), 1) - (1 if len(offtopic) > 1 else 0)),
            "completeness": 5 if frac == 1 else 4 if frac >= 0.6 else 2 if frac > 0 else 1,
            "instruction_following": max(1, 5 - 2 * len(violations)),
            "hallucination": 2 if found_false else 5,
            "style": max(2, 5 - min(chatty, 2) - (1 if offtopic else 0)),
        }
        # deterministic jitter: ~15% of judgements move one dimension by one point
        if self._rng_int(raw, "jitter", 100) < 15:
            dims = list(scores)
            d = dims[self._rng_int(raw, "dim", len(dims))]
            step = 1 if self._rng_int(raw, "dir", 2) else -1
            scores[d] = min(5, max(1, scores[d] + step))

        rationales = {
            "factual_accuracy": f'Contradicted claim: "{shorten(found_false[0], 70)}"' if found_false else "No claims conflict with the reference.",
            "relevance": f'Off-topic sentence: "{shorten(offtopic[0], 70)}"' if offtopic else "Stays on the question.",
            "completeness": f"Covers {len(covered)}/{len(facts)} key facts" + (f"; missing: {shorten(missing[0], 60)}" if missing else "."),
            "instruction_following": "; ".join(violations) if violations else "All checkable constraints satisfied.",
            "hallucination": "Asserts a known false claim." if found_false else "No known false claims detected.",
            "style": "Contains conversational padding." if chatty else "Direct and concise.",
        }
        return {"scores": scores, "rationales": rationales}


# ---------------------------------------------------------------------------
# HTTP backends
# ---------------------------------------------------------------------------
class _HTTPBackend:
    name = "http"
    timeout = 120.0

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        try:
            r = httpx.post(url, json=payload, headers=headers or {}, timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise BackendError(f"{self.name}: cannot connect to {url} ({exc}). Is the service reachable?") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name}: request to {url} failed: {exc}") from exc
        if r.status_code >= 400:
            raise BackendError(f"{self.name}: HTTP {r.status_code} from {url}: {r.text[:300]}")
        return r.json()


class OllamaBackend(_HTTPBackend):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            data = self._post(
                f"{self.host}/api/chat",
                {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
            )
        except BackendError as exc:
            raise BackendError(f"{exc}\nStart Ollama yourself (`ollama serve`) and pull the model (`ollama pull {self.model}`).") from exc
        return data.get("message", {}).get("content", "")


class AnthropicBackend(_HTTPBackend):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1024):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise BackendError("anthropic: ANTHROPIC_API_KEY is not set.")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = self._post(
            self.url,
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAIBackend(_HTTPBackend):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise BackendError("openai: OPENAI_API_KEY is not set.")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        data = self._post(
            self.url,
            {"model": self.model, "messages": messages, "temperature": temperature},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return data["choices"][0]["message"]["content"] or ""


BACKENDS = {"mock": MockBackend, "ollama": OllamaBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend}


def get_backend(name: str = "mock", **kwargs) -> LLMBackend:
    try:
        cls = BACKENDS[name.lower()]
    except KeyError:
        raise BackendError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
    return cls(**kwargs)
