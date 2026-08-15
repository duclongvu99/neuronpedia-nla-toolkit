"""Neuronpedia NLA (Natural Language Autoencoder) API client.

Zero dependencies: standard library only, Python 3.8+. Copy this one file anywhere
and it works.

    from nla import NLAClient

    c = NLAClient()
    print(c.sources())
    out = c.chat("What is the capital of Canada?")
    for e in c.explain_generated(out):
        print(e.position, e.token, e.cosine_similarity, e.description[:80])

The public API surface is three endpoints (GET /sources, POST /completion,
POST /explain). This client adds the things you need in practice and the raw
endpoints do not give you:

  * auto-batching of `positions` (the server caps 16 NEW positions per request)
  * normalisation of /completion, whose response shape differs between models
    that have an OpenRouter mapping and models that do not
  * the undocumented per-position reconstruction metrics (mse,
    cosine_similarity, l2_norm, generated) exposed as first-class fields
  * 429 handling with backoff, and the remaining-quota header surfaced

See API_NOTES.md for where the live API and the published spec disagree.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "NLAClient",
    "NLAError",
    "NLABadRequest",
    "NLARateLimited",
    "Source",
    "Token",
    "Completion",
    "Explanation",
    "DEFAULT_MODEL",
    "DEFAULT_SOURCE",
]

BASE_URL = "https://www.neuronpedia.org/api/nla"

# The server rejects a request carrying more than this many *new* (uncached)
# positions. Cached positions are free and do not count, but we cannot know
# what is cached before asking, so we batch conservatively.
MAX_NEW_POSITIONS = 16

DEFAULT_MODEL = "gemma-3-27b-it"
DEFAULT_SOURCE = "kitft-l41"

# Known (modelId -> nlaSourceId) defaults, so callers can pass just a model.
# Authoritative list is always GET /sources; this is only a convenience.
KNOWN_SOURCES = {
    "gemma-3-27b-it": "kitft-l41",
    "llama3.3-70b-it": "kitft-l53",
    "qwen2.5-1.5b-it": "andyxu-l18",
}


class NLAError(Exception):
    """Base class for all NLA API failures."""


class NLABadRequest(NLAError):
    """The server rejected the request (HTTP 4xx other than 429)."""


class NLARateLimited(NLAError):
    """HTTP 429. Limits are per-IP, not per-key."""


@dataclass
class Source:
    """One (modelId, nlaSourceId) pair from GET /sources."""

    model_id: str
    id: str
    display_name: str = ""
    author: str = ""
    layer_num: Optional[int] = None
    open_router_available: bool = False
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Source":
        return cls(
            model_id=d.get("modelId", ""),
            id=d.get("id", ""),
            display_name=d.get("displayName", "") or "",
            author=d.get("author", "") or "",
            layer_num=d.get("layerNum"),
            open_router_available=bool((d.get("model") or {}).get("openRouterAvailable")),
            raw=d,
        )


@dataclass
class Token:
    """A token in the NLA tokenizer's canonical view of the chat."""

    position: int
    token: str
    token_id: Optional[int] = None
    # Chat-structure metadata. Present on some responses, absent on others.
    role: Optional[str] = None
    section: Optional[str] = None
    message_index: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Token":
        return cls(
            position=d["position"],
            token=d.get("token", ""),
            token_id=d.get("token_id"),
            role=d.get("role"),
            section=d.get("section"),
            message_index=d.get("message_index"),
            raw=d,
        )


@dataclass
class Completion:
    """Normalised result of POST /completion.

    The endpoint returns two different shapes. Models with an OpenRouter
    mapping return {completion, full_text, tokens}; models without one return
    {text, tokens, prompt_length} and no separate `completion` field. Both are
    flattened into this object, so `full_text` and `tokens` are always usable
    as inputs to explain().
    """

    full_text: str
    tokens: List[Token]
    completion: Optional[str] = None
    model_id: str = ""
    nla_source_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def prompt_length(self) -> int:
        return len(self.tokens)

    def positions_of(self, substring: str) -> List[int]:
        """Positions whose token text contains `substring`. Handy for targeting."""
        return [t.position for t in self.tokens if substring in t.token]

    @classmethod
    def from_json(cls, d: Dict[str, Any], model_id: str, nla_source_id: str) -> "Completion":
        full_text = d.get("full_text") or d.get("text") or ""
        tokens = [Token.from_json(t) for t in (d.get("tokens") or [])]
        completion = d.get("completion")
        if completion is None:
            assistant_tokens = [
                t.token
                for t in tokens
                if t.role == "assistant" and t.section == "content"
            ]
            if assistant_tokens:
                completion = "".join(assistant_tokens)
            else:
                prompt_length = d.get("prompt_length")
                if isinstance(prompt_length, int) and 0 <= prompt_length < len(tokens):
                    completion = "".join(t.token for t in tokens[prompt_length:])
        return cls(
            full_text=full_text,
            tokens=tokens,
            completion=completion,
            model_id=model_id,
            nla_source_id=nla_source_id,
            raw=d,
        )


@dataclass
class Explanation:
    """One explained token position.

    `description` is the natural-language decoding of the residual-stream
    activation at this position. The three metric fields describe how well the
    autoencoder actually reconstructed that activation, so they tell you how
    much to trust the description:

      cosine_similarity : reconstruction fidelity, 1.0 is perfect
      mse               : reconstruction error, lower is better
      l2_norm           : magnitude of the original activation

    A fluent description sitting on a low cosine_similarity is exactly the case
    to distrust. None of these three are in the published spec.
    """

    position: int
    token: str
    description: str
    cosine_similarity: Optional[float] = None
    mse: Optional[float] = None
    l2_norm: Optional[float] = None
    generated: Optional[bool] = None
    role: Optional[str] = None
    section: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Explanation":
        return cls(
            position=d["position"],
            token=d.get("token", ""),
            description=d.get("description", "") or "",
            cosine_similarity=d.get("cosine_similarity"),
            mse=d.get("mse"),
            l2_norm=d.get("l2_norm"),
            generated=d.get("generated"),
            role=d.get("role"),
            section=d.get("section"),
            raw=d,
        )


def load_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the API key: argument, then NEURONPEDIA_API_KEY, then a creds file.

    Searched creds paths, first hit wins:
        ./.nla_creds, ~/.nla_creds, $NLA_CREDS_FILE

    Returns None if nothing is found. That is not fatal: as of 2026-08-15 the
    NLA endpoints serve unauthenticated requests too (see API_NOTES.md).
    """
    if explicit:
        return explicit.strip()
    env = os.environ.get("NEURONPEDIA_API_KEY")
    if env:
        return env.strip()
    candidates = [
        os.environ.get("NLA_CREDS_FILE"),
        os.path.join(os.getcwd(), ".nla_creds"),
        os.path.expanduser("~/.nla_creds"),
        # alongside this file, for when it is copied into another project
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nla_creds"),
    ]
    for path in candidates:
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                key = fh.read().strip()
            if key:
                return key
        except OSError:
            continue
    return None


class NLAClient:
    """Thin, retrying client for the three Neuronpedia NLA endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: float = 300.0,
        max_retries: int = 3,
        user_agent: str = "nla-toolkit/1.0",
    ) -> None:
        self.api_key = load_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        # Populated from the x-limit-remaining response header after each call.
        self.limit_remaining: Optional[int] = None

    # ---------------------------------------------------------------- transport

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        url = "{}/{}".format(self.base_url, path.lstrip("/"))
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["x-api-key"] = self.api_key

        delay = 2.0
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    remaining = resp.headers.get("x-limit-remaining")
                    if remaining is not None:
                        try:
                            self.limit_remaining = int(remaining)
                        except ValueError:
                            pass
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                try:
                    message = json.loads(body).get("error") or body
                except json.JSONDecodeError:
                    message = body
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    last_exc = NLARateLimited(
                        "429 rate limited on {} (limits are per-IP): {}".format(path, message[:300])
                    )
                    if attempt == self.max_retries - 1:
                        break
                    time.sleep(wait)
                    delay *= 2
                    continue
                if 400 <= exc.code < 500:
                    raise NLABadRequest("HTTP {} on {}: {}".format(exc.code, path, message[:500]))
                last_exc = NLAError("HTTP {} on {}: {}".format(exc.code, path, message[:300]))
            except urllib.error.URLError as exc:
                last_exc = NLAError("network error on {}: {}".format(path, exc))
            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= 2
        raise last_exc if last_exc else NLAError("request to {} failed".format(path))

    # ---------------------------------------------------------------- endpoints

    def sources(self) -> List[Source]:
        """GET /sources. Every usable (modelId, nlaSourceId) pair."""
        data = self._request("GET", "/sources")
        return [Source.from_json(s) for s in (data.get("sources") or [])]

    def resolve_source(self, model_id: str, nla_source_id: Optional[str] = None) -> str:
        if nla_source_id:
            return nla_source_id
        if model_id in KNOWN_SOURCES:
            return KNOWN_SOURCES[model_id]
        for s in self.sources():
            if s.model_id == model_id:
                return s.id
        raise NLABadRequest("no NLA source known for model {!r}".format(model_id))

    def completion(
        self,
        messages: Sequence[Dict[str, str]],
        model_id: str = DEFAULT_MODEL,
        nla_source_id: Optional[str] = None,
        completion_tokens: int = 64,
        temperature: float = 0.7,
    ) -> Completion:
        """POST /completion. Generate a reply and get its canonical tokenisation.

        `completion_tokens` is capped server-side at 512. The returned
        `full_text` and token positions line up exactly with explain(), which is
        the whole point of routing generation through this endpoint rather than
        calling a model yourself.
        """
        source = self.resolve_source(model_id, nla_source_id)
        payload = {
            "modelId": model_id,
            "nlaSourceId": source,
            "messages": list(messages),
            "completion_tokens": int(completion_tokens),
            "temperature": temperature,
        }
        return Completion.from_json(self._request("POST", "/completion", payload), model_id, source)

    def chat(
        self,
        prompt: str,
        model_id: str = DEFAULT_MODEL,
        nla_source_id: Optional[str] = None,
        completion_tokens: int = 64,
        temperature: float = 0.7,
        system: Optional[str] = None,
    ) -> Completion:
        """completion() for the common single-user-turn case."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.completion(
            messages,
            model_id=model_id,
            nla_source_id=nla_source_id,
            completion_tokens=completion_tokens,
            temperature=temperature,
        )

    def explain(
        self,
        positions: Iterable[int],
        text: Optional[str] = None,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        model_id: str = DEFAULT_MODEL,
        nla_source_id: Optional[str] = None,
        temperature: float = 0.7,
        batch_size: int = MAX_NEW_POSITIONS,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Explanation]:
        """POST /explain, automatically batched.

        Supply exactly one of `text` (already chat-templated, max 16384 chars)
        or `messages`. Any number of positions may be passed: they are split
        into requests of at most 16, because that is the server's cap on new
        positions. Results come back sorted by position.

        `progress(done, total)` is called after each batch if supplied.
        """
        if (text is None) == (messages is None):
            raise ValueError("pass exactly one of text= or messages=")
        source = self.resolve_source(model_id, nla_source_id)
        wanted = sorted({int(p) for p in positions})
        if not wanted:
            return []
        if batch_size < 1 or batch_size > MAX_NEW_POSITIONS:
            batch_size = MAX_NEW_POSITIONS

        out: List[Explanation] = []
        for start in range(0, len(wanted), batch_size):
            chunk = wanted[start : start + batch_size]
            payload: Dict[str, Any] = {
                "modelId": model_id,
                "nlaSourceId": source,
                "positions": chunk,
                "temperature": temperature,
            }
            if text is not None:
                payload["text"] = text
            else:
                payload["messages"] = list(messages or [])
            data = self._request("POST", "/explain", payload)
            out.extend(Explanation.from_json(r) for r in (data.get("results") or []))
            if progress:
                progress(min(start + batch_size, len(wanted)), len(wanted))
        out.sort(key=lambda e: e.position)
        return out

    def explain_completion(
        self,
        completion: Completion,
        positions: Optional[Iterable[int]] = None,
        temperature: float = 0.7,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Explanation]:
        """Explain positions of an existing Completion, reusing its exact text."""
        if positions is None:
            positions = [t.position for t in completion.tokens]
        return self.explain(
            positions,
            text=completion.full_text,
            model_id=completion.model_id,
            nla_source_id=completion.nla_source_id,
            temperature=temperature,
            progress=progress,
        )

    def explain_generated(
        self,
        completion: Completion,
        limit: Optional[int] = None,
        temperature: float = 0.7,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Explanation]:
        """Explain only the tokens the model itself generated.

        Selects tokens the /completion response marks as role=assistant AND
        section=content, which excludes the chat scaffold (`<start_of_turn>`,
        `model`, `<end_of_turn>`) that also carries role=assistant.

        Note: do NOT use the `generated` field on /explain results for this.
        When you pass `text=`, the server cannot tell prompt from completion and
        returns generated=False for every position, including assistant tokens.
        Role metadata from /completion is the reliable signal.

        Falls back to the tail of the sequence if role metadata is missing, so
        this is best-effort. Use explain_completion() with explicit positions
        when you need certainty about which tokens you are looking at.
        """
        gen = [
            t.position
            for t in completion.tokens
            if t.role == "assistant" and t.section == "content"
        ]
        if not gen:
            marker = max(
                (t.position for t in completion.tokens if "start_of_turn" in t.token or "assistant" in t.token),
                default=None,
            )
            gen = [t.position for t in completion.tokens if marker is None or t.position > marker]
        if limit is not None:
            gen = gen[:limit]
        return self.explain_completion(completion, gen, temperature=temperature, progress=progress)


# --------------------------------------------------------------------- helpers


def format_explanations(
    explanations: Sequence[Explanation],
    width: int = 100,
    full: bool = False,
) -> str:
    """Readable text block: token, fidelity, and the decoded description."""
    lines: List[str] = []
    for e in explanations:
        cos = "  cos={:.4f}".format(e.cosine_similarity) if e.cosine_similarity is not None else ""
        mse = "  mse={:.4f}".format(e.mse) if e.mse is not None else ""
        flag = ""
        if e.cosine_similarity is not None and e.cosine_similarity < 0.9:
            flag = "  [LOW FIDELITY]"
        lines.append("pos {:<4} {!r:<18}{}{}{}".format(e.position, e.token, cos, mse, flag))
        body = e.description.strip()
        if not full:
            body = body.split("\n\n")[0]
            if len(body) > width:
                body = body[: width - 1] + "…"
            lines.append("    " + body)
        else:
            for para in body.split("\n"):
                lines.append("    " + para)
        lines.append("")
    return "\n".join(lines)
