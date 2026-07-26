"""Token counting with an exact backend when available and a calibrated
BPE approximation when it is not.

Design goal: contextlint must run with zero third-party dependencies, because the
people who most need a cost audit are the ones who will not `pip install` a
dependency tree to get one. If `tiktoken` happens to be installed we use it and
report exact counts. Otherwise we fall back to an approximation that mimics the
GPT-2/cl100k pre-tokenizer and then estimates sub-token counts per chunk.

The approximation is deliberately conservative and its error is reported in the
output, so no number in contextlint is ever presented as more precise than it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# --------------------------------------------------------------------------
# Exact backend (optional)
# --------------------------------------------------------------------------

_TIKTOKEN = None
_TIKTOKEN_TRIED = False


def _tiktoken_encoder(encoding: str = "cl100k_base"):
    """Return a tiktoken encoder, or None if tiktoken is unavailable."""
    global _TIKTOKEN, _TIKTOKEN_TRIED
    if _TIKTOKEN_TRIED:
        return _TIKTOKEN
    _TIKTOKEN_TRIED = True
    try:  # pragma: no cover - depends on the host environment
        import tiktoken

        _TIKTOKEN = tiktoken.get_encoding(encoding)
    except Exception:
        _TIKTOKEN = None
    return _TIKTOKEN


# --------------------------------------------------------------------------
# Approximate backend
# --------------------------------------------------------------------------

# Mirrors the GPT-2 / cl100k pre-tokenizer closely enough for accounting
# purposes: contractions, letter runs, digit runs, punctuation, whitespace.
_PRETOKEN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"      # contractions
    r"| ?[^\W\d_]+"               # optional leading space + letters
    r"| ?\d+"                     # optional leading space + digits
    r"| ?[^\s\w]+"                # optional leading space + punctuation
    r"|\s+(?!\S)"                 # trailing whitespace runs
    r"|\s+",                      # any other whitespace
    re.UNICODE,
)

# Average characters per BPE token inside a single alphabetic word. Common
# English words are usually one token; longer or rarer words split. 4.0 is the
# widely cited average for English on cl100k; we model length-dependence rather
# than applying it flatly, which is where the naive chars/4 rule goes wrong on
# code and structured text.
_CHARS_PER_TOKEN = 4.0


def _approx_chunk_tokens(chunk: str) -> int:
    """Estimate the number of BPE tokens for a single pre-token chunk."""
    if not chunk:
        return 0

    stripped = chunk.strip()

    # Pure whitespace. BPE packs runs of spaces efficiently; roughly one token
    # per ~3 spaces, and a newline is generally its own token.
    if not stripped:
        newlines = chunk.count("\n")
        spaces = len(chunk) - newlines
        return max(1, newlines + (spaces + 2) // 3)

    # Digit runs split about every 3 characters on cl100k.
    if stripped.isdigit():
        return max(1, (len(stripped) + 2) // 3)

    # Punctuation rarely merges beyond pairs.
    if not any(c.isalnum() for c in stripped):
        return max(1, (len(stripped) + 1) // 2)

    n = len(stripped)
    # Short words are almost always a single token.
    if n <= 4:
        return 1
    # Everything else scales with length, with a floor of 1.
    est = n / _CHARS_PER_TOKEN
    # CamelCase and snake_case identifiers split at boundaries, which the plain
    # length model underestimates. Add a token per internal boundary.
    boundaries = len(re.findall(r"(?<=[a-z0-9])(?=[A-Z])|_", stripped))
    return max(1, round(est) + boundaries)


@lru_cache(maxsize=4096)
def _approx_tokens_cached(text: str) -> int:
    return sum(_approx_chunk_tokens(c) for c in _PRETOKEN.findall(text))


def approx_tokens(text: str) -> int:
    """Approximate BPE token count for `text` with no dependencies."""
    if not text:
        return 0
    # lru_cache on huge strings wastes memory; only cache the small ones, which
    # is where the repeated-fragment workload actually lives.
    if len(text) <= 4096:
        return _approx_tokens_cached(text)
    return sum(_approx_chunk_tokens(c) for c in _PRETOKEN.findall(text))


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CountResult:
    tokens: int
    exact: bool
    backend: str

    @property
    def accuracy_note(self) -> str:
        if self.exact:
            return "exact (tiktoken)"
        return "approximate (+/- ~10%; `pip install tiktoken` for exact counts)"


class TokenCounter:
    """Counts tokens, preferring an exact backend and degrading gracefully."""

    def __init__(self, encoding: str = "cl100k_base", force_approx: bool = False):
        self.encoding = encoding
        self._enc = None if force_approx else _tiktoken_encoder(encoding)

    @property
    def exact(self) -> bool:
        return self._enc is not None

    @property
    def backend(self) -> str:
        return f"tiktoken/{self.encoding}" if self.exact else "builtin-approx"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:  # pragma: no cover
            try:
                return len(self._enc.encode(text, disallowed_special=()))
            except Exception:
                pass
        return approx_tokens(text)

    def result(self, text: str) -> CountResult:
        return CountResult(self.count(text), self.exact, self.backend)


# Per-message overhead for chat-formatted requests. Each message carries role
# and delimiter tokens on top of its content.
CHAT_MESSAGE_OVERHEAD = 4
CHAT_REQUEST_OVERHEAD = 3


def count_messages(counter: TokenCounter, messages: list) -> int:
    """Count tokens for an OpenAI-style `messages` array, including the
    per-message framing overhead that naive content-only counts miss."""
    total = CHAT_REQUEST_OVERHEAD
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += CHAT_MESSAGE_OVERHEAD
        content = msg.get("content")
        if isinstance(content, str):
            total += counter.count(content)
        elif isinstance(content, list):
            # Multimodal content blocks.
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += counter.count(part["text"])
        if isinstance(msg.get("name"), str):
            total += counter.count(msg["name"])
    return total
