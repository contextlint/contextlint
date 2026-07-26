"""Safe transformations, applied for real so the headline number is measured.

The problem this solves: individual findings overlap. JSON indentation is also
whitespace; a duplicated block also contains duplicated sentences. Summing the
per-finding token counts therefore *overstates* the total, sometimes badly.

So contextlint does not sum. It applies every CERTAIN-confidence transformation to
the actual text, re-counts, and reports the measured difference. That number is
a floor you can trust, and `--fix` writes out the transformed prompts so you can
verify it yourself rather than taking the tool's word for it.

Every transformation here must satisfy one rule: it removes tokens the model
cannot act on differently. Anything requiring judgement lives in analyzers.py as
a JUDGEMENT finding and is deliberately not applied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .counter import TokenCounter
from .loaders import Request

_SENTENCE_SPLIT_KEEP = re.compile(r"(?<=[.!?])(\s+)|(\n{2,})")
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------
# 1. Whitespace normalisation
# --------------------------------------------------------------------------


def normalise_whitespace(text: str) -> str:
    out = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)   # trailing space
    out = re.sub(r"\n{3,}", "\n\n", out)                      # blank-line runs
    # Collapse multi-space runs, but never inside a fenced code block, where
    # indentation is semantic.
    parts = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", out)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", parts[i])
    return "".join(parts).strip("\n")


# --------------------------------------------------------------------------
# 2. JSON minification, using a balanced scan rather than a regex
# --------------------------------------------------------------------------


def _find_json_spans(text: str) -> List[Tuple[int, int]]:
    """Locate top-level {...} / [...] spans by balanced scanning.

    A regex cannot do this: nested objects defeat both greedy and non-greedy
    matching. This walks the string tracking depth and string state, which is
    what actually finds a pretty-printed payload embedded in a prompt.
    """
    spans: List[Tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        opener, closer = ch, "}" if ch == "{" else "]"
        depth = 0
        in_str = False
        escaped = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    spans.append((i, j + 1))
                    break
            j += 1
        if depth != 0 or j >= n:
            i += 1
        else:
            i = j + 1
    return spans


def minify_json_blobs(text: str, counter: Optional[TokenCounter] = None) -> str:
    """Replace pretty-printed JSON with its compact equivalent.

    Only rewrites a span if it parses as JSON, contains a newline (so it is
    genuinely pretty-printed), and the compact form is actually shorter.
    """
    spans = _find_json_spans(text)
    if not spans:
        return text
    out: List[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        blob = text[start:end]
        if "\n" not in blob or len(blob) < 40:
            continue
        try:
            parsed = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        compact = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        if len(compact) >= len(blob):
            continue
        out.append(text[cursor:start])
        out.append(compact)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# --------------------------------------------------------------------------
# 3. Verbatim sentence de-duplication within a single message
# --------------------------------------------------------------------------


def dedupe_sentences(text: str, min_chars: int = 40) -> str:
    """Drop later verbatim repeats of a substantial sentence.

    Conservative by construction: only exact matches after whitespace and case
    normalisation, only sentences of `min_chars` or more, and the first
    occurrence always survives in its original position.
    """
    # Split into sentence-ish units while preserving the separators.
    units = re.split(r"(?<=[.!?])(?=\s)|(?<=\n)", text)
    seen = set()
    kept: List[str] = []
    for unit in units:
        key = _WS.sub(" ", unit).strip().lower()
        if len(key) >= min_chars:
            if key in seen:
                continue
            seen.add(key)
        kept.append(unit)
    result = "".join(kept)
    return re.sub(r"[ \t]{2,}", " ", result)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@dataclass
class OptimisationResult:
    before_tokens: int
    after_tokens: int
    calls: int

    @property
    def saved_per_pass(self) -> int:
        return max(0, self.before_tokens - self.after_tokens)

    @property
    def pct(self) -> float:
        if not self.before_tokens:
            return 0.0
        return self.saved_per_pass / self.before_tokens * 100


def apply_safe_fixes(text: str, counter: Optional[TokenCounter] = None) -> str:
    """Apply every provably-safe transformation, in the order that composes."""
    out = minify_json_blobs(text, counter)
    out = dedupe_sentences(out)
    out = normalise_whitespace(out)
    return out


def measure(requests: Sequence[Request], tc: TokenCounter) -> OptimisationResult:
    """Measure the real, non-double-counted saving from the safe fixes."""
    before = after = 0
    calls = 0
    for r in requests:
        for m in r.messages:
            b = tc.count(m.content)
            a = tc.count(apply_safe_fixes(m.content, tc))
            before += b * r.weight
            after += min(a, b) * r.weight
        calls += r.weight
    return OptimisationResult(before_tokens=before, after_tokens=after, calls=calls)


def rewrite_request(r: Request, tc: TokenCounter) -> Request:
    """Return a copy of `r` with safe fixes applied to every message."""
    import copy

    clone = copy.deepcopy(r)
    for m in clone.messages:
        fixed = apply_safe_fixes(m.content, tc)
        if tc.count(fixed) <= tc.count(m.content):
            m.content = fixed
    return clone
