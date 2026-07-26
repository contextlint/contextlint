"""The checks.

Every finding must answer three questions honestly:

  1. How many tokens are provably being spent on this?
  2. What fraction of that is recoverable, and how confident are we?
  3. What is the exact change that recovers it?

A finding that cannot answer all three is noise, and contextlint does not emit it.
Savings are separated by confidence so that a "you can save 38%" headline is
never built out of guesses:

  CERTAIN   - the tokens are provably redundant; removing them cannot change
              model behaviour (whitespace, minifiable JSON, byte-identical
              duplicated blocks).
  HIGH      - a well-supported provider feature or a mechanical restructuring
              recovers them (prompt caching of a stable prefix).
  JUDGEMENT - a human must decide; contextlint quantifies the prize but will not
              claim the saving.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .counter import TokenCounter, count_messages
from .loaders import Request

CERTAIN = "certain"
HIGH = "high"
JUDGEMENT = "judgement"

_CONFIDENCE_RANK = {CERTAIN: 0, HIGH: 1, JUDGEMENT: 2}


@dataclass
class Finding:
    check: str
    title: str
    confidence: str
    tokens_saved: int             # total across the whole corpus, per pass
    detail: str
    fix: str
    calls_affected: int = 0
    evidence: List[str] = field(default_factory=list)

    @property
    def sort_key(self):
        return (_CONFIDENCE_RANK.get(self.confidence, 3), -self.tokens_saved)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _truncate(text: str, n: int = 88) -> str:
    flat = _WS.sub(" ", text).strip()
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def _total_weight(requests: Sequence[Request]) -> int:
    return sum(r.weight for r in requests)


# --------------------------------------------------------------------------
# 1. Cacheable stable prefix  -- usually the single largest win
# --------------------------------------------------------------------------

# Providers that support prompt caching impose a minimum cacheable length.
# 1024 tokens is the most common floor; below it caching is unavailable.
CACHE_MIN_TOKENS = 1024

# Cached input tokens are billed at a large discount on every major provider
# that offers the feature. We use a deliberately conservative 0.5 (50% off)
# rather than the 0.9 some providers advertise, so the projected saving is a
# floor rather than a best case.
CACHE_DISCOUNT = 0.5


def check_cacheable_prefix(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    """Find a stable leading block shared across many calls.

    If N calls all start with the same long preamble, that preamble is being
    paid for N times at full price. Prompt caching bills it once at full price
    and the rest at a steep discount. This is the highest-value and most
    frequently missed saving in production LLM systems.
    """
    findings: List[Finding] = []
    by_model: Dict[Optional[str], List[Request]] = defaultdict(list)
    for r in requests:
        by_model[r.model].append(r)

    for model, group in by_model.items():
        weight = _total_weight(group)
        if len(group) < 2 or weight < 2:
            continue

        texts = [r.full_text for r in group if r.full_text]
        if len(texts) < 2:
            continue

        prefix = texts[0]
        for t in texts[1:]:
            limit = min(len(prefix), len(t))
            i = 0
            while i < limit and prefix[i] == t[i]:
                i += 1
            prefix = prefix[:i]
            if len(prefix) < 64:
                break

        if len(prefix) < 64:
            continue

        prefix_tokens = tc.count(prefix)
        if prefix_tokens < CACHE_MIN_TOKENS:
            # Still worth reporting if the repetition is large in aggregate,
            # but as a restructuring rather than a caching finding.
            if prefix_tokens >= 128 and weight >= 3:
                repeated = prefix_tokens * (weight - 1)
                findings.append(
                    Finding(
                        check="stable-prefix-below-cache-floor",
                        title="Repeated preamble is too short to cache",
                        confidence=JUDGEMENT,
                        tokens_saved=repeated,
                        calls_affected=weight,
                        detail=(
                            f"{weight} calls share an identical {prefix_tokens}-token "
                            f"opening block, costing {repeated:,} redundant tokens in "
                            f"aggregate. That is below the ~{CACHE_MIN_TOKENS}-token "
                            "minimum most providers require for prompt caching, so "
                            "caching cannot be switched on as-is."
                        ),
                        fix=(
                            "Either consolidate more of the stable instructions into "
                            f"the prefix to push it past {CACHE_MIN_TOKENS} tokens and "
                            "then enable prompt caching, or shorten the preamble so "
                            "you stop paying for it on every call."
                        ),
                        evidence=[_truncate(prefix)],
                    )
                )
            continue

        # Cacheable. Saving applies to every call after the first.
        billable_now = prefix_tokens * weight
        billable_cached = prefix_tokens + prefix_tokens * (weight - 1) * (1 - CACHE_DISCOUNT)
        saved = int(billable_now - billable_cached)

        label = f" (model={model})" if model else ""
        findings.append(
            Finding(
                check="cacheable-prefix",
                title=f"Stable {prefix_tokens:,}-token prefix is not being cached{label}",
                confidence=HIGH,
                tokens_saved=saved,
                calls_affected=weight,
                detail=(
                    f"All {weight} calls begin with a byte-identical "
                    f"{prefix_tokens:,}-token block, so you are paying full input "
                    f"price for it {weight} times ({billable_now:,} tokens). It "
                    f"clears the ~{CACHE_MIN_TOKENS}-token prompt-caching floor, so "
                    "it qualifies to be billed once and then at a discount."
                ),
                fix=(
                    "Enable prompt caching and mark this prefix as the cached "
                    "segment. Keep it byte-stable and strictly first in the "
                    "request: any variable content (timestamps, user IDs, "
                    "retrieved chunks) placed before or inside it invalidates the "
                    f"cache. Projected saving assumes a conservative "
                    f"{int(CACHE_DISCOUNT * 100)}% discount on cached reads; check "
                    "your provider's actual cached-input rate, which is often better."
                ),
                evidence=[_truncate(prefix, 160)],
            )
        )

    return findings


# --------------------------------------------------------------------------
# 1b. The system prompt re-sent on every single call
# --------------------------------------------------------------------------


def check_repeated_system(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    """Report the single most common real-world cost: one fixed system prompt
    paid for on every call.

    Deliberately separate from the caching check. That one only fires when the
    shared prefix clears the provider's cacheable minimum; this one fires
    whenever the repetition exists, because the operator needs to see the
    aggregate cost of their system prompt whether or not caching is available.
    """
    groups: Dict[str, int] = defaultdict(int)
    tokens: Dict[str, int] = {}
    for r in requests:
        sys_text = r.system_text.strip()
        if not sys_text:
            continue
        key = _norm(sys_text)
        groups[key] += r.weight
        tokens.setdefault(key, tc.count(sys_text))

    findings: List[Finding] = []
    for key, weight in sorted(groups.items(), key=lambda kv: -kv[1]):
        t = tokens[key]
        if weight < 2 or t < 50:
            continue
        redundant = t * (weight - 1)
        if redundant < 200:
            continue
        findings.append(
            Finding(
                check="repeated-system-prompt",
                title=f"A {t:,}-token system prompt is re-sent on all {weight:,} calls",
                confidence=HIGH,
                tokens_saved=int(redundant * CACHE_DISCOUNT),
                calls_affected=weight,
                detail=(
                    f"The same {t:,}-token system prompt is billed on every one of "
                    f"{weight:,} calls, which is {t * weight:,} input tokens in total "
                    f"and {redundant:,} tokens of pure repetition. This is normally "
                    "the largest single line on an LLM bill and the one people are "
                    "least aware of, because per-call it looks small."
                ),
                fix=(
                    "Two levers, in this order. First, enable prompt caching so the "
                    "repetition is billed at the discounted cached-input rate — this "
                    "requires no change to the prompt's content, only that it stay "
                    f"byte-stable and strictly first. Second, shorten it: a {t:,}-token "
                    "system prompt has usually accumulated rules that no longer earn "
                    "their place. The figure shown assumes only caching at a "
                    f"conservative {int(CACHE_DISCOUNT * 100)}% discount; shortening "
                    "it saves on top of that."
                ),
                evidence=[_truncate(key, 160)],
            )
        )
        break  # report the dominant one only; the rest is noise
    return findings


# --------------------------------------------------------------------------
# 2. Duplicated blocks across the corpus
# --------------------------------------------------------------------------


def check_duplicate_blocks(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    """Find substantial paragraphs repeated verbatim across different calls.

    Distinct from the prefix check: these are blocks that recur but not
    necessarily at the start, so caching will not catch them. They usually
    indicate copy-pasted instruction blocks that should be a single shared
    constant, and often reveal instructions duplicated *within* one request.
    """
    # Only non-system text. Repetition of the system prompt is attributed to
    # check_repeated_system, and counting it here as well would double-count the
    # same tokens across two findings.
    counts: Counter = Counter()
    weights: Counter = Counter()
    for r in requests:
        seen_here = set()
        for block in re.split(r"\n\s*\n", r.non_system_text):
            key = _norm(block)
            if len(key) < 80:
                continue
            counts[key] += 1
            if key not in seen_here:
                weights[key] += r.weight
                seen_here.add(key)

    findings: List[Finding] = []
    total = 0
    examples: List[str] = []
    affected = 0
    for key, n in counts.most_common(40):
        if n < 2:
            continue
        block_tokens = tc.count(key)
        if block_tokens < 24:
            continue
        wasted = block_tokens * (weights[key] - 1)
        if wasted < 40:
            continue
        total += wasted
        affected = max(affected, weights[key])
        if len(examples) < 4:
            examples.append(f"x{weights[key]} ({block_tokens} tok): {_truncate(key)}")

    if total >= 100:
        findings.append(
            Finding(
                check="duplicate-blocks",
                title="Identical instruction blocks repeated across calls",
                confidence=CERTAIN,
                tokens_saved=total,
                calls_affected=affected,
                detail=(
                    f"{total:,} tokens are spent re-sending paragraphs that appear "
                    "verbatim in more than one request and are not part of the "
                    "cacheable leading prefix."
                ),
                fix=(
                    "Hoist these into one shared constant and move them into the "
                    "stable prefix so caching covers them. Where the same block "
                    "appears twice inside a single request, delete the second copy "
                    "outright: restating an instruction does not make a model "
                    "follow it more reliably, and it measurably crowds the context."
                ),
                evidence=examples,
            )
        )
    return findings


# --------------------------------------------------------------------------
# 3. Whitespace bloat  -- small per call, real at volume, zero risk
# --------------------------------------------------------------------------


def check_whitespace(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    total = 0
    affected = 0
    examples: List[str] = []

    for r in requests:
        text = r.full_text
        if not text:
            continue
        cleaned = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        if cleaned == text:
            continue
        delta = tc.count(text) - tc.count(cleaned)
        if delta <= 0:
            continue
        total += delta * r.weight
        affected += r.weight
        if len(examples) < 3:
            examples.append(f"{r.source}: {delta} tok/call")

    if total < 25:
        return []
    return [
        Finding(
            check="whitespace-bloat",
            title="Trailing whitespace, runs of blank lines and padded indentation",
            confidence=CERTAIN,
            tokens_saved=total,
            calls_affected=affected,
            detail=(
                f"{total:,} tokens are pure formatting slack: trailing spaces, three "
                "or more consecutive newlines, and multi-space runs used for visual "
                "alignment. Models do not read alignment, but tokenizers bill it."
            ),
            fix=(
                "Strip trailing whitespace, collapse blank-line runs to one, and "
                "collapse multi-space runs. This cannot change model behaviour, so "
                "it is the safest saving on this list. Best applied as a "
                "normalisation step at prompt-build time rather than by hand."
            ),
            evidence=examples,
        )
    ]


# --------------------------------------------------------------------------
# 4. Pretty-printed JSON embedded in prompts
# --------------------------------------------------------------------------

def check_pretty_json(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    # Uses a balanced-brace scan rather than a regex: nested objects defeat both
    # greedy and non-greedy matching, which silently misses the large embedded
    # retrieval payloads that are the whole point of this check.
    from .optimizer import _find_json_spans

    total = 0
    affected = 0
    examples: List[str] = []

    for r in requests:
        req_saving = 0
        text = r.full_text
        for start, end in _find_json_spans(text):
            blob = text[start:end]
            if "\n" not in blob or len(blob) < 40:
                continue
            try:
                parsed = json.loads(blob)
            except (json.JSONDecodeError, ValueError):
                continue
            minified = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
            delta = tc.count(blob) - tc.count(minified)
            if delta > 8:
                req_saving += delta
        if req_saving:
            total += req_saving * r.weight
            affected += r.weight
            if len(examples) < 3:
                examples.append(f"{r.source}: {req_saving} tok/call")

    if total < 25:
        return []
    return [
        Finding(
            check="pretty-printed-json",
            title="Indented JSON is being sent where minified JSON would do",
            confidence=CERTAIN,
            tokens_saved=total,
            calls_affected=affected,
            detail=(
                f"{total:,} tokens go to JSON indentation and the newlines around "
                "it. `json.dumps(obj, indent=2)` is roughly 15-30% more expensive "
                "than the compact form and is a very common accidental default."
            ),
            fix=(
                'Serialise with `json.dumps(obj, separators=(",", ":"))`. Models '
                "parse compact JSON as reliably as indented JSON. If a human also "
                "reads these payloads, indent at the logging layer and send compact "
                "to the model."
            ),
            evidence=examples,
        )
    ]


# --------------------------------------------------------------------------
# 5. Filler and ceremony
# --------------------------------------------------------------------------

# Curated deliberately narrowly. Every entry is a phrase that carries no
# instruction a model acts on differently, and that appears constantly in
# real prompts. Politeness and role-play framing are excluded when they
# plausibly steer behaviour -- the aim is precision, not a big list.
FILLER_PATTERNS = [
    (r"\byou are a (?:very )?(?:helpful|friendly|useful)(?:,? (?:helpful|friendly|useful))* (?:ai )?assistant\b\.?", "generic assistant framing"),
    (r"\bplease note that\b", "empty hedge"),
    (r"\bit (?:is|'s) important (?:to note )?that\b", "empty hedge"),
    (r"\bkeep in mind that\b", "empty hedge"),
    (r"\bas (?:an|a) (?:ai|language model)\b", "self-reference boilerplate"),
    (r"\bdo your best\b", "no-op instruction"),
    (r"\btake a deep breath\b", "no-op instruction"),
    (r"\bthink (?:about it )?carefully(?: before answering)?\b", "vague, better stated as a concrete step"),
    (r"\bi (?:will|'ll) tip you\b[^.\n]*", "no-op instruction"),
    (r"\bthis is very important (?:to|for) my career\b", "no-op instruction"),
    (r"\bfeel free to\b", "filler"),
    (r"\bin order to\b", "wordy: use 'to'"),
    (r"\bat this point in time\b", "wordy: use 'now'"),
    (r"\bdue to the fact that\b", "wordy: use 'because'"),
]

_COMPILED_FILLER = [(re.compile(p, re.IGNORECASE), why) for p, why in FILLER_PATTERNS]


def check_filler(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    total = 0
    affected = 0
    hits: Counter = Counter()
    reasons: Dict[str, str] = {}

    for r in requests:
        req_saving = 0
        for rx, why in _COMPILED_FILLER:
            for m in rx.finditer(r.full_text):
                phrase = m.group(0)
                t = tc.count(phrase)
                if t <= 0:
                    continue
                req_saving += t
                hits[_norm(phrase)[:60]] += r.weight
                reasons[_norm(phrase)[:60]] = why
        if req_saving:
            total += req_saving * r.weight
            affected += r.weight

    if total < 15:
        return []

    examples = [
        f'x{n}  "{phrase}"  -- {reasons.get(phrase, "")}'
        for phrase, n in hits.most_common(5)
    ]
    return [
        Finding(
            check="filler-phrases",
            title="Ceremonial phrasing that does not steer the model",
            confidence=JUDGEMENT,
            tokens_saved=total,
            calls_affected=affected,
            detail=(
                f"{total:,} tokens go to stock phrases with no instructional "
                "content: generic assistant framing, empty hedges, and folk "
                "prompt-engineering incantations."
            ),
            fix=(
                "Delete them and re-run your evals. Marked JUDGEMENT rather than "
                "CERTAIN on purpose: a handful of these do occasionally shift "
                "behaviour on specific models, so measure rather than assume. Cut "
                "them one group at a time so a regression is attributable."
            ),
            evidence=examples,
        )
    ]


# --------------------------------------------------------------------------
# 6. Restated instructions inside a single prompt
# --------------------------------------------------------------------------


def check_redundant_sentences(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    total = 0
    affected = 0
    examples: List[str] = []

    for r in requests:
        seen: Dict[str, int] = {}
        req_saving = 0
        for sentence in _SENTENCE_SPLIT.split(r.full_text):
            key = _norm(sentence)
            if len(key) < 40:
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                req_saving += tc.count(key)
        if req_saving > 10:
            total += req_saving * r.weight
            affected += r.weight
            if len(examples) < 3:
                dupes = [k for k, v in seen.items() if v > 1]
                if dupes:
                    examples.append(f"{r.source}: {_truncate(dupes[0])}")

    if total < 25:
        return []
    return [
        Finding(
            check="restated-instructions",
            title="The same sentence appears more than once in a single prompt",
            confidence=CERTAIN,
            tokens_saved=total,
            calls_affected=affected,
            detail=(
                f"{total:,} tokens are spent restating sentences verbatim inside "
                "one request. This usually accumulates as prompts are edited by "
                "several people over time."
            ),
            fix=(
                "Keep one copy, placed where it belongs in the instruction order. "
                "If a rule was repeated because the model kept violating it, "
                "repetition is the wrong fix: state it once, positively, with an "
                "explicit output contract."
            ),
            evidence=examples,
        )
    ]


# --------------------------------------------------------------------------
# 7. Few-shot example weight
# --------------------------------------------------------------------------

_EXAMPLE_MARKER = re.compile(
    r"^\s*(?:###\s*)?(?:example|input|output|q|a|user|assistant)\s*\d*\s*[:\-]",
    re.IGNORECASE | re.MULTILINE,
)


def check_fewshot(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    total = 0
    affected = 0
    examples: List[str] = []

    for r in requests:
        text = r.full_text
        markers = _EXAMPLE_MARKER.findall(text)
        if len(markers) < 6:
            continue
        # Estimate the span occupied by demonstrations: from the first marker on.
        first = _EXAMPLE_MARKER.search(text)
        if not first:
            continue
        span = text[first.start():]
        span_tokens = tc.count(span)
        if span_tokens < 200:
            continue
        # Report the prize as halving the demonstration count, a common and
        # usually safe reduction -- but flagged JUDGEMENT because only an eval
        # can confirm it for a given task.
        prize = span_tokens // 2
        total += prize * r.weight
        affected += r.weight
        if len(examples) < 3:
            examples.append(
                f"{r.source}: ~{len(markers)} demonstration markers, {span_tokens:,} tok"
            )

    if total < 100:
        return []
    return [
        Finding(
            check="fewshot-weight",
            title="Large few-shot block worth testing at half the size",
            confidence=JUDGEMENT,
            tokens_saved=total,
            calls_affected=affected,
            detail=(
                f"Demonstrations account for a substantial share of these prompts; "
                f"halving them would free roughly {total:,} tokens. Few-shot count "
                "is usually chosen once and never revisited, and accuracy typically "
                "plateaus well before the number of examples people settle on."
            ),
            fix=(
                "Run your eval set at full, half, and quarter of the current "
                "example count and keep the smallest that holds quality. Prefer "
                "diverse examples over many similar ones. If the examples are "
                "stable across calls, move them into the cacheable prefix so they "
                "cost a fraction even if you keep them all."
            ),
            evidence=examples,
        )
    ]


# --------------------------------------------------------------------------
# 8. Input/output imbalance (only when completions are present)
# --------------------------------------------------------------------------


def check_io_ratio(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    pairs = [(r, r.completion) for r in requests if r.completion]
    if len(pairs) < 3:
        return []

    ratios = []
    for r, completion in pairs:
        inp = count_messages(tc, [{"role": m.role, "content": m.content} for m in r.messages])
        out = tc.count(completion)
        if out > 0:
            ratios.append((inp / out, inp, out, r.source))

    if not ratios:
        return []
    ratios.sort(reverse=True)
    worst = [x for x in ratios if x[0] >= 50]
    if len(worst) < max(2, len(ratios) // 10):
        return []

    median = ratios[len(ratios) // 2][0]
    return [
        Finding(
            check="io-imbalance",
            title="Very large inputs producing very small outputs",
            confidence=JUDGEMENT,
            tokens_saved=0,
            calls_affected=len(worst),
            detail=(
                f"{len(worst)} of {len(ratios)} logged calls send at least 50x more "
                f"input than they receive output (median ratio across all calls: "
                f"{median:.0f}x). A tiny answer extracted from a very large context "
                "is the classic signature of over-retrieval: more chunks are being "
                "stuffed in than the answer needs."
            ),
            fix=(
                "Instrument which retrieved chunks are actually cited in the "
                "answer, then cut top-k until quality moves. Add a reranking step "
                "so you can send fewer, better chunks. This check reports no token "
                "figure because the right k is an empirical question -- but it is "
                "usually the largest single line on a RAG bill."
            ),
            evidence=[
                f"{src}: {inp:,} in -> {out:,} out ({ratio:.0f}x)"
                for ratio, inp, out, src in worst[:3]
            ],
        )
    ]



# --------------------------------------------------------------------------
# 9. Tool / function schemas re-sent on every call
# --------------------------------------------------------------------------


def check_tool_schema_bloat(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    """The agent-system cost almost nobody measures.

    Tool schemas are input tokens. They are re-sent, in full, on every single
    call in the loop — and an agent that takes eight steps to finish a task
    pays for the whole toolbox eight times. Two separate findings come out of
    this: the aggregate cost of the schemas, and the schemas belonging to tools
    the model never actually calls.
    """
    findings: List[Finding] = []

    with_tools = [r for r in requests if r.tools]
    if not with_tools:
        return findings

    weight = _total_weight(with_tools)
    total_schema_tokens = 0
    per_tool: Dict[str, int] = {}
    declared: Dict[str, int] = defaultdict(int)
    called: Counter = Counter()

    for r in with_tools:
        for t in r.tools:
            name = str(t.get("name", "?"))
            cost = tc.count(json.dumps(t, separators=(",", ":"), ensure_ascii=False))
            per_tool.setdefault(name, cost)
            declared[name] += r.weight
            total_schema_tokens += cost * r.weight
        for name in r.tools_called:
            called[name] += r.weight

    avg_per_call = total_schema_tokens // weight if weight else 0
    if avg_per_call >= 100:
        findings.append(
            Finding(
                check="tool-schema-cost",
                title=f"Tool schemas cost {avg_per_call:,} tokens on every call",
                confidence=HIGH,
                # Tool definitions are stable, so caching recovers the repeats.
                tokens_saved=int(
                    (total_schema_tokens - avg_per_call) * CACHE_DISCOUNT
                ),
                calls_affected=weight,
                detail=(
                    f"{len(per_tool)} tool schema(s) totalling ~{avg_per_call:,} tokens are "
                    f"sent with each of {weight:,} calls — {total_schema_tokens:,} input "
                    "tokens in aggregate. In an agentic loop this multiplies by the number "
                    "of steps per task, so a 6-step task pays for the whole toolbox 6 times. "
                    "Tool schemas are the least-audited line on an agent bill because they "
                    "live in code, not in the prompt."
                ),
                fix=(
                    "Tool definitions are stable across calls, so put them inside the cached "
                    "prefix — that alone recovers most of this with no behaviour change. Then "
                    "trim: JSON Schema `description` fields are usually where the bulk sits, "
                    "and they can be tightened hard without hurting selection accuracy. Drop "
                    "`title` fields, collapse deep `properties` nesting, and prefer enums over "
                    "prose that explains allowed values."
                ),
                evidence=[
                    f"{n}: {c} tok" for n, c in
                    sorted(per_tool.items(), key=lambda kv: -kv[1])[:5]
                ],
            )
        )

    # Unused tools — only meaningful when we can actually see what was called.
    if called:
        never = {
            n: per_tool[n] for n in per_tool
            if called.get(n, 0) == 0 and declared[n] >= max(2, weight // 2)
        }
        if never:
            wasted = sum(per_tool[n] * declared[n] for n in never)
            if wasted >= 200:
                findings.append(
                    Finding(
                        check="unused-tool-schemas",
                        title=f"{len(never)} tool(s) are declared on every call but never invoked",
                        confidence=JUDGEMENT,
                        tokens_saved=wasted,
                        calls_affected=weight,
                        detail=(
                            f"Across the logged calls, {len(never)} of {len(per_tool)} declared "
                            f"tools were never actually called, costing {wasted:,} input tokens "
                            "to advertise. Toolboxes accumulate: a tool added for one workflow "
                            "stays in the schema for every other one."
                        ),
                        fix=(
                            "Route tools per task type rather than declaring the union of all "
                            "of them, so each call advertises only what it might plausibly "
                            "use. Marked JUDGEMENT because a tool unused in this sample may "
                            "still be load-bearing for a rarer path — check your traces over a "
                            "longer window before deleting anything, and consider that a tool "
                            "never chosen may have a description problem rather than be "
                            "genuinely unnecessary."
                        ),
                        evidence=[f"{n}: {c} tok, never called" for n, c in
                                  sorted(never.items(), key=lambda kv: -kv[1])[:5]],
                    )
                )
    return findings


# --------------------------------------------------------------------------
# 10. Unbounded conversation history
# --------------------------------------------------------------------------


def check_long_history(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    """Full-history resends grow cost quadratically in turn count.

    If turn N re-sends turns 1..N-1, total spend over a conversation scales with
    the square of its length. Most chat implementations do this by default and
    only discover it when the bill arrives.
    """
    long_ones = []
    for r in requests:
        turns = [m for m in r.messages if m.role in ("user", "assistant")]
        if len(turns) >= 10:
            hist_tokens = sum(tc.count(m.content) for m in turns[:-1])
            if hist_tokens >= 500:
                long_ones.append((r, len(turns), hist_tokens))

    if not long_ones:
        return []

    weight = sum(r.weight for r, _, _ in long_ones)
    # A sliding window of the last 6 turns is a common, usually safe default.
    WINDOW = 6
    saved = 0
    for r, n_turns, _ in long_ones:
        turns = [m for m in r.messages if m.role in ("user", "assistant")]
        dropped = turns[:-WINDOW] if len(turns) > WINDOW else []
        saved += sum(tc.count(m.content) for m in dropped) * r.weight

    if saved < 200:
        return []

    worst = max(long_ones, key=lambda x: x[2])
    return [
        Finding(
            check="unbounded-history",
            title=f"Full conversation history re-sent on {weight:,} call(s)",
            confidence=JUDGEMENT,
            tokens_saved=saved,
            calls_affected=weight,
            detail=(
                f"{len(long_ones)} call(s) carry 10 or more conversational turns, the longest "
                f"spending {worst[2]:,} tokens on history alone. Because turn N re-sends turns "
                "1 to N-1, total cost over a conversation grows with the square of its length "
                "— doubling the conversation length roughly quadruples the spend."
            ),
            fix=(
                f"Cap history at a sliding window (the figure here assumes the last {WINDOW} "
                "turns) and summarise what falls out of it into a short running digest placed "
                "in the stable prefix, so it gets cached rather than re-billed. Marked "
                "JUDGEMENT because the right window is task-dependent: coreference-heavy "
                "conversations need more, task-oriented ones usually need far less than they "
                "currently keep."
            ),
            evidence=[
                f"{r.source}: {n} turns, {h:,} tok of history"
                for r, n, h in sorted(long_ones, key=lambda x: -x[2])[:3]
            ],
        )
    ]

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

ALL_CHECKS = [
    check_cacheable_prefix,
    check_repeated_system,
    check_duplicate_blocks,
    check_whitespace,
    check_pretty_json,
    check_filler,
    check_redundant_sentences,
    check_fewshot,
    check_io_ratio,
    check_tool_schema_bloat,
    check_long_history,
]


def run_all(requests: Sequence[Request], tc: TokenCounter) -> List[Finding]:
    findings: List[Finding] = []
    for check in ALL_CHECKS:
        try:
            findings.extend(check(requests, tc))
        except Exception as exc:  # a broken check must not kill the report
            findings.append(
                Finding(
                    check=getattr(check, "__name__", "unknown"),
                    title="Check failed to run",
                    confidence=JUDGEMENT,
                    tokens_saved=0,
                    detail=f"{type(exc).__name__}: {exc}",
                    fix="Please open an issue with a redacted sample that reproduces this.",
                )
            )
    findings.sort(key=lambda f: f.sort_key)
    return findings
