"""Report rendering: terminal, markdown and JSON.

Presentation rule: token counts are facts, dollar figures are derived from a
rate the user supplies. The default rate is a labelled placeholder, and the
report says so every single time, because an unlabelled dollar figure built on
a guessed price is worse than no dollar figure at all.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import List, Optional, Sequence

from .analyzers import CERTAIN, HIGH, JUDGEMENT, Finding
from .counter import TokenCounter, count_messages
from .loaders import Request
from .optimizer import OptimisationResult

# A generic mid-tier input/output rate in USD per million tokens. This is a
# PLACEHOLDER, not a claim about any specific model's price. Every surface that
# shows a dollar figure also shows this caveat.
DEFAULT_PRICE_IN = 3.00
DEFAULT_PRICE_OUT = 15.00

_CONF_LABEL = {
    CERTAIN: "CERTAIN",
    HIGH: "HIGH",
    JUDGEMENT: "JUDGEMENT",
}


def _supports_colour(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class _Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t): return self._wrap("1", t)
    def dim(self, t): return self._wrap("2", t)
    def red(self, t): return self._wrap("31", t)
    def green(self, t): return self._wrap("32", t)
    def yellow(self, t): return self._wrap("33", t)
    def cyan(self, t): return self._wrap("36", t)


class Summary:
    """Corpus-level totals."""

    def __init__(self, requests: Sequence[Request], tc: TokenCounter):
        import json as _json

        self.request_records = len(requests)
        self.calls = sum(r.weight for r in requests)
        self.message_tokens = 0
        self.tool_tokens = 0
        self.output_tokens = 0
        self.sources = set()

        for r in requests:
            payload = [{"role": m.role, "content": m.content} for m in r.messages]
            self.message_tokens += count_messages(tc, payload) * r.weight
            # Tool/function schemas are billed as input too. Excluding them here
            # while the tool-schema checks report savings against them would
            # produce a nonsense recoverable percentage.
            for t in r.tools:
                self.tool_tokens += tc.count(
                    _json.dumps(t, separators=(",", ":"), ensure_ascii=False)
                ) * r.weight
            if r.completion:
                self.output_tokens += tc.count(r.completion) * r.weight
            self.sources.add(r.source.split(":")[0])

    @property
    def input_tokens(self) -> int:
        """Everything billed as input: messages plus tool schemas."""
        return self.message_tokens + self.tool_tokens

    @property
    def avg_input(self) -> int:
        return self.input_tokens // self.calls if self.calls else 0


def money(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000.0 * price_per_million


def render_terminal(
    findings: List[Finding],
    summary: Summary,
    tc: TokenCounter,
    price_in: float,
    price_out: float,
    calls_per_month: Optional[int],
    opt: Optional[OptimisationResult] = None,
    stream=None,
) -> None:
    stream = stream or sys.stdout
    s = _Style(_supports_colour(stream))
    w = stream.write

    scale = 1.0
    if calls_per_month and summary.calls:
        scale = calls_per_month / summary.calls

    w("\n" + s.bold("contextlint") + s.dim(" — LLM context cost audit") + "\n")
    w(s.dim("─" * 66) + "\n\n")

    # ---- Corpus ----
    w(s.bold("Corpus\n"))
    w(f"  Records analysed     {summary.request_records:,}"
      f" from {len(summary.sources)} source(s)\n")
    w(f"  Calls represented    {summary.calls:,}\n")
    w(f"  Input tokens         {summary.input_tokens:,}"
      f"  (avg {summary.avg_input:,}/call)\n")
    if summary.tool_tokens:
        w(f"    of which messages  {summary.message_tokens:,}\n")
        w(f"    of which tools     {summary.tool_tokens:,}"
          f"  {s.dim('(tool schemas are input too)')}\n")
    if summary.output_tokens:
        w(f"  Output tokens        {summary.output_tokens:,}\n")
    w(f"  Token counts         {tc.result('').accuracy_note}\n\n")

    if not findings and (opt is None or opt.saved_per_pass == 0):
        w(s.green("  No recoverable waste found above reporting thresholds.\n"))
        w(s.dim("  That is a genuine result, not an error — some prompt sets are\n"
                "  already tight. Try --calls to model production volume, or point\n"
                "  contextlint at a request log to enable the retrieval checks.\n\n"))
        return

    recoverable = {CERTAIN: 0, HIGH: 0, JUDGEMENT: 0}
    for f in findings:
        recoverable[f.confidence] = recoverable.get(f.confidence, 0) + f.tokens_saved

    measured = opt.saved_per_pass if opt else 0
    safe = measured + recoverable[HIGH]
    pct = (safe / summary.input_tokens * 100) if summary.input_tokens else 0.0

    # ---- Headline ----
    w(s.bold("Recoverable\n"))
    w(f"  {s.green('Measured')}  {measured:>12,} tok"
      f"   {s.dim('safe fixes applied and re-counted, not summed')}\n")
    w(f"  {s.cyan('High')}      {recoverable[HIGH]:>12,} tok"
      f"   {s.dim('prompt caching of the repeated prefix, on top')}\n")
    w(f"  {s.yellow('Judgement')} {recoverable[JUDGEMENT]:>12,} tok"
      f"   {s.dim('quantified prize, needs an eval to confirm')}\n")
    w(f"\n  {s.bold(f'{pct:.1f}% of input tokens')} are recoverable without "
      f"touching behaviour.\n")
    w(s.dim("  'Measured' is the real delta from rewriting the text, so findings\n"
            "  that overlap are not counted twice. Use --fix to emit the rewrites.\n\n"))

    # ---- Money ----
    w(s.bold("Cost impact\n"))
    w(s.dim(f"  Rate used: ${price_in:.2f}/M input, ${price_out:.2f}/M output.\n"))
    if price_in == DEFAULT_PRICE_IN:
        w(s.yellow("  ^ PLACEHOLDER RATE. Pass --price-in with the figure from your\n"
                   "    own provider bill for a real number.\n"))
    w(f"  Current input spend, as measured   ${money(summary.input_tokens, price_in):,.2f}\n")
    w(f"  Saving, no behaviour change       "
      f"{s.green(f'${money(safe, price_in):,.2f}')}\n")
    if calls_per_month:
        w(s.dim(f"\n  Projected to {calls_per_month:,} calls/month "
                f"({scale:.1f}x the analysed sample):\n"))
        w(f"    Monthly input spend             ${money(int(summary.input_tokens * scale), price_in):,.2f}\n")
        w(f"    Monthly saving                  "
          f"{s.green(f'${money(int(safe * scale), price_in):,.2f}')}\n")
        w(f"    Annualised saving               "
          f"{s.green(f'${money(int(safe * scale * 12), price_in):,.2f}')}\n")
    w("\n")

    # ---- Findings ----
    w(s.bold(f"Findings ({len(findings)})\n"))
    w(s.dim("─" * 66) + "\n")
    for i, f in enumerate(findings, 1):
        colour = {CERTAIN: s.green, HIGH: s.cyan, JUDGEMENT: s.yellow}[f.confidence]
        tag = colour(f"[{_CONF_LABEL[f.confidence]}]")
        w(f"\n{i}. {s.bold(f.title)}\n")
        w(f"   {tag}  {f.tokens_saved:,} tok")
        if f.calls_affected:
            w(f"  ·  {f.calls_affected:,} call(s) affected")
        w(f"  ·  {s.dim(f.check)}\n")
        w(f"\n   {_wrap_text(f.detail, 3)}\n")
        w(f"\n   {s.bold('Fix:')} {_wrap_text(f.fix, 3, first_indent=False)}\n")
        if f.evidence:
            w(f"\n   {s.dim('Evidence:')}\n")
            for ev in f.evidence:
                w(f"     {s.dim('·')} {ev}\n")
    w("\n" + s.dim("─" * 66) + "\n")
    w(s.dim("Token counts are measured. Dollar figures depend on the rate above.\n"))
    w(s.dim("JUDGEMENT findings are prizes, not promises — measure before shipping.\n\n"))


def _wrap_text(text: str, indent: int, width: int = 63, first_indent: bool = True) -> str:
    import textwrap

    pad = " " * indent
    lines = textwrap.wrap(text, width=width) or [""]
    out = lines[0] if not first_indent else lines[0]
    for line in lines[1:]:
        out += "\n" + pad + line
    return out


def render_markdown(
    findings: List[Finding],
    summary: Summary,
    tc: TokenCounter,
    price_in: float,
    price_out: float,
    calls_per_month: Optional[int],
    opt: Optional[OptimisationResult] = None,
) -> str:
    measured = opt.saved_per_pass if opt else 0
    high = sum(f.tokens_saved for f in findings if f.confidence == HIGH)
    safe = measured + high
    pct = (safe / summary.input_tokens * 100) if summary.input_tokens else 0.0

    lines = [
        "# contextlint report",
        "",
        "## Corpus",
        "",
        f"- Records analysed: **{summary.request_records:,}**",
        f"- Calls represented: **{summary.calls:,}**",
        f"- Input tokens: **{summary.input_tokens:,}** (avg {summary.avg_input:,}/call)",
    ]
    if summary.tool_tokens:
        lines.append(f"  - messages: {summary.message_tokens:,} · "
                     f"tool schemas: {summary.tool_tokens:,}")
    lines += [
    ]
    if summary.output_tokens:
        lines.append(f"- Output tokens: **{summary.output_tokens:,}**")
    lines += [
        f"- Counting method: {tc.result('').accuracy_note}",
        "",
        "## Recoverable",
        "",
        "| Class | Tokens | Meaning |",
        "|---|---:|---|",
        f"| MEASURED | {measured:,} | Safe fixes applied to the text and re-counted "
        "— overlapping findings are not double-counted |",
        f"| HIGH | {high:,} | Prompt caching of the repeated prefix, on top of the above |",
        f"| JUDGEMENT | {sum(f.tokens_saved for f in findings if f.confidence == JUDGEMENT):,} "
        "| Quantified prize, needs an eval to confirm |",
    ]

    lines += [
        "",
        f"**{pct:.1f}% of input tokens are recoverable without changing behaviour.**",
        "",
        "> `MEASURED` is the real delta from rewriting the prompts, not a sum of "
        "findings. Run with `--fix` to emit the rewritten prompts and verify it.",
        "",
        "## Cost impact",
        "",
        f"Rate used: `${price_in:.2f}/M` input, `${price_out:.2f}/M` output.",
    ]
    if price_in == DEFAULT_PRICE_IN:
        lines.append("")
        lines.append("> ⚠️ **Placeholder rate.** Pass `--price-in` with the figure "
                     "from your own provider bill for a real number.")
    lines += [
        "",
        f"- Measured input spend: **${money(summary.input_tokens, price_in):,.2f}**",
        f"- Saving without behaviour change: **${money(safe, price_in):,.2f}**",
    ]
    if calls_per_month and summary.calls:
        scale = calls_per_month / summary.calls
        lines += [
            "",
            f"Projected to **{calls_per_month:,} calls/month** ({scale:.1f}x sample):",
            "",
            f"- Monthly input spend: **${money(int(summary.input_tokens * scale), price_in):,.2f}**",
            f"- Monthly saving: **${money(int(safe * scale), price_in):,.2f}**",
            f"- Annualised saving: **${money(int(safe * scale * 12), price_in):,.2f}**",
        ]

    lines += ["", f"## Findings ({len(findings)})", ""]
    if not findings:
        lines.append("No recoverable waste found above reporting thresholds.")
    for i, f in enumerate(findings, 1):
        lines += [
            f"### {i}. {f.title}",
            "",
            f"`{_CONF_LABEL[f.confidence]}` · **{f.tokens_saved:,} tokens** · "
            f"{f.calls_affected:,} call(s) · `{f.check}`",
            "",
            f.detail,
            "",
            f"**Fix.** {f.fix}",
        ]
        if f.evidence:
            lines += ["", "<details><summary>Evidence</summary>", ""]
            lines += [f"- `{ev}`" for ev in f.evidence]
            lines += ["", "</details>"]
        lines.append("")

    lines += [
        "---",
        "",
        "Token counts are measured. Dollar figures depend on the rate above. "
        "`JUDGEMENT` findings are prizes, not promises — measure before shipping.",
        "",
        "Generated by [contextlint](https://github.com/contextlint/contextlint).",
    ]
    return "\n".join(lines)


def render_json(
    findings: List[Finding],
    summary: Summary,
    tc: TokenCounter,
    price_in: float,
    price_out: float,
    calls_per_month: Optional[int],
    opt: Optional[OptimisationResult] = None,
) -> str:
    measured = opt.saved_per_pass if opt else 0
    high = sum(f.tokens_saved for f in findings if f.confidence == HIGH)
    safe = measured + high
    payload = {
        "tool": "contextlint",
        "counting": {
            "backend": tc.backend,
            "exact": tc.exact,
        },
        "corpus": {
            "records": summary.request_records,
            "calls": summary.calls,
            "input_tokens": summary.input_tokens,
            "message_tokens": summary.message_tokens,
            "tool_schema_tokens": summary.tool_tokens,
            "output_tokens": summary.output_tokens,
            "avg_input_tokens_per_call": summary.avg_input,
        },
        "pricing": {
            "input_per_million_usd": price_in,
            "output_per_million_usd": price_out,
            "is_placeholder_rate": price_in == DEFAULT_PRICE_IN,
        },
        "recoverable_tokens": {
            "measured_safe_rewrite": measured,
            "high_prompt_caching": high,
            "judgement": sum(
                f.tokens_saved for f in findings if f.confidence == JUDGEMENT
            ),
            "_note": "measured_safe_rewrite is a re-count after applying the safe "
                     "transformations, so overlapping findings are not double counted; "
                     "per-finding tokens_saved values are attributions and may overlap",
        },
        "recoverable_tokens_confident": safe,
        "recoverable_pct_confident": (
            round(safe / summary.input_tokens * 100, 2) if summary.input_tokens else 0.0
        ),
        "estimated_saving_usd_confident": round(money(safe, price_in), 4),
        "findings": [asdict(f) for f in findings],
    }
    if calls_per_month and summary.calls:
        scale = calls_per_month / summary.calls
        payload["projection"] = {
            "calls_per_month": calls_per_month,
            "scale_factor": round(scale, 4),
            "monthly_saving_usd": round(money(int(safe * scale), price_in), 2),
            "annual_saving_usd": round(money(int(safe * scale * 12), price_in), 2),
        }
    return json.dumps(payload, indent=2)
