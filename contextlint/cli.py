"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .analyzers import CERTAIN, HIGH, JUDGEMENT, run_all
from .counter import TokenCounter
from .loaders import load_inputs
from .optimizer import apply_safe_fixes, measure
from .report import (
    DEFAULT_PRICE_IN,
    DEFAULT_PRICE_OUT,
    Summary,
    render_json,
    render_markdown,
    render_terminal,
)

EPILOG = """\
examples:
  contextlint prompts/                     audit a directory of prompt templates
  contextlint requests.jsonl               audit a request log
  contextlint prompts/ --calls 50000       project the saving at production volume
  contextlint log.jsonl --price-in 0.60    use your real input rate
  cat prompt.txt | contextlint -           audit one prompt from stdin
  contextlint prompts/ --format markdown -o report.md
  contextlint prompts/ --fail-over-pct 15  fail CI if >15% is recoverable

input formats:
  Prompt templates   .txt .md .prompt .j2 .jinja2 .tmpl .yaml .yml .toml .xml
  Request logs       .jsonl .ndjson .json  (OpenAI chat shape, Anthropic
                     system+messages shape, and the LangChain / LiteLLM /
                     Helicone / Langfuse export variants of those)

Nothing leaves your machine. contextlint makes no network calls of any kind.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="contextlint",
        description="Audit LLM prompt and context spend. Finds recoverable "
                    "tokens and separates what is provably safe to cut from "
                    "what needs an eval.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "paths", nargs="*", default=["-"],
        help="files or directories to audit; '-' or omitted reads stdin",
    )
    p.add_argument(
        "--format", "-f", choices=("terminal", "markdown", "json"),
        default="terminal", help="output format (default: terminal)",
    )
    p.add_argument("--output", "-o", metavar="FILE", help="write to FILE instead of stdout")

    money = p.add_argument_group("pricing")
    money.add_argument(
        "--price-in", type=float, default=DEFAULT_PRICE_IN, metavar="USD",
        help=f"input price per million tokens (default: {DEFAULT_PRICE_IN:.2f}, "
             "a PLACEHOLDER — use your real rate)",
    )
    money.add_argument(
        "--price-out", type=float, default=DEFAULT_PRICE_OUT, metavar="USD",
        help=f"output price per million tokens (default: {DEFAULT_PRICE_OUT:.2f})",
    )
    money.add_argument(
        "--calls", type=int, metavar="N",
        help="monthly call volume to project the saving onto; without this, "
             "figures describe only the sample analysed",
    )

    tuning = p.add_argument_group("analysis")
    tuning.add_argument(
        "--weight", type=int, default=1, metavar="N",
        help="treat each record as standing for N real calls (default: 1)",
    )
    tuning.add_argument(
        "--min-confidence", choices=(CERTAIN, HIGH, JUDGEMENT), default=JUDGEMENT,
        help="hide findings below this confidence (default: judgement, i.e. show all)",
    )
    tuning.add_argument(
        "--approx", action="store_true",
        help="force the built-in approximate counter even if tiktoken is installed",
    )

    ci = p.add_argument_group("continuous integration")
    ci.add_argument(
        "--fail-over-pct", type=float, metavar="PCT",
        help="exit 1 if recoverable Certain+High tokens exceed PCT%% of input",
    )
    ci.add_argument(
        "--fail-over-tokens", type=int, metavar="N",
        help="exit 1 if recoverable Certain+High tokens exceed N",
    )

    fix = p.add_argument_group("rewriting")
    fix.add_argument(
        "--fix", metavar="DIR",
        help="write the safely-rewritten prompts into DIR so you can diff and "
             "verify the measured saving yourself; originals are never modified",
    )

    p.add_argument("--version", action="version", version=f"contextlint {__version__}")
    return p


_CONF_ORDER = {CERTAIN: 0, HIGH: 1, JUDGEMENT: 2}


def _force_utf8_streams() -> None:
    """Emit UTF-8 regardless of the platform locale.

    The reports carry box-drawing rules and a warning sign. On Windows a
    redirected stdout falls back to the ANSI code page, which cannot encode
    them, so `contextlint prompts/ > report.md` died with a UnicodeEncodeError
    while the same run to a terminal succeeded. --output already commits to
    UTF-8, so this just makes stdout agree with it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # a capture object or StringIO stood in for the stream
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_streams()
    args = build_parser().parse_args(argv)

    if args.weight < 1:
        print("contextlint: --weight must be 1 or greater", file=sys.stderr)
        return 2
    if args.calls is not None and args.calls < 1:
        print("contextlint: --calls must be 1 or greater", file=sys.stderr)
        return 2

    requests = load_inputs(args.paths)
    if not requests:
        print(
            "contextlint: no prompts or request records found.\n"
            "  Point it at a directory of prompt files, a .jsonl request log,\n"
            "  or pipe a prompt in on stdin. Run `contextlint --help` for formats.",
            file=sys.stderr,
        )
        return 2

    if args.weight > 1:
        for r in requests:
            r.weight = args.weight

    tc = TokenCounter(force_approx=args.approx)
    findings = run_all(requests, tc)
    opt = measure(requests, tc)

    if args.fix:
        written = _write_fixes(requests, tc, args.fix)
        print(f"contextlint: wrote {written} rewritten file(s) to {args.fix}",
              file=sys.stderr)

    cutoff = _CONF_ORDER[args.min_confidence]
    visible = [f for f in findings if _CONF_ORDER.get(f.confidence, 3) <= cutoff]

    summary = Summary(requests, tc)

    if args.format == "terminal" and args.output is None:
        render_terminal(visible, summary, tc, args.price_in, args.price_out,
                        args.calls, opt=opt)
    else:
        if args.format == "json":
            text = render_json(visible, summary, tc, args.price_in,
                               args.price_out, args.calls, opt=opt)
        elif args.format == "markdown":
            text = render_markdown(visible, summary, tc, args.price_in,
                                   args.price_out, args.calls, opt=opt)
        else:
            import io

            buf = io.StringIO()
            render_terminal(visible, summary, tc, args.price_in, args.price_out,
                            args.calls, opt=opt, stream=buf)
            text = buf.getvalue()

        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(text if text.endswith("\n") else text + "\n")
            except OSError as exc:
                print(f"contextlint: cannot write {args.output}: {exc}", file=sys.stderr)
                return 2
            print(f"contextlint: wrote {args.output}", file=sys.stderr)
        else:
            print(text)

    # CI gates. Uses the measured rewrite delta plus caching, matching the
    # headline the report prints, so a green CI run and a clean report agree.
    safe = opt.saved_per_pass + sum(
        f.tokens_saved for f in visible if f.confidence == HIGH
    )
    if args.fail_over_tokens is not None and safe > args.fail_over_tokens:
        print(
            f"contextlint: {safe:,} recoverable tokens exceeds "
            f"--fail-over-tokens {args.fail_over_tokens:,}",
            file=sys.stderr,
        )
        return 1
    if args.fail_over_pct is not None and summary.input_tokens:
        pct = safe / summary.input_tokens * 100
        if pct > args.fail_over_pct:
            print(
                f"contextlint: {pct:.1f}% recoverable exceeds "
                f"--fail-over-pct {args.fail_over_pct:.1f}%",
                file=sys.stderr,
            )
            return 1
    return 0


def _write_fixes(requests, tc, outdir: str) -> int:
    """Emit rewritten prompt files alongside a diff-friendly layout."""
    import os

    os.makedirs(outdir, exist_ok=True)
    written = 0
    seen = {}
    for r in requests:
        base = os.path.basename(r.source.split(":")[0]) or "stdin"
        stem, ext = os.path.splitext(base)
        # Rewrites are plain text regardless of the source container, so a
        # .jsonl source becomes .txt rather than pretending to still be a log.
        if ext.lower() in (".jsonl", ".ndjson", ".json"):
            ext = ".txt"
        seen[base] = seen.get(base, 0) + 1
        suffix = f".{seen[base]}" if seen[base] > 1 else ""
        target = os.path.join(outdir, f"{stem}{suffix}{ext or '.txt'}")
        body = []
        for m in r.messages:
            fixed = apply_safe_fixes(m.content, tc)
            if tc.count(fixed) > tc.count(m.content):
                fixed = m.content
            if len(r.messages) > 1:
                body.append(f"<<<{m.role}>>>\n{fixed}")
            else:
                body.append(fixed)
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("\n\n".join(body) + "\n")
            written += 1
        except OSError:
            continue
    return written


if __name__ == "__main__":
    raise SystemExit(main())
