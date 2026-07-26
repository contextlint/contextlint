"""Tests for contextlint.

Run with: python -m pytest -q
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlint.analyzers import CERTAIN, HIGH, run_all
from contextlint.counter import TokenCounter, approx_tokens, count_messages
from contextlint.loaders import Message, Request, load_jsonl, normalise_record
from contextlint.optimizer import (
    _find_json_spans,
    apply_safe_fixes,
    dedupe_sentences,
    measure,
    minify_json_blobs,
    normalise_whitespace,
)


# ---------------------------------------------------------------- counter


def test_approx_tokens_scales_with_length():
    assert approx_tokens("") == 0
    assert approx_tokens("hello") >= 1
    short = approx_tokens("hello world")
    long = approx_tokens("hello world " * 50)
    assert long > short * 20


def test_approx_within_sane_range_of_chars_over_four():
    """The approximation should stay in the same ballpark as the chars/4 rule
    for ordinary English prose, which is the regime it is calibrated for."""
    text = ("The quick brown fox jumps over the lazy dog. " * 40)
    est = approx_tokens(text)
    naive = len(text) / 4
    assert 0.6 * naive < est < 1.7 * naive, (est, naive)


def test_counter_reports_backend_and_is_deterministic():
    tc = TokenCounter(force_approx=True)
    assert tc.exact is False
    assert tc.backend == "builtin-approx"
    assert tc.count("repeatable input") == tc.count("repeatable input")


def test_chat_overhead_is_added_per_message():
    tc = TokenCounter(force_approx=True)
    one = count_messages(tc, [{"role": "user", "content": "hi"}])
    two = count_messages(
        tc, [{"role": "user", "content": "hi"}, {"role": "user", "content": "hi"}]
    )
    assert two > one


def test_multimodal_content_blocks_are_counted():
    tc = TokenCounter(force_approx=True)
    n = count_messages(
        tc,
        [{"role": "user", "content": [{"type": "text", "text": "a longer piece of text"}]}],
    )
    assert n > 4


# ---------------------------------------------------------------- loaders


def test_normalise_openai_shape():
    req = normalise_record(
        {"model": "m", "messages": [{"role": "system", "content": "S"},
                                    {"role": "user", "content": "U"}]},
        "src",
    )
    assert req is not None
    assert req.model == "m"
    assert req.system_text == "S"
    assert req.non_system_text == "U"


def test_normalise_anthropic_top_level_system():
    req = normalise_record(
        {"system": "policy text", "messages": [{"role": "user", "content": "q"}]}, "src"
    )
    assert req.system_text == "policy text"


def test_normalise_extracts_completion_from_choices():
    req = normalise_record(
        {"messages": [{"role": "user", "content": "q"}],
         "choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        "src",
    )
    assert req.completion == "answer"


def test_normalise_unwraps_nested_request_body():
    req = normalise_record(
        {"body": {"model": "x", "messages": [{"role": "user", "content": "q"}]}}, "src"
    )
    assert req is not None and req.model == "x"


def test_loader_skips_malformed_jsonl_lines_without_dying():
    text = '{"messages":[{"role":"user","content":"ok"}]}\nNOT JSON\n\n'
    reqs = load_jsonl(text, "s")
    assert len(reqs) == 1


def test_normalise_rejects_unusable_records():
    assert normalise_record({"nothing": "here"}, "s") is None
    assert normalise_record("a string", "s") is None


# ---------------------------------------------------------------- optimizer


def test_normalise_whitespace_removes_only_slack():
    src = "line one   \n\n\n\nline  two"
    out = normalise_whitespace(src)
    assert out == "line one\n\nline two"


def test_normalise_whitespace_preserves_fenced_code_indentation():
    src = "text\n\n```\ndef f():\n    return 1\n```\n"
    out = normalise_whitespace(src)
    assert "    return 1" in out


def test_find_json_spans_handles_nesting():
    text = 'before {"a": {"b": [1, 2, {"c": 3}]}} after'
    spans = _find_json_spans(text)
    assert spans, "should locate at least one span"
    start, end = spans[0]
    assert json.loads(text[start:end])["a"]["b"][2]["c"] == 3


def test_find_json_spans_ignores_braces_inside_strings():
    text = '{"a": "a } brace in a string", "b": 1}'
    spans = _find_json_spans(text)
    assert spans[0] == (0, len(text))


def test_minify_json_shortens_and_preserves_semantics():
    obj = {"retrieved": [{"id": i, "body": "text " * 5} for i in range(4)]}
    pretty = json.dumps(obj, indent=2)
    out = minify_json_blobs(f"Context:\n{pretty}\nEnd")
    assert len(out) < len(f"Context:\n{pretty}\nEnd")
    spans = _find_json_spans(out)
    assert json.loads(out[spans[0][0]:spans[0][1]]) == obj


def test_minify_leaves_non_json_braces_alone():
    src = "Use {placeholder} in the template.\nAnother {one} here."
    assert minify_json_blobs(src) == src


def test_dedupe_sentences_keeps_first_occurrence_only():
    s = ("Always cite the document id you used in the answer. "
         "Something else entirely goes here. "
         "Always cite the document id you used in the answer.")
    out = dedupe_sentences(s)
    assert out.count("Always cite the document id") == 1
    assert "Something else entirely" in out


def test_dedupe_leaves_short_repeats_alone():
    """Short repeated fragments are often legitimate structure, so the
    de-duplicator must not touch them."""
    s = "Yes. No. Yes. No."
    assert dedupe_sentences(s) == s


def test_apply_safe_fixes_is_idempotent():
    src = 'A sentence that is quite long and repeated later on.   \n\n\n\n{"k": \n  1}\nA sentence that is quite long and repeated later on.'
    once = apply_safe_fixes(src)
    assert apply_safe_fixes(once) == once


def test_apply_safe_fixes_never_increases_tokens():
    tc = TokenCounter(force_approx=True)
    samples = [
        "plain text",
        "",
        "   ",
        '{"a":1}',
        "already\ntight\ntext",
        "Repeated long sentence here for testing purposes. " * 3,
    ]
    for s in samples:
        assert tc.count(apply_safe_fixes(s, tc)) <= tc.count(s), repr(s)


def test_measure_reports_a_real_delta():
    tc = TokenCounter(force_approx=True)
    reqs = [
        Request(messages=[Message("system", "Trailing space here.   \n\n\n\nMore.")],
                weight=10)
    ]
    res = measure(reqs, tc)
    assert res.before_tokens > res.after_tokens
    assert res.saved_per_pass > 0
    assert 0 < res.pct <= 100


# ---------------------------------------------------------------- analyzers


def _many(system: str, n: int, user_prefix: str = "question "):
    return [
        Request(
            messages=[Message("system", system), Message("user", f"{user_prefix}{i}")],
            model="m",
        )
        for i in range(n)
    ]


def test_repeated_system_prompt_is_detected():
    tc = TokenCounter(force_approx=True)
    system = "You must follow these rules carefully in every response. " * 20
    findings = run_all(_many(system, 12), tc)
    checks = {f.check for f in findings}
    assert "repeated-system-prompt" in checks
    f = next(f for f in findings if f.check == "repeated-system-prompt")
    assert f.confidence == HIGH
    assert f.calls_affected == 12
    assert f.tokens_saved > 0


def test_no_repeated_system_finding_for_a_single_call():
    tc = TokenCounter(force_approx=True)
    findings = run_all(_many("Some system prompt " * 30, 1), tc)
    assert "repeated-system-prompt" not in {f.check for f in findings}


def test_cacheable_prefix_fires_only_past_the_cache_floor():
    tc = TokenCounter(force_approx=True)
    # Well over 1024 tokens of identical prefix.
    big = "This is a stable instruction sentence that repeats. " * 400
    findings = run_all(_many(big, 5), tc)
    assert "cacheable-prefix" in {f.check for f in findings}


def test_short_shared_prefix_reports_below_floor_instead():
    tc = TokenCounter(force_approx=True)
    # Must clear the 128-token reporting floor but stay under the 1024-token
    # cacheable minimum, which is the window this finding exists to explain.
    small = "Short shared preamble that is not big enough to cache. " * 30
    findings = run_all(_many(small, 6), tc)
    checks = {f.check for f in findings}
    assert "cacheable-prefix" not in checks
    assert "stable-prefix-below-cache-floor" in checks


def test_clean_prompt_produces_no_certain_findings():
    tc = TokenCounter(force_approx=True)
    reqs = [Request(messages=[Message("user", "Summarise the attached contract.")])]
    findings = run_all(reqs, tc)
    assert not [f for f in findings if f.confidence == CERTAIN]


def test_findings_are_sorted_by_confidence_then_size():
    tc = TokenCounter(force_approx=True)
    system = ("Rule one is stated here at some length.   \n\n\n\n"
              "Rule one is stated here at some length. " + "padding sentence. " * 40)
    findings = run_all(_many(system, 8), tc)
    ranks = [{CERTAIN: 0, HIGH: 1}.get(f.confidence, 2) for f in findings]
    assert ranks == sorted(ranks)


def test_a_broken_check_does_not_kill_the_run(monkeypatch):
    import contextlint.analyzers as A

    def exploding(requests, tc):
        raise RuntimeError("boom")

    monkeypatch.setattr(A, "ALL_CHECKS", [exploding, A.check_whitespace])
    findings = A.run_all(
        [Request(messages=[Message("user", "trailing   \n\n\n\n x")], weight=99)],
        TokenCounter(force_approx=True),
    )
    assert any("boom" in f.detail for f in findings)


# ---------------------------------------------------------------- cli


def _run(args, stdin=""):
    return subprocess.run(
        [sys.executable, "-m", "contextlint", *args],
        input=stdin, capture_output=True, text=True, encoding="utf-8",
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env={**os.environ, "NO_COLOR": "1"},
    )


def test_cli_json_output_is_valid_and_labels_placeholder_rate():
    r = _run(["examples/requests.jsonl", "-f", "json"])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["tool"] == "contextlint"
    assert payload["pricing"]["is_placeholder_rate"] is True
    assert payload["corpus"]["records"] > 0


def test_cli_real_rate_clears_the_placeholder_flag():
    r = _run(["examples/requests.jsonl", "-f", "json", "--price-in", "0.6"])
    assert json.loads(r.stdout)["pricing"]["is_placeholder_rate"] is False


def test_cli_markdown_contains_the_overlap_caveat():
    r = _run(["examples/requests.jsonl", "-f", "markdown"])
    assert r.returncode == 0
    assert "not a sum of findings" in r.stdout


def test_cli_exit_2_on_no_input():
    r = _run(["-"], stdin="")
    assert r.returncode == 2
    assert "no prompts or request records found" in r.stderr


def test_cli_fail_over_pct_gate():
    assert _run(["examples/requests.jsonl", "--fail-over-pct", "1"]).returncode == 1
    assert _run(["examples/requests.jsonl", "--fail-over-pct", "99"]).returncode == 0


def test_cli_rejects_bad_weight():
    assert _run(["examples/requests.jsonl", "--weight", "0"]).returncode == 2


def test_cli_fix_writes_files_and_leaves_originals_untouched():
    before = open("examples/prompts/triage.txt", encoding="utf-8").read()
    with tempfile.TemporaryDirectory() as d:
        r = _run(["examples/prompts", "--fix", d, "-f", "json"])
        assert r.returncode == 0
        written = os.listdir(d)
        assert written
    after = open("examples/prompts/triage.txt", encoding="utf-8").read()
    assert before == after, "--fix must never modify the source files"


def test_cli_projection_scales_with_calls():
    a = json.loads(_run(["examples/requests.jsonl", "-f", "json", "--calls", "1000"]).stdout)
    b = json.loads(_run(["examples/requests.jsonl", "-f", "json", "--calls", "2000"]).stdout)
    assert b["projection"]["monthly_saving_usd"] > a["projection"]["monthly_saving_usd"]


# ------------------------------------------------- tool schemas & history


def _tool(name, desc_len=1):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "What this tool does and when you should reach for it. " * desc_len,
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "the query " * desc_len}},
            },
        },
    }


def _tool_request(tools, called, i=0):
    return normalise_record(
        {
            "model": "m",
            "messages": [{"role": "system", "content": "sys " * 40},
                         {"role": "user", "content": f"q{i}"}],
            "tools": tools,
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"function": {"name": c, "arguments": "{}"}} for c in called],
            }}],
        },
        f"log:{i}",
    )


def test_loader_extracts_tools_and_invocations():
    r = _tool_request([_tool("search"), _tool("unused")], ["search"])
    assert [t["name"] for t in r.tools] == ["search", "unused"]
    assert r.tools_called == ["search"]


def test_loader_extracts_legacy_functions_array():
    r = normalise_record(
        {"messages": [{"role": "user", "content": "q"}],
         "functions": [{"name": "legacy", "description": "d", "parameters": {}}]},
        "s",
    )
    assert [t["name"] for t in r.tools] == ["legacy"]


def test_loader_extracts_anthropic_tool_use():
    r = normalise_record(
        {"messages": [{"role": "user", "content": "q"}],
         "tools": [{"name": "fetch", "description": "d", "input_schema": {}}],
         "content": [{"type": "tool_use", "name": "fetch", "input": {}}]},
        "s",
    )
    assert [t["name"] for t in r.tools] == ["fetch"]
    assert r.tools_called == ["fetch"]


def test_tool_schema_cost_is_reported():
    tc = TokenCounter(force_approx=True)
    tools = [_tool(f"t{i}", desc_len=4) for i in range(8)]
    reqs = [_tool_request(tools, ["t0"], i) for i in range(12)]
    findings = run_all(reqs, tc)
    f = next(f for f in findings if f.check == "tool-schema-cost")
    assert f.confidence == HIGH
    assert f.tokens_saved > 0
    assert f.calls_affected == 12


def test_unused_tools_are_identified():
    tc = TokenCounter(force_approx=True)
    tools = [_tool(f"t{i}", desc_len=4) for i in range(8)]
    reqs = [_tool_request(tools, ["t0"], i) for i in range(12)]
    findings = run_all(reqs, tc)
    f = next(f for f in findings if f.check == "unused-tool-schemas")
    # t1..t7 declared but never called; t0 was called so must not be listed.
    assert "7 tool(s)" in f.title
    assert not any(ev.startswith("t0:") for ev in f.evidence)


def test_no_unused_tool_finding_when_all_tools_are_used():
    tc = TokenCounter(force_approx=True)
    tools = [_tool(f"t{i}", desc_len=4) for i in range(3)]
    reqs = [_tool_request(tools, ["t0", "t1", "t2"], i) for i in range(12)]
    findings = run_all(reqs, tc)
    assert "unused-tool-schemas" not in {f.check for f in findings}


def test_no_tool_findings_when_no_tools_present():
    tc = TokenCounter(force_approx=True)
    findings = run_all(_many("plain system prompt " * 30, 5), tc)
    assert "tool-schema-cost" not in {f.check for f in findings}


def test_unbounded_history_is_detected():
    tc = TokenCounter(force_approx=True)
    turns = []
    for i in range(14):
        turns.append(Message("user" if i % 2 == 0 else "assistant",
                             f"Turn {i}: a reasonably substantial message body here. " * 4))
    reqs = [Request(messages=[Message("system", "sys")] + turns, source="chat.jsonl", weight=50)]
    findings = run_all(reqs, tc)
    f = next(f for f in findings if f.check == "unbounded-history")
    assert f.tokens_saved > 0
    assert "14 turns" in f.evidence[0]


def test_short_conversations_produce_no_history_finding():
    tc = TokenCounter(force_approx=True)
    turns = [Message("user" if i % 2 == 0 else "assistant", f"turn {i}") for i in range(4)]
    findings = run_all([Request(messages=turns, weight=100)], tc)
    assert "unbounded-history" not in {f.check for f in findings}


def test_recoverable_percentage_can_never_exceed_100():
    """Regression guard.

    An earlier version counted tool-schema savings in the numerator while
    excluding tool-schema tokens from the input total, producing a reported
    '566% recoverable'. Any finding that claims savings against a token class
    must have that class included in the corpus denominator.
    """
    r = _run(["examples/agent_requests.jsonl", "-f", "json"])
    payload = json.loads(r.stdout)
    assert 0 <= payload["recoverable_pct_confident"] <= 100, payload["recoverable_pct_confident"]
    assert (
        payload["corpus"]["input_tokens"]
        == payload["corpus"]["message_tokens"] + payload["corpus"]["tool_schema_tokens"]
    )


def test_tool_tokens_are_counted_in_the_corpus_total():
    tc = TokenCounter(force_approx=True)
    from contextlint.report import Summary

    tools = [_tool(f"t{i}", desc_len=4) for i in range(5)]
    with_tools = Summary([_tool_request(tools, ["t0"])], tc)
    without = Summary([_tool_request([], [])], tc)
    assert with_tools.tool_tokens > 0
    assert without.tool_tokens == 0
    assert with_tools.input_tokens > without.input_tokens
