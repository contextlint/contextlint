"""Input loading.

contextlint accepts the shapes people actually have lying around:

  * a directory of prompt templates (.txt .md .prompt .j2 .tmpl .yaml)
  * a JSONL request log, one JSON object per line, in the near-universal
    OpenAI chat-completions shape ({"messages": [...]}), including the
    variants emitted by LangChain, LiteLLM, Helicone and Langfuse exports
  * an Anthropic-shaped record ({"system": ..., "messages": [...]})
  * a single file, or stdin

Everything is normalised into a list of `Request` objects so the analyzers
never need to care where the data came from.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".prompt", ".prompts",
    ".j2", ".jinja", ".jinja2", ".tmpl", ".template",
    ".yaml", ".yml", ".toml", ".xml",
}
LOG_SUFFIXES = {".jsonl", ".ndjson"}
JSON_SUFFIXES = {".json"}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".tox", "site-packages", ".idea", ".vscode",
}

MAX_FILE_BYTES = 8 * 1024 * 1024


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Request:
    """One logical model call (or one prompt template treated as a call)."""

    messages: List[Message] = field(default_factory=list)
    completion: Optional[str] = None
    model: Optional[str] = None
    source: str = "<unknown>"
    # Tool/function schemas sent with the request. These are re-sent on every
    # call and are a large, almost never measured cost in agent systems.
    tools: List[dict] = field(default_factory=list)
    # Names of tools the model actually invoked, so unused schemas can be found.
    tools_called: List[str] = field(default_factory=list)
    # How many real calls this record stands for. Prompt templates default to 1
    # but the user can scale with --calls to model production volume.
    weight: int = 1

    @property
    def system_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system")

    @property
    def non_system_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role != "system")

    @property
    def full_text(self) -> str:
        return "\n".join(m.content for m in self.messages)


# --------------------------------------------------------------------------
# Normalisation of arbitrary log records
# --------------------------------------------------------------------------

_CONTENT_KEYS = ("content", "text", "value", "prompt", "input")


def _stringify_content(content) -> str:
    """Flatten the several shapes a message body can take."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in _CONTENT_KEYS:
                    val = item.get(key)
                    if isinstance(val, str):
                        parts.append(val)
                        break
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in _CONTENT_KEYS:
            val = content.get(key)
            if isinstance(val, str):
                return val
    return ""


def _extract_tools(obj: dict) -> List[dict]:
    """Pull tool/function schemas from any of the shapes providers use."""
    out: List[dict] = []
    for key in ("tools", "functions"):
        raw = obj.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            # OpenAI tools wrap the schema under "function"; Anthropic and the
            # legacy OpenAI "functions" array are flat.
            inner = item.get("function") if isinstance(item.get("function"), dict) else item
            if isinstance(inner, dict) and inner.get("name"):
                out.append(inner)
    return out


def _extract_tools_called(obj: dict) -> List[str]:
    """Find which tools the model actually invoked in the response."""
    names: List[str] = []

    choices = obj.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict) and fn.get("name"):
                        names.append(str(fn["name"]))
            fc = msg.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                names.append(str(fc["name"]))

    # Anthropic: tool_use blocks in the response content.
    content = obj.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
                names.append(str(block["name"]))

    # Assistant messages inside the request carry prior tool calls too.
    for m in obj.get("messages") or []:
        if not isinstance(m, dict):
            continue
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function")
                if isinstance(fn, dict) and fn.get("name"):
                    names.append(str(fn["name"]))
    return names


def _extract_completion(obj: dict) -> Optional[str]:
    """Pull the model's response out of a log record if it is present."""
    for key in ("completion", "response", "output", "output_text"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict):
            flat = _stringify_content(val)
            if flat:
                return flat

    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                flat = _stringify_content(msg.get("content"))
                if flat:
                    return flat
            if isinstance(first.get("text"), str):
                return first["text"]
    return None


def normalise_record(obj, source: str) -> Optional[Request]:
    """Turn one parsed JSON object into a Request, or None if unusable."""
    if not isinstance(obj, dict):
        return None

    # Some exporters nest the actual request one level down.
    for wrapper in ("body", "request", "payload", "kwargs"):
        inner = obj.get(wrapper)
        if isinstance(inner, dict) and ("messages" in inner or "prompt" in inner):
            merged = dict(inner)
            for k in ("model", "response", "choices", "completion", "output",
                      "tools", "functions", "content"):
                if k in obj and k not in merged:
                    merged[k] = obj[k]
            obj = merged
            break

    messages: List[Message] = []

    # Anthropic-style top-level system parameter.
    sys_param = obj.get("system")
    if sys_param:
        text = _stringify_content(sys_param)
        if text:
            messages.append(Message("system", text))

    raw_messages = obj.get("messages")
    if isinstance(raw_messages, list):
        for m in raw_messages:
            if not isinstance(m, dict):
                if isinstance(m, str):
                    messages.append(Message("user", m))
                continue
            role = m.get("role") or m.get("from") or "user"
            text = _stringify_content(
                m.get("content", m.get("text", m.get("value")))
            )
            if text:
                messages.append(Message(str(role).lower(), text))
    elif isinstance(obj.get("prompt"), str):
        messages.append(Message("user", obj["prompt"]))
    elif isinstance(obj.get("input"), str):
        messages.append(Message("user", obj["input"]))

    if not messages:
        return None

    model = obj.get("model")
    return Request(
        messages=messages,
        completion=_extract_completion(obj),
        model=model if isinstance(model, str) else None,
        source=source,
        tools=_extract_tools(obj),
        tools_called=_extract_tools_called(obj),
    )


# --------------------------------------------------------------------------
# File and directory walking
# --------------------------------------------------------------------------


def _read_text(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None


def load_jsonl(text: str, source: str) -> List[Request]:
    out: List[Request] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        req = normalise_record(obj, f"{source}:{lineno}")
        if req:
            out.append(req)
    return out


def load_json(text: str, source: str) -> List[Request]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, list):
        out = []
        for i, item in enumerate(obj):
            req = normalise_record(item, f"{source}[{i}]")
            if req:
                out.append(req)
        return out
    req = normalise_record(obj, source)
    return [req] if req else []


def load_prompt_file(text: str, source: str) -> List[Request]:
    """Treat a plain prompt template as a single system-role request."""
    if not text.strip():
        return []
    return [Request(messages=[Message("system", text)], source=source)]


def load_path(path: str) -> List[Request]:
    """Load a file or recursively walk a directory."""
    if os.path.isfile(path):
        return _load_one_file(path)

    if not os.path.isdir(path):
        return []

    requests: List[Request] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(files):
            suffix = os.path.splitext(name)[1].lower()
            if suffix in TEXT_SUFFIXES | LOG_SUFFIXES | JSON_SUFFIXES:
                requests.extend(_load_one_file(os.path.join(root, name)))
    return requests


def _load_one_file(path: str) -> List[Request]:
    suffix = os.path.splitext(path)[1].lower()
    text = _read_text(path)
    if text is None:
        return []
    if suffix in LOG_SUFFIXES:
        return load_jsonl(text, path)
    if suffix in JSON_SUFFIXES:
        return load_json(text, path)
    return load_prompt_file(text, path)


def load_stdin() -> List[Request]:
    text = sys.stdin.read()
    if not text.strip():
        return []
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        # Could be JSONL or a single JSON document; try both.
        reqs = load_jsonl(text, "<stdin>")
        if reqs:
            return reqs
        return load_json(text, "<stdin>")
    return load_prompt_file(text, "<stdin>")


def load_inputs(paths: Iterable[str]) -> List[Request]:
    paths = list(paths)
    if not paths or paths == ["-"]:
        return load_stdin()
    requests: List[Request] = []
    for p in paths:
        if p == "-":
            requests.extend(load_stdin())
        else:
            requests.extend(load_path(p))
    return requests
