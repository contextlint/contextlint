# contextlint

An open-source project by Rayan El Fayoumi.

**Audit what your LLM prompts actually cost, and separate what's provably safe to cut from what needs an eval.**

No dependencies. No network calls. No API key. Nothing leaves your machine.

```bash
pip install git+https://github.com/contextlint/contextlint.git
contextlint prompts/ --calls 200000 --price-in 0.60
```

> Not on PyPI yet — install from source with the line above. It has no dependencies, so this is
> a single fast download. `pip install contextlint` will work once it's published; I'd rather the
> command in this README actually run today than promise one that doesn't.

```
contextlint — LLM context cost audit
──────────────────────────────────────────────────────────────────

Corpus
  Records analysed     8 from 1 source(s)
  Calls represented    8
  Input tokens         17,724  (avg 2,215/call)

Recoverable
  Measured         7,256 tok   safe fixes applied and re-counted, not summed
  High             2,558 tok   prompt caching of the repeated prefix, on top
  Judgement       12,879 tok   quantified prize, needs an eval to confirm

  55.4% of input tokens are recoverable without touching behaviour.
```

---

## Why this exists

Most teams discover their LLM bill is mostly waste only after it's large. The waste is
boring and invisible per-call: a system prompt re-sent on every request, `indent=2` on a
retrieval payload, a rule pasted in twice by two different people, six few-shot examples
where three would do. None of it shows up in a code review. All of it shows up on the invoice.

`contextlint` measures it. Point it at your prompts or a request log and it tells you how many
tokens are recoverable, how confident you should be in each number, and exactly what to change.

## The thing that makes it different: it doesn't sum overlapping findings

Every other "prompt optimizer" adds its findings together and reports the total. That number
is wrong, because the findings overlap — JSON indentation is *also* whitespace, and a
duplicated block *also* contains duplicated sentences. Adding them up inflates the headline,
sometimes by 2x.

`contextlint` applies every provably-safe transformation to the actual text, re-counts, and
reports the measured difference. That's the `Measured` line. It's a floor you can trust, and
you can verify it yourself:

```bash
contextlint prompts/ --fix ./rewritten     # originals are never modified
diff -u prompts/agent.md rewritten/agent.md
```

## Confidence classes

Findings are graded, and the grades mean something specific.

| Class | What it means |
|---|---|
| **MEASURED** | The tokens were removed and the text re-counted. Removal cannot change model behaviour: trailing whitespace, blank-line runs, minifiable JSON, verbatim duplicate sentences. |
| **HIGH** | A provider feature or mechanical restructuring recovers it — chiefly prompt caching of a stable prefix. No content change required. |
| **JUDGEMENT** | contextlint quantifies the prize but will not claim the saving. Few-shot trimming, filler removal, over-retrieval. **Run your evals.** |

If a tool tells you it can cut 40% of your prompt with no downside and doesn't distinguish
between these, be suspicious. Cutting few-shot examples is not the same kind of act as
stripping trailing spaces.

## What it checks

| Check | Finds |
|---|---|
| `repeated-system-prompt` | A fixed system prompt billed on every call. Usually the single largest line on the bill, and the least noticed. |
| `cacheable-prefix` | A stable leading block over the ~1024-token caching floor that isn't being cached. |
| `stable-prefix-below-cache-floor` | The same repetition, but too short to cache — so it explains *why* caching isn't available to you yet. |
| `duplicate-blocks` | Paragraphs repeated verbatim across calls, outside the cacheable prefix. |
| `restated-instructions` | The same sentence appearing twice in one prompt. |
| `pretty-printed-json` | `indent=2` payloads where compact JSON would do. Found by balanced-brace scanning, not a regex, so nested objects don't hide. |
| `whitespace-bloat` | Trailing spaces, blank-line runs, alignment padding. Never touches fenced code blocks. |
| `filler-phrases` | "You are a helpful assistant", "take a deep breath", "I will tip you", empty hedges. |
| `fewshot-weight` | Large demonstration blocks worth testing at half size. |
| `tool-schema-cost` | Tool/function schemas re-sent on every call. In an agentic loop this multiplies by steps per task — a 6-step task pays for the whole toolbox 6 times. Least-audited line on an agent bill, because it lives in code rather than in the prompt. |
| `unused-tool-schemas` | Tools declared on every call that the model never actually invokes. |
| `unbounded-history` | Full conversation history re-sent each turn, so cost grows with the *square* of conversation length. |
| `io-imbalance` | Huge inputs producing tiny outputs — the signature of over-retrieval. |

## Inputs it understands

**Prompt templates** — `.txt` `.md` `.prompt` `.j2` `.jinja2` `.tmpl` `.yaml` `.yml` `.toml` `.xml`

**Request logs** — `.jsonl` `.ndjson` `.json`, in the OpenAI chat-completions shape, the
Anthropic `system` + `messages` shape, and the export variants emitted by LangChain, LiteLLM,
Helicone and Langfuse. Nested `body`/`request`/`payload` wrappers are unwrapped automatically.
Malformed lines are skipped, not fatal.

```bash
contextlint prompts/                        # a directory of templates
contextlint requests.jsonl                  # a request log
cat prompt.txt | contextlint -              # stdin
contextlint prompts/ requests.jsonl         # both at once
```

## Getting a real number instead of a placeholder

Token counts are measured facts. Dollar figures need your actual rate, so pass it:

```bash
contextlint log.jsonl --price-in 0.60 --price-out 2.40 --calls 200000
```

Without `--price-in`, contextlint uses a generic placeholder and **says so, loudly, every
time**. It will not quietly invent a price for your model.

`--calls N` projects the sample onto your real monthly volume. Without it, the figures
describe only the sample you fed in. If you're auditing templates rather than logs, use
`--weight N` to say how many real calls each template stands for.

## Use it in CI

Stop prompt bloat from creeping back after you fix it:

```yaml
- run: pip install git+https://github.com/contextlint/contextlint.git
- run: contextlint prompts/ --fail-over-pct 10
```

Exit codes: `0` clean · `1` gate exceeded · `2` usage error or no input found.

## Exact vs approximate counts

By default contextlint uses a built-in approximation so it installs with zero dependencies and
runs anywhere. It mimics the GPT-2/cl100k pre-tokenizer and models sub-token splits per chunk,
including identifier boundaries that the naive `chars / 4` rule gets badly wrong on code.
Typical error is under about 10%, and every report states which backend produced the numbers.

For exact counts:

```bash
pip install tiktoken    # alongside contextlint; picked up automatically
```

Conclusions rarely change — a 40% saving doesn't become a 4% saving — but if you're
negotiating a budget on these numbers, use the exact backend.

## All options

```
contextlint [paths...] [options]

  -f, --format {terminal,markdown,json}   output format
  -o, --output FILE                       write to a file
      --price-in USD                      input price per million tokens
      --price-out USD                     output price per million tokens
      --calls N                            monthly call volume to project onto
      --weight N                           real calls each record stands for
      --min-confidence {certain,high,judgement}
      --approx                             force the built-in counter
      --fix DIR                            emit safely-rewritten prompts
      --fail-over-pct PCT                  exit 1 if over PCT% recoverable
      --fail-over-tokens N                 exit 1 if over N tokens recoverable
```

## Privacy

contextlint makes **no network calls of any kind**. There is no telemetry, no phone-home, no
API key, and no upload step. Your prompts are read from disk and the report is written to
your terminal. You can verify this in `loaders.py` and `counter.py` — there is no HTTP client
anywhere in the codebase.

## Limitations, stated plainly

- The approximate counter is an approximation. It is reported as one.
- The prompt-caching saving assumes a conservative 50% discount on cached input reads.
  Most providers do better; check your own cached-input rate.
- `JUDGEMENT` findings are prizes, not promises. contextlint cannot run your evals.
- Image and audio tokens in multimodal requests are not counted; only text is.
- It reads text. It does not know that your prompt is doing something clever, so if a check
  flags something load-bearing, trust yourself over the tool and please open an issue.

## If you want the deep version done for you

The CLI is free, MIT, and always will be — it is not a trial and nothing is held back from it.

Separately, I take on **paid deep audits** as an independent consultant: **$19 USD**, one prompt
set or repo, **delivered within 24 hours**.

You get a written report with measured before/after token counts, the rewritten prompts as a
diff you can apply, a prompt-caching plan specific to your call pattern, and a model-tier
comparison for your actual workload. It covers the things the CLI marks `JUDGEMENT` and can't
decide for you, plus the structural issues that need a human reading your pipeline.

**If the measured safe saving comes in under 15%, you pay nothing.** I'd rather tell you your
prompts are already tight than take $19 for a report that says so.

- **How it's done:** automated analysis plus AI-assisted review, checked by a human before it
  goes out. Saying so up front because you deserve to know what you're buying.
- **Your data:** send redacted prompts if you like — structure is what matters, not your
  content. Files are deleted after delivery and never used to train anything.
- **Payment:** a Stripe payment link — any card, no account needed, nothing to sign up for.
  Sent *after* delivery for first-time customers, so you see the report before you pay anything.
- **Who you're buying from:** Rayan El Fayoumi, sole proprietor, Ontario, Canada. Consulting services are
  invoiced in my own name; ContextLint is the name of the software, not a company.

📧 **rayanelfayoumi+contextlint@gmail.com** — send your prompts or a redacted request log
and roughly how many calls a month you're making.

## Contributing

Issues and PRs welcome. New checks should include a test and must state their confidence
class honestly — a check that can't justify `CERTAIN` should not claim it.

```bash
git clone https://github.com/contextlint/contextlint.git
cd contextlint
python -m pytest -q
```

## License

MIT.
