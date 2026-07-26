# contextlint report

## Corpus

- Records analysed: **17**
- Calls represented: **17**
- Input tokens: **23,537** (avg 1,384/call)
  - messages: 19,619 · tool schemas: 3,918
- Output tokens: **228**
- Counting method: approximate (+/- ~10%; `pip install tiktoken` for exact counts)

## Recoverable

| Class | Tokens | Meaning |
|---|---:|---|
| MEASURED | 7,324 | Safe fixes applied to the text and re-counted — overlapping findings are not double-counted |
| HIGH | 4,556 | Prompt caching of the repeated prefix, on top of the above |
| JUDGEMENT | 9,440 | Quantified prize, needs an eval to confirm |

**50.5% of input tokens are recoverable without changing behaviour.**

> `MEASURED` is the real delta from rewriting the prompts, not a sum of findings. Run with `--fix` to emit the rewritten prompts and verify it.

## Cost impact

Rate used: `$0.60/M` input, `$2.40/M` output.

- Measured input spend: **$0.01**
- Saving without behaviour change: **$0.01**

Projected to **200,000 calls/month** (11764.7x sample):

- Monthly input spend: **$166.14**
- Monthly saving: **$83.86**
- Annualised saving: **$1,006.31**

## Findings (9)

### 1. The same sentence appears more than once in a single prompt

`CERTAIN` · **5,450 tokens** · 9 call(s) · `restated-instructions`

5,450 tokens are spent restating sentences verbatim inside one request. This usually accumulates as prompts are edited by several people over time.

**Fix.** Keep one copy, placed where it belongs in the instruction order. If a rule was repeated because the model kept violating it, repetition is the wrong fix: state it once, positively, with an explicit output contract.

<details><summary>Evidence</summary>

- `examples/requests.jsonl:1: surcharges are applied per the tariff in effect on the ship date and are itemised separ…`
- `examples/requests.jsonl:2: surcharges are applied per the tariff in effect on the ship date and are itemised separ…`
- `examples/requests.jsonl:3: surcharges are applied per the tariff in effect on the ship date and are itemised separ…`

</details>

### 2. Indented JSON is being sent where minified JSON would do

`CERTAIN` · **1,648 tokens** · 8 call(s) · `pretty-printed-json`

1,648 tokens go to JSON indentation and the newlines around it. `json.dumps(obj, indent=2)` is roughly 15-30% more expensive than the compact form and is a very common accidental default.

**Fix.** Serialise with `json.dumps(obj, separators=(",", ":"))`. Models parse compact JSON as reliably as indented JSON. If a human also reads these payloads, indent at the logging layer and send compact to the model.

<details><summary>Evidence</summary>

- `examples/requests.jsonl:1: 206 tok/call`
- `examples/requests.jsonl:2: 206 tok/call`
- `examples/requests.jsonl:3: 206 tok/call`

</details>

### 3. Trailing whitespace, runs of blank lines and padded indentation

`CERTAIN` · **1,027 tokens** · 9 call(s) · `whitespace-bloat`

1,027 tokens are pure formatting slack: trailing spaces, three or more consecutive newlines, and multi-space runs used for visual alignment. Models do not read alignment, but tokenizers bill it.

**Fix.** Strip trailing whitespace, collapse blank-line runs to one, and collapse multi-space runs. This cannot change model behaviour, so it is the safest saving on this list. Best applied as a normalisation step at prompt-build time rather than by hand.

<details><summary>Evidence</summary>

- `examples/requests.jsonl:1: 128 tok/call`
- `examples/requests.jsonl:2: 128 tok/call`
- `examples/requests.jsonl:3: 128 tok/call`

</details>

### 4. A 731-token system prompt is re-sent on all 9 calls

`HIGH` · **2,924 tokens** · 9 call(s) · `repeated-system-prompt`

The same 731-token system prompt is billed on every one of 9 calls, which is 6,579 input tokens in total and 5,848 tokens of pure repetition. This is normally the largest single line on an LLM bill and the one people are least aware of, because per-call it looks small.

**Fix.** Two levers, in this order. First, enable prompt caching so the repetition is billed at the discounted cached-input rate — this requires no change to the prompt's content, only that it stay byte-stable and strictly first. Second, shorten it: a 731-token system prompt has usually accumulated rules that no longer earn their place. The figure shown assumes only caching at a conservative 50% discount; shortening it saves on top of that.

<details><summary>Evidence</summary>

- `you are a very helpful, friendly assistant for northwind logistics. ## your role you answer customer questions about shipments, delivery windows, customs paper…`

</details>

### 5. Tool schemas cost 653 tokens on every call

`HIGH` · **1,632 tokens** · 6 call(s) · `tool-schema-cost`

8 tool schema(s) totalling ~653 tokens are sent with each of 6 calls — 3,918 input tokens in aggregate. In an agentic loop this multiplies by the number of steps per task, so a 6-step task pays for the whole toolbox 6 times. Tool schemas are the least-audited line on an agent bill because they live in code, not in the prompt.

**Fix.** Tool definitions are stable across calls, so put them inside the cached prefix — that alone recovers most of this with no behaviour change. Then trim: JSON Schema `description` fields are usually where the bulk sits, and they can be tightened hard without hurting selection accuracy. Drop `title` fields, collapse deep `properties` nesting, and prefer enums over prose that explains allowed values.

<details><summary>Evidence</summary>

- `track_shipment: 85 tok`
- `reroute_parcel: 85 tok`
- `check_tariff: 83 tok`
- `get_invoice: 82 tok`
- `lookup_customs_docs: 82 tok`

</details>

### 6. Large few-shot block worth testing at half the size

`JUDGEMENT` · **7,632 tokens** · 10 call(s) · `fewshot-weight`

Demonstrations account for a substantial share of these prompts; halving them would free roughly 7,632 tokens. Few-shot count is usually chosen once and never revisited, and accuracy typically plateaus well before the number of examples people settle on.

**Fix.** Run your eval set at full, half, and quarter of the current example count and keep the smallest that holds quality. Prefer diverse examples over many similar ones. If the examples are stable across calls, move them into the cacheable prefix so they cost a fraction even if you keep them all.

<details><summary>Evidence</summary>

- `examples/requests.jsonl:1: ~18 demonstration markers, 1,813 tok`
- `examples/requests.jsonl:2: ~18 demonstration markers, 1,819 tok`
- `examples/requests.jsonl:3: ~18 demonstration markers, 1,819 tok`

</details>

### 7. 3 tool(s) are declared on every call but never invoked

`JUDGEMENT` · **1,440 tokens** · 6 call(s) · `unused-tool-schemas`

Across the logged calls, 3 of 8 declared tools were never actually called, costing 1,440 input tokens to advertise. Toolboxes accumulate: a tool added for one workflow stays in the schema for every other one.

**Fix.** Route tools per task type rather than declaring the union of all of them, so each call advertises only what it might plausibly use. Marked JUDGEMENT because a tool unused in this sample may still be load-bearing for a rarer path — check your traces over a longer window before deleting anything, and consider that a tool never chosen may have a description problem rather than be genuinely unnecessary.

<details><summary>Evidence</summary>

- `reroute_parcel: 85 tok, never called`
- `escalate_to_human: 78 tok, never called`
- `open_claim: 77 tok, never called`

</details>

### 8. Ceremonial phrasing that does not steer the model

`JUDGEMENT` · **368 tokens** · 11 call(s) · `filler-phrases`

368 tokens go to stock phrases with no instructional content: generic assistant framing, empty hedges, and folk prompt-engineering incantations.

**Fix.** Delete them and re-run your evals. Marked JUDGEMENT rather than CERTAIN on purpose: a handful of these do occasionally shift behaviour on specific models, so measure rather than assume. Cut them one group at a time so a regression is attributable.

<details><summary>Evidence</summary>

- `x18  "please note that"  -- empty hedge`
- `x10  "you are a very helpful, friendly assistant"  -- generic assistant framing`
- `x10  "it is important to note that"  -- empty hedge`
- `x10  "do your best"  -- no-op instruction`
- `x10  "think carefully before answering"  -- vague, better stated as a concrete step`

</details>

### 9. Very large inputs producing very small outputs

`JUDGEMENT` · **0 tokens** · 8 call(s) · `io-imbalance`

8 of 8 logged calls send at least 50x more input than they receive output (median ratio across all calls: 79x). A tiny answer extracted from a very large context is the classic signature of over-retrieval: more chunks are being stuffed in than the answer needs.

**Fix.** Instrument which retrieved chunks are actually cited in the answer, then cut top-k until quality moves. Add a reranking step so you can send fewer, better chunks. This check reports no token figure because the right k is an empirical question -- but it is usually the largest single line on a RAG bill.

<details><summary>Evidence</summary>

- `examples/requests.jsonl:1: 2,212 in -> 25 out (88x)`
- `examples/requests.jsonl:5: 2,213 in -> 26 out (85x)`
- `examples/requests.jsonl:8: 2,215 in -> 28 out (79x)`

</details>

---

Token counts are measured. Dollar figures depend on the rate above. `JUDGEMENT` findings are prizes, not promises — measure before shipping.

Generated by [contextlint](https://github.com/contextlint/contextlint).
