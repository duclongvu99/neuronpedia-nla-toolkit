# API notes: where the live NLA API and its published spec disagree

Everything here was observed against `https://www.neuronpedia.org/api/nla/*` on **2026-08-15**, checked against the OpenAPI spec embedded in <https://www.neuronpedia.org/api-doc>. Each item states what the spec says, what the API does, and how it was checked, so you can re-verify rather than take it on trust.

None of this is a complaint about Neuronpedia. The API is good and free. These are the gaps a client author trips over.

---

## 1. `/explain` returns reconstruction metrics that the spec omits

**Spec** documents each entry of `results[]` as `{position, token, description}`.

**Actual** response carries fourteen fields:

```
token, token_id, position, l2_norm, description, mse, cosine_similarity,
generated, fragment_index, fragment_count, role, section, channel, message_index
```

The three that matter:

| Field | Meaning | Observed range |
|---|---|---|
| `cosine_similarity` | reconstruction fidelity, 1.0 is perfect | 0.60 to 0.994 |
| `mse` | reconstruction error, lower is better | 0.0059 to 0.165 |
| `l2_norm` | magnitude of the original activation | ~58k to ~69k on gemma L41 |

This is the most consequential omission in the spec. Without `cosine_similarity` you cannot distinguish a description that reflects the activation from one that is confabulated, and the natural-language output gives you no other signal. Measured means on one shared prompt: gemma `kitft-l41` 0.9795, llama `kitft-l53` 0.8815, qwen `andyxu-l18` 0.6610.

Top level also carries an undocumented `cacheId`, and echoes `messages` back when you supply messages rather than text.

---

## 2. `/completion` returns two different shapes depending on the model

**Spec** documents one shape: `{completion, full_text, tokens}`.

**Actual**, for a model whose `model.openRouterAvailable` is `true` (`gemma-3-27b-it`, `llama3.3-70b-it`):

```json
{"completion": "391", "full_text": "<|begin_of_text|>...391<|eot_id|>", "tokens": [...]}
```

**Actual**, for `qwen2.5-1.5b-it`, whose `openRouterAvailable` is `false`:

```json
{"tokens": [...], "prompt_length": 49, "text": "<|im_start|>system\n...391<|im_end|>"}
```

No `completion` key, no `full_text` key. The generated text is present but only inside `text`, concatenated with the prompt. A client that reads `response["full_text"]` will `KeyError` on Qwen and a client that reads `response["completion"]` will silently get `None`.

Note also that `openRouterAvailable: false` does **not** mean generation is unavailable. Qwen generated correctly (17 x 23 asked, `391` returned); it simply takes a different server-side path. The flag describes Neuronpedia's routing, not your capability.

`nla.py` normalises both into one `Completion` object where `full_text` and `tokens` are always populated.

---

## 3. The `generated` flag is not a prompt-vs-completion signal

`results[].generated` looks like it should tell you which tokens the model produced. It does not, at least not when you pass `text=`.

Passing a chat-templated `text` string and explaining positions that are unambiguously inside the assistant turn returned `generated: false` for every one of them. The server cannot recover that structure from a flat string.

**Use the role metadata from `/completion` instead.** Its `tokens[]` entries carry fully populated `role` and `section`:

```
pos 2   'user'             role='user'       section='header'
pos 4   'Explain'          role='user'       section='content'
pos 16  'model'            role='assistant'  section='header'
pos 18  'Code'             role='assistant'  section='content'
pos 42  '<end_of_turn>'    role='assistant'  section='footer'
```

Filter on `role == "assistant" AND section == "content"`. Filtering on `role` alone sweeps in the chat scaffold (`<start_of_turn>`, `model`, `<end_of_turn>`), which also carries `role='assistant'`. This was a real bug in an early version of this client.

---

## 4. Authentication is not enforced

**Spec** declares `security: [{apiKey: []}, {}]` with `apiKey` as header `x-api-key`. The trailing `{}` does mean auth is optional, which is easy to miss.

**Actual**, tested four ways against `/explain` and `/sources`:

| Request | Result |
|---|---|
| Valid key | 200 |
| Deliberately invalid key (`sk-np-totallyinvalid...`) | 200, full results |
| No key header | 200, full results |

The `x-limit-remaining` header decremented identically in all three cases, so the quota follows the **IP address, not the key**. Issuing everyone on a team their own key does not increase total throughput; it is one shared bucket per source IP.

Practical consequences: you cannot raise your limit by authenticating, and a shared office or campus IP shares one budget. Also, do not assume this will stay true. Send a key if you have one.

---

## 5. Rate limits are per endpoint, per IP, and visible in a header

The spec mentions 120/hr for `/explain` and 240/hr for `/completion` in prose. It does not mention that the remaining count is returned on every response:

```
x-limit-remaining: 114
```

Observed buckets, each independent: `/sources` around 1200/hr, `/completion` 240/hr, `/explain` 120/hr. `nla.py` exposes the latest value as `client.limit_remaining`.

---

## 6. The 16-position cap counts new positions per request, not per prompt

The spec says "at most 16 *new* (uncached) positions" per request. Worth spelling out what that means in practice:

- Requesting 26 positions in one call returns `400` with a genuinely helpful message: `received 17 new positions (0 of 17 requested were already cached); the limit is 16 new positions per request`.
- Splitting into two requests of 16 and 10 works, and results merge cleanly.
- Re-requesting the same 26 positions afterwards succeeds in a single pass, because they are all cached now. Timing: 22.2s cold, 2.7s warm.
- **But the cached request still consumes your hourly quota.** The 120/hr counter dropped by 2 on the warm repeat. Caching exempts you from the 16-new cap, not from the rate limit.

Cache determinism was checked: 26/26 descriptions were byte-identical on repeat for a fixed `(text, modelId, nlaSourceId, temperature)`.

---

## 7. Generation is not reproducible across sessions

Not a spec discrepancy so much as a property nobody documents, and the one most likely to damage an experiment.

`/completion` routes through OpenRouter, and which backend provider serves a request can change between sessions. On `llama3.3-70b-it` at `temperature=0.0` with an identical prompt:

- Four consecutive calls in one window: byte-identical.
- Varying only `completion_tokens` across 5, 12, 20, 20, 32: consistent shared prefix, exactly as greedy decoding should behave.
- The same call in a later window: `"## Step 1: Determine the cost of one pen"`, where earlier it had returned `"To find out how much you pay, you need to multiply..."`.

So `temperature=0` gives determinism within a window, not across them.

**Mitigation:** generate once, persist `full_text`, and drive all subsequent `/explain` calls from the saved string. Explanations are deterministic and cached, so pinning the text pins the entire analysis.

---

## 8. Minor observations

- `GET /sources` returns `av` and `ar` HuggingFace repo ids plus a `norm` constant that differs by orders of magnitude between sources (0.0579 gemma, 0.9169 llama, 0.0013 qwen). Do not compare raw `l2_norm` values across sources without accounting for it.
- `tokens[].channel` is present in responses and always `null` on the current three models. Its presence alongside `role` and `section` suggests support for reasoning-model channels is anticipated.
- The OpenAPI document is not served at a stable JSON path. `/api/openapi` returns the HTML app shell. The spec is inlined into the `/api-doc` page and has to be scraped from it.
- `/explain` accepts either `text` or `messages` but not both, and `text` is capped at 16384 characters.
- `completion_tokens` is capped server side at 512.

---

## How these were verified

Raw `curl` against each of the three endpoints, then the same calls through `nla.py`, with responses diffed against the spec's schemas. To reproduce:

```bash
python3 nla_cli.py sources --json
python3 nla_cli.py chat "What is 17 times 23? Answer with just the number." --json
python3 nla_cli.py explain "What is the capital of Canada?" --positions 4,7,9 --json
python3 examples/analyze_thinking.py --compare "Explain why the sky is blue."
```

If you find any of this has changed, please open an issue. It is a snapshot of a live service, not a contract.
