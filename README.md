# neuronpedia-nla-toolkit

Zero-dependency Python client and CLI for [Neuronpedia's NLA API](https://www.neuronpedia.org/api-doc#tag/nla).

NLA (Natural Language Autoencoder) decodes a model's internal activations into plain English, one token position at a time. Neuronpedia serves it over three HTTP endpoints. This client handles the batching rule, normalises a response shape that changes between models, and exposes the reconstruction-fidelity metrics that the published spec omits but that you need in order to know when to believe the output.

Measured against the live API, not inferred from docs. Discrepancies are recorded in [API_NOTES.md](API_NOTES.md).

## Quickstart

Python 3.8+. No install, no dependencies.

```bash
git clone https://github.com/duclongvu99/neuronpedia-nla-toolkit
cd neuronpedia-nla-toolkit

python3 nla_cli.py sources                                    # list NLA sources
python3 nla_cli.py trace "Why does code review matter?"       # generate, then explain its own tokens
```

An API key is optional (see [below](#api-key)). To use one: `export NEURONPEDIA_API_KEY=sk-np-...` or `echo 'sk-np-...' > ~/.nla_creds`.

### Tiếng Việt

```bash
python3 nla_cli.py sources
python3 nla_cli.py trace "Vì sao review code lại quan trọng?"
```

Ba điều cần nhớ:

1. **Luôn đọc `cosine_similarity` trước khi tin phần mô tả tiếng Anh.** NLA luôn trả về câu văn trôi chảy, kể cả khi tái tạo sai.
2. **Chỉ có đúng 3 model được hỗ trợ.** Không dùng được cho model bất kỳ.
3. **Sinh văn bản một lần rồi lưu `full_text`.** Việc sinh không tái lập được giữa các phiên; việc giải thích thì có.

## Sources

`GET /api/nla/sources` is authoritative. As of 2026-08-15:

| Model | NLA source | Layer | Author |
|---|---|---|---|
| `gemma-3-27b-it` | `kitft-l41` | 41 | Anthropic (kitft) |
| `llama3.3-70b-it` | `kitft-l53` | 53 | Anthropic (kitft) |
| `qwen2.5-1.5b-it` | `andyxu-l18` | 18 | Andy Xu |

All three are open-weight, and the NLA weights are public on HuggingFace. One NLA is bound to one model at one layer: no layer sweeps, and no bringing your own model without training an AV/AR pair.

## Fidelity: read this number first

Every `/explain` result carries `cosine_similarity`, `mse`, and `l2_norm`, describing how well the autoencoder reconstructed the activation it is describing. None appear in the published spec.

Same prompt, first six generated tokens, temperature 0:

| Model | mean cos |
|---|---|
| `gemma-3-27b-it` | 0.9795 |
| `llama3.3-70b-it` | 0.8815 |
| `qwen2.5-1.5b-it` | 0.6610 |

The prompt was about **buying pens**. Qwen at 0.66 reported "the cost of **a trip**", "the cost of **a meal at a restaurant**", "the cost of **a service**". Gemma at 0.98 correctly reported "a math problem answer explaining cost calculation... for a shopping scenario".

Low-fidelity output is exactly as fluent and confident as high-fidelity output. Nothing in the English signals confabulation; only the metric does. Treat `>= 0.95` as informative, `0.90-0.95` as directional, below `0.90` as unreliable.

## Usage

```python
from nla import NLAClient

client = NLAClient()

# Generate through the API so token positions line up with /explain.
out = client.chat("A shop sells pens at 7 dollars each. I buy 13. How much?",
                  model_id="llama3.3-70b-it", completion_tokens=32)

for e in client.explain_generated(out, limit=8):
    trust = "ok" if (e.cosine_similarity or 0) >= 0.95 else "LOW"
    print(f"{e.position:>4} {e.token!r:<14} cos={e.cosine_similarity:.4f} [{trust}]")
    print("     ", e.description.split("\n")[0])
```

Any number of positions is safe. The server caps 16 *new* positions per request; the client splits transparently.

```python
explanations = client.explain_completion(out, positions=range(20, 60))   # 3 requests, merged
explanations = client.explain([4, 7, 9], text=saved_full_text, model_id="gemma-3-27b-it")
```

CLI:

```bash
python3 nla_cli.py sources
python3 nla_cli.py chat    "prompt"
python3 nla_cli.py explain "prompt" --positions 4,7,9     # or 10-25, or --all
python3 nla_cli.py trace   "prompt" --limit 8
python3 examples/analyze_thinking.py --compare "prompt"   # rank all three NLAs
```

Add `--json` for machine-readable output, `--full` for complete descriptions, `--model` to switch models.

## Pros and cons

**Good:** no infrastructure, you read a 70B model's internals from a laptop. No provider API keys at all. Output is readable immediately, unlike SAE feature ids. It ships its own error bars. Explanations are deterministic and cached (26/26 byte-identical on repeat, 2.7s warm against 22.2s cold). Open weights all the way down.

**Bad:** three models and that is the whole list. One fixed layer each. Fidelity ranges 0.98 to 0.66 and the bad output looks identical to the good. Descriptions are verbose and templated. Generation is not reproducible across sessions. Rate limits are per IP, not per key, so a shared office burns one budget. Roughly 0.85s per position, so long transcripts are slow. These are not reasoning models, so there is no `<think>` trace.

Good for reading how internal state evolves across a short passage, for teaching, and for building intuition without a GPU. Weak for layer-wise analysis, unsupported models, or high-volume batch work.

## Reproducibility

For models with an OpenRouter mapping, `/completion` routes through OpenRouter and the backend provider can change between sessions. At `temperature=0.0`, four consecutive calls were byte-identical, but the same call in a later window returned different text. `/explain` is deterministic and cached.

**Generate once, save `full_text`, then explain from the saved text.** That makes the analysis reproducible; regenerating does not.

## API key

Not required. Valid key, invalid key, and no key header all return 200, and `x-limit-remaining` decrements identically, so quota follows the IP rather than the key. Set one anyway if you have an account, since this is not contractual.

Limits, each its own bucket: ~1200/hr `/sources`, 240/hr `/completion`, 120/hr `/explain`.

## License

MIT. Not affiliated with Neuronpedia or Anthropic. The NLA method and hosted API are theirs; this is an independent client built by testing against the live service.
