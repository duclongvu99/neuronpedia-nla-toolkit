# neuronpedia-nla-toolkit

Zero-dependency Python client and CLI for [Neuronpedia's NLA API](https://www.neuronpedia.org/api-doc#tag/nla).

NLA (Natural Language Autoencoder) decodes a model's internal activations into plain English, one token at a time. This client handles the server's batching cap, normalises a response shape that changes between models, and exposes the reconstruction-fidelity metrics the published spec omits.

Measured against the live API. Full findings in [API_NOTES.md](API_NOTES.md).

## Quickstart

Python 3.8+. No install, no dependencies, API key optional.

```bash
git clone https://github.com/duclongvu99/neuronpedia-nla-toolkit
cd neuronpedia-nla-toolkit
python3 nla_cli.py sources
python3 nla_cli.py trace "Why does code review matter?"
```

## Fidelity: read the number before the English

Every `/explain` result carries `cosine_similarity`, `mse`, and `l2_norm`, none of them in the spec. They say how well the autoencoder actually reconstructed the activation it is describing.

| Model | NLA source | Layer | mean cos (EN) | mean cos (VI) |
|---|---|---|---|---|
| `gemma-3-27b-it` | `kitft-l41` | 41 | 0.98 | 0.98 |
| `llama3.3-70b-it` | `kitft-l53` | 53 | 0.88 | 0.92 |
| `qwen2.5-1.5b-it` | `andyxu-l18` | 18 | 0.66 | 0.55 |

On a prompt about buying pens, Qwen at 0.66 reported "the cost of a meal at a restaurant"; Gemma at 0.98 correctly reported "a math problem answer explaining cost calculation". Low-fidelity output is exactly as fluent and confident as high-fidelity output. Only the metric separates them.

Treat `>= 0.95` as informative, `0.90` to `< 0.95` as directional, below `0.90` as unreliable. For Vietnamese, use Gemma: Qwen drops to 0.55 and degenerates into repetition loops.

These three (model, layer) pairs are the entire list. No layer sweeps, and no bringing your own model without training an AV/AR pair.

## Usage

```python
from nla import NLAClient

client = NLAClient()
out = client.chat("A shop sells pens at 7 dollars each. I buy 13. How much?",
                  model_id="llama3.3-70b-it", completion_tokens=32)

for e in client.explain_generated(out, limit=8):
    print(e.position, repr(e.token), round(e.cosine_similarity, 4))
    print("   ", e.description.split("\n")[0])
```

Positions batch automatically past the server's cap of 16 new positions per request:

```python
client.explain_completion(out, positions=range(20, 60))            # 3 requests, merged
client.explain([4, 7, 9], text=saved_text, model_id="gemma-3-27b-it")
```

```bash
python3 nla_cli.py chat    "prompt"
python3 nla_cli.py explain "prompt" --positions 4,7,9      # or 10-25, or --all
python3 nla_cli.py trace   "prompt" --limit 8
python3 examples/analyze_thinking.py --compare "prompt"    # rank all three sources
```

`--json` works on every command; `--full` on `explain` and `trace`; `--model` on `chat`, `explain`, and `trace`.

Non-English works. To check fidelity on your own language, run the comparison with a prompt in it:

```bash
python3 examples/analyze_thinking.py --compare "Vì sao review code lại quan trọng?"
```

## Pros and cons

**Good:** no infrastructure, you read a 70B model's internals from a laptop. No provider API keys. Readable output, unlike SAE feature ids. It ships its own error bars. Explanations are deterministic and cached (2.7s warm against 22.2s cold). Open weights throughout.

**Bad:** three models, one fixed layer each. Fidelity ranges 0.98 to 0.55 and bad output looks identical to good. Descriptions are verbose and templated. Roughly 0.85s per position. Not reasoning models, so no `<think>` trace.

## Gotchas

- **Reproducibility.** For models with an OpenRouter mapping, generation can change between sessions even at `temperature=0.0`. `/explain` is deterministic and cached. So generate once, save `full_text`, and explain from the saved text.
- **The API key does nothing.** Valid key, invalid key, and no key all return 200, and quota follows the IP, not the key.
- **Rate limits are per IP**, each endpoint its own bucket: ~1200/hr `/sources`, 240/hr `/completion`, 120/hr `/explain`. A shared office burns one budget.

## License

MIT. Not affiliated with Neuronpedia or Anthropic. The NLA method and hosted API are theirs; this is an independent client built by testing against the live service.
