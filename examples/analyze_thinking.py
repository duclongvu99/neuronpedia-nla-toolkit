#!/usr/bin/env python3
"""Read an open model's per-token 'thinking' through the NLA API.

    python3 examples/analyze_thinking.py --model llama3.3-70b-it \
        "A shop sells pens at 7 dollars each. I buy 13 pens. How much do I pay?"

    # compare how faithful each of the three NLAs is on the same prompt
    python3 examples/analyze_thinking.py --compare "Explain why the sky is blue."

The recipe, and why each step is what it is:

  1. Generate via POST /completion, NOT with your own copy of the model.
     The endpoint returns the canonical NLA tokenisation, so the `position`
     values it gives you are exactly the ones /explain accepts. Generate
     anywhere else and your positions silently point at the wrong tokens.

  2. Keep only role=assistant AND section=content tokens. That drops the chat
     scaffold (`<start_of_turn>`, `model`, `<end_of_turn>`), which also carries
     role=assistant but is not the model's answer.

  3. Explain those positions, then READ cosine_similarity before you read the
     English. The NLA emits fluent, confident prose at every fidelity level.
     Measured on one shared prompt: gemma L41 ~0.98, llama L53 ~0.88,
     qwen L18 ~0.66. At 0.66 the descriptions are confabulated: on a prompt
     about buying pens, the qwen NLA reported "the cost of a meal at a
     restaurant". The English gives you no hint that this happened. The metric
     does.

  4. For anything you intend to publish, PIN THE TEXT. Generation is not
     reproducible across sessions: /completion routes through OpenRouter, and
     which backend provider serves you can change between runs. Measured on
     llama3.3-70b-it at temperature=0.0, identical calls returned
     "## Step 1: Determine the cost of one pen" in one window and
     "To find out how much you pay, you need to multiply..." in another, while
     four consecutive calls inside a single window were byte-identical.
     So: generate ONCE, save `full_text` to disk, and from then on call
     /explain with text=<saved full_text> instead of regenerating. Explanations
     are cached and deterministic for a fixed (text, model, source,
     temperature), so pinning the text makes the whole analysis reproducible.

Hard limits worth knowing before you plan around this:
  * Only three (model, layer) pairs exist. You cannot point NLA at an arbitrary
    open model, nor at another size of the same family, nor at your own
    fine-tune, without training an AV/AR pair for it.
  * One fixed layer per model. No layer sweeps.
  * Rate limits are per IP, not per key: 120 explain/hr, 240 completion/hr.
    Everyone behind one office IP shares that budget.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nla import KNOWN_SOURCES, NLAClient, NLAError  # noqa: E402

# Below this, treat the description as unreliable rather than informative.
FIDELITY_FLOOR = 0.90


def analyze(client: NLAClient, prompt: str, model: str, n_tokens: int, temperature: float, quiet: bool = False):
    """Generate with `model`, then explain the tokens it produced."""
    out = client.chat(
        prompt,
        model_id=model,
        nla_source_id=KNOWN_SOURCES.get(model),
        completion_tokens=n_tokens,
        temperature=temperature,
    )
    answer = "".join(t.token for t in out.tokens if t.role == "assistant" and t.section == "content")

    if not quiet:
        print("model said: {!r}".format(answer[:300]))
        print()

    explanations = client.explain_generated(out, limit=n_tokens, temperature=temperature)
    if not explanations:
        print("no assistant-content tokens found; nothing to explain", file=sys.stderr)
        return out, []

    if not quiet:
        for e in explanations:
            cos = e.cosine_similarity
            flag = ""
            if cos is not None and cos < FIDELITY_FLOOR:
                flag = "   <-- LOW FIDELITY, do not trust this description"
            print("pos {:<4} {!r:<16} cos={}{}".format(
                e.position, e.token, "{:.4f}".format(cos) if cos is not None else "n/a", flag))
            print("     " + e.description.strip().split("\n")[0][:160])
        print()

    return out, explanations


def summarise(explanations):
    scores = [e.cosine_similarity for e in explanations if e.cosine_similarity is not None]
    if not scores:
        return None
    weak = sum(1 for s in scores if s < FIDELITY_FLOOR)
    return {
        "n": len(scores),
        "mean": statistics.mean(scores),
        "min": min(scores),
        "max": max(scores),
        "weak": weak,
    }


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt")
    p.add_argument("--model", default="llama3.3-70b-it", choices=sorted(KNOWN_SOURCES),
                   help="default %(default)s: the best-fidelity genuinely-open model")
    p.add_argument("--tokens", type=int, default=8, help="generated tokens to explain (each 16 costs 1 request)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--compare", action="store_true", help="run all three NLAs on this prompt and rank fidelity")
    args = p.parse_args(argv)

    client = NLAClient()
    try:
        if not args.compare:
            _, ex = analyze(client, args.prompt, args.model, args.tokens, args.temperature)
            s = summarise(ex)
            if s:
                print("fidelity: mean cos {:.4f} over {} tokens; {} below {}".format(
                    s["mean"], s["n"], s["weak"], FIDELITY_FLOOR))
            return 0

        rows = []
        for model in ["gemma-3-27b-it", "llama3.3-70b-it", "qwen2.5-1.5b-it"]:
            print("=" * 78)
            print("MODEL:", model, " NLA:", KNOWN_SOURCES[model])
            _, ex = analyze(client, args.prompt, model, args.tokens, args.temperature)
            s = summarise(ex)
            if s:
                rows.append((model, s))
        print("=" * 78)
        print("{:<20} {:>9} {:>9} {:>9} {:>14}".format("MODEL", "mean cos", "min", "max", "below floor"))
        for model, s in sorted(rows, key=lambda r: -r[1]["mean"]):
            print("{:<20} {:>9.4f} {:>9.4f} {:>9.4f} {:>10}/{}".format(
                model, s["mean"], s["min"], s["max"], s["weak"], s["n"]))
        print("\nHigher mean cos = the description is actually describing this activation.")
        print("Low-fidelity rows still produce confident, fluent English. That is the trap.")
        return 0
    except NLAError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
