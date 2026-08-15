#!/usr/bin/env python3
"""Command line front end for the Neuronpedia NLA API.

    python3 nla_cli.py sources
    python3 nla_cli.py chat "What is the capital of Canada?"
    python3 nla_cli.py explain "What is the capital of Canada?" --positions 4,7,9
    python3 nla_cli.py explain "What is the capital of Canada?" --all
    python3 nla_cli.py trace "What is the capital of Canada?" --limit 8

`trace` is the interesting one: it generates a reply, then explains the tokens
the model produced, so you see the running commentary behind its own output.

Add --json to any subcommand for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List

from nla import (
    DEFAULT_MODEL,
    NLABadRequest,
    NLAClient,
    NLAError,
    format_explanations,
)


def _client(args: argparse.Namespace) -> NLAClient:
    return NLAClient(api_key=args.api_key)


def _progress(done: int, total: int) -> None:
    if total > 16:
        print("  ... explained {}/{} positions".format(done, total), file=sys.stderr)


def cmd_sources(args: argparse.Namespace) -> int:
    client = _client(args)
    sources = client.sources()
    if args.json:
        print(json.dumps([s.raw for s in sources], indent=2))
        return 0
    print("{:<18} {:<13} {:<6} {:<12} {}".format(
        "MODEL", "SOURCE ID", "LAYER", "OPENROUTER?", "AUTHOR"
    ))
    for s in sources:
        print(
            "{:<18} {:<13} {:<6} {:<12} {}".format(
                s.model_id,
                s.id,
                "L{}".format(s.layer_num) if s.layer_num is not None else "-",
                "yes" if s.open_router_available else "no",
                s.author,
            )
        )
    print("\nOPENROUTER? describes Neuronpedia's route, not whether generation works.")
    print("All currently listed NLA models can generate; other models may be explain-only.")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    client = _client(args)
    out = client.chat(
        args.prompt,
        model_id=args.model,
        nla_source_id=args.source,
        completion_tokens=args.tokens,
        temperature=args.temperature,
        system=args.system,
    )
    if args.json:
        print(json.dumps(out.raw, indent=2))
        return 0
    if out.completion is not None:
        print("--- completion ---")
        print(out.completion)
    print("\n--- tokenised full chat ({} tokens) ---".format(out.prompt_length))
    for t in out.tokens:
        print("{:>4}  {!r}".format(t.position, t.token))
    return 0


def _parse_positions(spec: str) -> List[int]:
    """Accept '4,7,9' and ranges like '10-20', mixed."""
    positions: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise NLABadRequest("invalid --positions value {!r}".format(part))
        if "-" in part:
            lo_text, _, hi_text = part.partition("-")
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise NLABadRequest("position range must be ascending: {!r}".format(part))
            positions.extend(range(lo, hi + 1))
        else:
            positions.append(int(part))
    return positions


def cmd_explain(args: argparse.Namespace) -> int:
    client = _client(args)
    messages = [{"role": "user", "content": args.prompt}]

    if args.all or not args.positions:
        # Need the canonical token list first to know how many positions exist.
        out = client.chat(
            args.prompt,
            model_id=args.model,
            nla_source_id=args.source,
            completion_tokens=1,
            temperature=args.temperature,
        )
        positions = [t.position for t in out.tokens]
        if not args.all:
            print(
                "No --positions given. Prompt tokenises to {} positions; "
                "pass --positions or --all.".format(len(positions)),
                file=sys.stderr,
            )
            for t in out.tokens:
                print("{:>4}  {!r}".format(t.position, t.token), file=sys.stderr)
            return 2
        explanations = client.explain(
            positions,
            text=out.full_text,
            model_id=args.model,
            nla_source_id=out.nla_source_id,
            temperature=args.temperature,
            progress=_progress,
        )
    else:
        explanations = client.explain(
            _parse_positions(args.positions),
            messages=messages,
            model_id=args.model,
            nla_source_id=args.source,
            temperature=args.temperature,
            progress=_progress,
        )

    if args.json:
        print(json.dumps([e.raw for e in explanations], indent=2))
        return 0
    print(format_explanations(explanations, full=args.full))
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    client = _client(args)
    out = client.chat(
        args.prompt,
        model_id=args.model,
        nla_source_id=args.source,
        completion_tokens=args.tokens,
        temperature=args.temperature,
        system=args.system,
    )
    explanations = client.explain_generated(
        out, limit=args.limit, temperature=args.temperature, progress=_progress
    )
    if args.json:
        print(json.dumps({"completion": out.completion, "results": [e.raw for e in explanations]}, indent=2))
        return 0
    if out.completion is not None:
        print("--- model said ---")
        print(out.completion.strip())
        print()
    print("--- what the NLA reads at each generated token ---")
    print(format_explanations(explanations, full=args.full))
    if client.limit_remaining is not None:
        print("(quota remaining on this IP: {})".format(client.limit_remaining), file=sys.stderr)
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser, after_command: bool = False) -> None:
    default = argparse.SUPPRESS if after_command else None
    parser.add_argument(
        "--api-key", default=default, help="overrides NEURONPEDIA_API_KEY and .nla_creds"
    )
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS if after_command else DEFAULT_MODEL,
        help="Neuronpedia model id (default: %(default)s)" if not after_command else "Neuronpedia model id",
    )
    parser.add_argument(
        "--source", default=default, help="NLA source id (default: resolved from --model)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS if after_command else 0.7,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if after_command else False,
        help="raw JSON output",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=argparse.SUPPRESS if after_command else False,
        help="print whole descriptions, not just the first paragraph",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nla_cli.py",
        description="Explore Neuronpedia's NLA API from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_common_arguments(p)

    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sources", help="list available (model, source) pairs")
    _add_common_arguments(s, after_command=True)
    s.set_defaults(func=cmd_sources)

    c = sub.add_parser("chat", help="generate a reply and show its canonical tokenisation")
    _add_common_arguments(c, after_command=True)
    c.add_argument("prompt")
    c.add_argument("--tokens", type=int, default=64, help="max tokens to generate, capped at 512")
    c.add_argument("--system", default=None)
    c.set_defaults(func=cmd_chat)

    e = sub.add_parser("explain", help="explain activations at token positions of a prompt")
    _add_common_arguments(e, after_command=True)
    e.add_argument("prompt")
    e.add_argument("--positions", default=None, help="e.g. 4,7,9 or 10-20 (batched automatically)")
    e.add_argument("--all", action="store_true", help="explain every position (costs one request per 16)")
    e.set_defaults(func=cmd_explain)

    t = sub.add_parser("trace", help="generate a reply, then explain the generated tokens")
    _add_common_arguments(t, after_command=True)
    t.add_argument("prompt")
    t.add_argument("--tokens", type=int, default=48, help="max tokens to generate")
    t.add_argument("--limit", type=int, default=8, help="how many generated tokens to explain")
    t.add_argument("--system", default=None)
    t.set_defaults(func=cmd_trace)

    return p


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NLAError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
