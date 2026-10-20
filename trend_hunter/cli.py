from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .io import (
    discover_calendar,
    load_calendar,
    load_voices,
    parse_as_of,
    write_audit,
    write_output,
)
from .pipeline import discover_trends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover emerging player trends without hindsight."
    )
    parser.add_argument("--input", required=True, help="Path to voices.jsonl")
    parser.add_argument("--as-of", required=True, help="Inclusive date in YYYY-MM-DD format")
    parser.add_argument("--out", required=True, help="Destination JSON path")
    parser.add_argument("--calendar", help="Optional versions.csv or versions.xlsx")
    parser.add_argument(
        "--audit",
        help="Optional selection-audit JSON path; does not change the official output schema",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=("hybrid", "auto", "openai", "jev", "offline"),
        default=None,
        help=(
            "hybrid (default) clusters locally and reviews only shortlisted candidates; "
            "auto is a backwards-compatible alias"
        ),
    )
    parser.add_argument("--max-trends", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        settings = replace(
            Settings.from_env(semantic_mode=args.semantic_mode),
            max_trends=max(0, min(args.max_trends, 8)),
        )
        if settings.semantic_mode == "openai" and not settings.has_openai_key:
            raise ValueError("--semantic-mode openai requires OPENAI_API_KEY")
        if settings.semantic_mode == "jev" and not settings.has_typesafe_key:
            raise ValueError("--semantic-mode jev requires TYPESAFE_API_KEY")
        cutoff = parse_as_of(args.as_of, settings.timezone)
        input_path = Path(args.input)
        calendar_path = Path(args.calendar) if args.calendar else discover_calendar(input_path)
        voices = load_voices(input_path, cutoff, settings.timezone)
        events = load_calendar(calendar_path, cutoff)
        trends, diagnostics = discover_trends(voices, events, cutoff.date(), settings)
        write_output(args.out, args.as_of, trends)
        if args.audit:
            write_audit(args.audit, args.as_of, diagnostics)
        print(f"ok: wrote {len(trends)} trends to {args.out}", file=sys.stderr)
        if args.audit:
            print(f"ok: wrote selection audit to {args.audit}", file=sys.stderr)
        if args.verbose:
            print(json.dumps(diagnostics, ensure_ascii=False), file=sys.stderr)
        runtime = diagnostics.get("semantic_runtime", {})
        provider = runtime.get("provider", "offline")
        model = runtime.get("model")
        detail = f" ({model})" if model else ""
        print(f"semantic backend: {provider}{detail}", file=sys.stderr)
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Trend discovery failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1
