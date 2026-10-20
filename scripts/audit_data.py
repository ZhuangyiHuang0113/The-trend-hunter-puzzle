#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trend_hunter.config import Settings
from trend_hunter.io import discover_calendar, load_calendar, load_voices, parse_as_of
from trend_hunter.preprocess import canonical_for_duplicates, preprocess_voices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="voices.jsonl")
    parser.add_argument("--as-of", default="2026-08-30")
    parser.add_argument("--calendar")
    args = parser.parse_args()
    settings = Settings.from_env(semantic_mode="offline")
    cutoff = parse_as_of(args.as_of, settings.timezone)
    voices = preprocess_voices(load_voices(args.input, cutoff, settings.timezone))
    calendar = args.calendar or discover_calendar(args.input)
    events = load_calendar(calendar, cutoff)
    duplicate_counts = Counter(canonical_for_duplicates(voice.normalized) for voice in voices)
    report = {
        "rows": len(voices),
        "range": [voices[0].created_at.isoformat(), voices[-1].created_at.isoformat()]
        if voices
        else [],
        "platforms": Counter(voice.platform for voice in voices),
        "users": len({voice.user_name for voice in voices}),
        "replies": sum(bool(voice.parent_id) for voice in voices),
        "empty_content": sum(not voice.normalized for voice in voices),
        "low_quality": sum(voice.quality < settings.min_quality for voice in voices),
        "duplicated_rows": sum(count for count in duplicate_counts.values() if count > 1),
        "calendar": [
            {"date": event.date.isoformat(), "version": event.version, "notes": event.notes}
            for event in events
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
