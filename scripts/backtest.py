#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

OFFICIAL_TREND_FIELDS = {
    "title",
    "title_en",
    "category",
    "status",
    "first_seen",
    "voice_ids",
    "why",
}


def _voice_dates(path: Path) -> dict[str, date]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["id"])] = date.fromisoformat(str(row["created_at"])[:10])
    return result


def _validate_output(payload: dict, as_of: str, voice_dates: dict[str, date]) -> None:
    if payload.get("as_of") != as_of or not isinstance(payload.get("trends"), list):
        raise ValueError(f"invalid output envelope for {as_of}")
    cutoff = date.fromisoformat(as_of)
    for trend in payload["trends"]:
        if set(trend) != OFFICIAL_TREND_FIELDS:
            raise ValueError(f"invalid trend schema for {as_of}: {set(trend)}")
        for voice_id in trend["voice_ids"]:
            if voice_id not in voice_dates or voice_dates[voice_id] > cutoff:
                raise ValueError(f"future or unknown voice_id {voice_id} in {as_of}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict rolling-origin snapshots.")
    parser.add_argument("--input", default="voices.jsonl")
    parser.add_argument("--calendar", default="versions.xlsx")
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument(
        "--semantic-mode",
        default="offline",
        choices=("hybrid", "auto", "openai", "jev", "offline"),
    )
    parser.add_argument("--out-dir", default="backtests")
    parser.add_argument(
        "--baseline-files",
        nargs="*",
        default=[],
        help="Optional older output files; comparisons are proxy metrics, not accuracy",
    )
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    voice_dates = _voice_dates(Path(args.input))
    baselines = {}
    for baseline_path in args.baseline_files:
        payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        baselines[payload["as_of"]] = payload
    previous_titles: set[str] = set()
    for as_of in args.dates:
        output = output_dir / f"trends-{as_of}.json"
        audit = output_dir / f"selection-audit-{as_of}.json"
        command = [
            sys.executable,
            "run.py",
            "--input",
            args.input,
            "--as-of",
            as_of,
            "--out",
            str(output),
            "--semantic-mode",
            args.semantic_mode,
            "--audit",
            str(audit),
        ]
        if Path(args.calendar).exists():
            command.extend(["--calendar", args.calendar])
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
        payload = json.loads(output.read_text(encoding="utf-8"))
        _validate_output(payload, as_of, voice_dates)
        audit_payload = json.loads(audit.read_text(encoding="utf-8"))
        selected_audit = [row for row in audit_payload["candidates"] if row["selected"]]
        titles = [trend["title"] for trend in payload["trends"]]
        title_set = set(titles)
        fading_conflicts = sum(
            row["selected_status"] == "fading"
            and row["window_metrics"]["growth_ratio"] > 1.2
            for row in selected_audit
        )
        baseline = baselines.get(as_of)
        baseline_titles = {trend["title"] for trend in baseline["trends"]} if baseline else set()
        summaries.append(
            {
                "as_of": as_of,
                "proxy_metrics": {
                    "selected_count": len(titles),
                    "duplicate_title_count": len(titles) - len(title_set),
                    "fading_growth_conflicts": fading_conflicts,
                    "not_real_selected": sum(
                        trend["status"] == "not real" for trend in payload["trends"]
                    ),
                    "title_stability_from_previous": (
                        len(title_set & previous_titles) / max(len(title_set | previous_titles), 1)
                        if previous_titles
                        else None
                    ),
                    "baseline_title_overlap": (
                        len(title_set & baseline_titles) / max(len(title_set | baseline_titles), 1)
                        if baseline
                        else None
                    ),
                },
                "trends": [
                    {
                        "title": trend["title"],
                        "status": trend["status"],
                        "first_seen": trend["first_seen"],
                    }
                    for trend in payload["trends"]
                ],
            }
        )
        previous_titles = title_set
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
