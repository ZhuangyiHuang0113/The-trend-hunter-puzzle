from __future__ import annotations

import json

from trend_hunter.io import load_calendar, load_voices, parse_as_of


def test_as_of_is_inclusive_end_of_hong_kong_day(tmp_path):
    source = tmp_path / "voices.jsonl"
    rows = [
        {
            "id": "eligible",
            "parent_id": None,
            "platform": "QQ",
            "channel": "群",
            "user_name": "A",
            "title": None,
            "content": "当天最后一条",
            "created_at": "2026-07-24T23:59:59+08:00",
        },
        {
            "id": "future",
            "parent_id": None,
            "platform": "QQ",
            "channel": "群",
            "user_name": "B",
            "title": None,
            "content": "未来消息",
            "created_at": "2026-07-25T00:00:00+08:00",
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    cutoff = parse_as_of("2026-07-24")
    voices = load_voices(source, cutoff, "Asia/Hong_Kong")
    assert [voice.id for voice in voices] == ["eligible"]
    assert cutoff.utcoffset().total_seconds() == 8 * 3600


def test_utc_row_is_compared_in_hong_kong_time(tmp_path):
    source = tmp_path / "voices.jsonl"
    row = {
        "id": "utc-next-day",
        "parent_id": None,
        "platform": "Discord",
        "channel": "chat",
        "user_name": "A",
        "title": None,
        "content": "timezone boundary",
        "created_at": "2026-07-24T16:30:00+00:00",
    }
    source.write_text(json.dumps(row), encoding="utf-8")
    voices = load_voices(source, parse_as_of("2026-07-24"), "Asia/Hong_Kong")
    assert voices == []


def test_future_calendar_events_are_not_visible(tmp_path):
    source = tmp_path / "versions.csv"
    source.write_text(
        "date,version,notes\n2026-07-21,2.0,new character\n2026-07-31,2.1,maintenance\n",
        encoding="utf-8",
    )
    events = load_calendar(source, parse_as_of("2026-07-24"))
    assert [(event.date.isoformat(), event.version) for event in events] == [("2026-07-21", "2.0")]


def test_input_line_order_does_not_change_loaded_order(tmp_path):
    source = tmp_path / "voices.jsonl"
    rows = [
        {
            "id": voice_id,
            "parent_id": None,
            "platform": "QQ",
            "channel": "群",
            "user_name": voice_id,
            "title": None,
            "content": voice_id,
            "created_at": created_at,
        }
        for voice_id, created_at in (
            ("later", "2026-07-24T12:00:00+08:00"),
            ("early-b", "2026-07-23T12:00:00+08:00"),
            ("early-a", "2026-07-23T12:00:00+08:00"),
        )
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    voices = load_voices(source, parse_as_of("2026-07-24"), "Asia/Hong_Kong")
    assert [voice.id for voice in voices] == ["early-a", "early-b", "later"]
