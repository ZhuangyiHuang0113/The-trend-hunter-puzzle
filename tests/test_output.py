from __future__ import annotations

import json

from trend_hunter.io import write_audit, write_output
from trend_hunter.models import Trend


def test_output_is_utf8_and_has_expected_shape(tmp_path):
    trend = Trend(
        title="抽卡概率争议",
        title_en="Gacha rate concerns",
        category="抽卡经济与公平性",
        status="growing",
        first_seen="2026-07-20",
        voice_ids=["a", "b"],
        why="讨论正在扩散。",
        representative_voice={"voice_id": "a", "text": "这次抽卡概率是不是改了？"},
    )
    destination = tmp_path / "trends.json"
    write_output(destination, "2026-07-24", [trend])
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2026-07-24"
    assert payload["trends"][0]["title"] == "抽卡概率争议"
    assert set(payload["trends"][0]) == {
        "title",
        "title_en",
        "category",
        "status",
        "first_seen",
        "voice_ids",
        "representative_voice",
        "why",
    }


def test_audit_is_separate_and_keeps_fixed_debug_slots(tmp_path):
    destination = tmp_path / "selection-audit.json"
    diagnostics = {
        "selected_slots": [7, None, None],
        "selection_audit": [
            {
                "candidate_id": 7,
                "selected": True,
                "priority_score": 0.72,
                "rejection_reason": "",
            }
        ],
    }
    write_audit(destination, "2026-07-24", diagnostics)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2026-07-24"
    assert payload["selected_slots"] == [7, None, None]
    assert payload["candidates"][0]["candidate_id"] == 7


def test_standard_output_includes_representative_original_text(tmp_path):
    trend = Trend(
        title="更新后闪退",
        title_en="Crashes after updates",
        category="性能、闪退与设备负担",
        status="fading",
        first_seen="2026-06-01",
        voice_ids=["best-id", "supporting-id"],
        why="代表原话明确指出触发条件。",
        representative_voice={
            "voice_id": "best-id",
            "text": "自动保存会卡死，更新之后一直闪退。",
        },
    )
    destination = tmp_path / "trends.json"
    write_output(destination, "2026-08-31", [trend])
    row = json.loads(destination.read_text(encoding="utf-8"))["trends"][0]
    assert list(row).index("representative_voice") == list(row).index("voice_ids") + 1
    assert row["representative_voice"] == {
        "voice_id": "best-id",
        "text": "自动保存会卡死，更新之后一直闪退。",
    }


def test_output_rejects_representative_voice_that_is_not_first_voice_id(tmp_path):
    trend = Trend(
        title="更新后闪退",
        title_en="Crashes after updates",
        category="性能、闪退与设备负担",
        status="growing",
        first_seen="2026-06-01",
        voice_ids=["best-id", "supporting-id"],
        why="代表原话明确指出触发条件。",
        representative_voice={"voice_id": "supporting-id", "text": "更新后闪退。"},
    )

    try:
        write_output(tmp_path / "trends.json", "2026-08-31", [trend])
    except ValueError as error:
        assert str(error) == "representative_voice.voice_id must equal voice_ids[0]"
    else:
        raise AssertionError("mismatched representative voice should be rejected")
