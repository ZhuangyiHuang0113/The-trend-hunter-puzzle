from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from trend_hunter.config import Settings
from trend_hunter.decision import decide_status
from trend_hunter.diagnostics import enrich_topic_diagnostics
from trend_hunter.models import CalendarEvent, Voice
from trend_hunter.signals import _slow_shift_cusum, calculate_topic_metrics


def make_voices(daily_counts: list[int], duplicate: bool = False) -> list[Voice]:
    result = []
    start = date(2026, 7, 1)
    number = 0
    for offset, count in enumerate(daily_counts):
        for item in range(count):
            number += 1
            result.append(
                Voice(
                    id=str(number),
                    parent_id=None,
                    platform=["QQ", "Weibo", "TapTap"][item % 3],
                    channel="c",
                    user_name=f"u{number}" if not duplicate else f"u{item % 20}",
                    title=None,
                    content="同一句宣传" if duplicate else f"具体玩家意见 {offset} {item}",
                    normalized="同一句宣传" if duplicate else f"具体玩家意见 {offset} {item}",
                    semantic_text="同一句宣传" if duplicate else f"具体玩家意见 {offset} {item}",
                    quality=0.8,
                    created_at=datetime.combine(
                        start + timedelta(days=offset),
                        datetime.min.time(),
                        tzinfo=ZoneInfo("Asia/Hong_Kong"),
                    )
                    + timedelta(minutes=item),
                )
            )
    return result


def metrics(rows: list[Voice], as_of: date, events=None):
    vectors = np.ones((len(rows), 4), dtype=np.float32) / 2
    traffic = {start + timedelta(days=i): 100.0 for i in range(40) for start in [date(2026, 7, 1)]}
    result = calculate_topic_metrics(
        1,
        list(range(len(rows))),
        rows,
        vectors,
        as_of,
        traffic,
        events or [],
        Settings(),
    )
    enrich_topic_diagnostics([result], rows)
    # Status tests isolate temporal/authenticity decisions from MDL's separate admission role.
    result.mdl_gain_bits = 20.0
    result.mdl_score = 0.9
    decide_status(result, as_of, Settings())
    return result


def test_high_rt_without_persistence_is_not_automatically_growing():
    rows = make_voices([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
    result = metrics(rows, date(2026, 7, 10))
    result.cusum_score = 0.0
    result.cusum_exceed_days = 0
    result.cusum_alarm = False
    result.growth_ratio = 1.0
    decide_status(result, date(2026, 7, 10), Settings())
    assert result.status_decision is not None
    assert result.status_decision.official_status != "growing"
    assert result.authenticity > 0.5
    assert result.topic_rt > 1.0


def test_rt_and_persistent_cusum_jointly_support_growing():
    rows = make_voices([1] * 21 + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    result = metrics(rows, date(2026, 7, 31))
    assert result.status == "growing"
    assert result.topic_rt > 1.0
    assert result.cusum_exceed_days >= 3


def test_more_posts_from_the_same_people_is_not_self_propagation():
    rows = make_voices([3, 3, 3, 8, 8, 8])
    repeated_users = ["a", "b", "c"]
    rows = [
        Voice(
            id=row.id,
            parent_id=row.parent_id,
            platform=row.platform,
            channel=row.channel,
            user_name=repeated_users[index % len(repeated_users)],
            title=row.title,
            content=row.content,
            created_at=row.created_at,
            normalized=row.normalized,
            semantic_text=row.semantic_text,
            quality=row.quality,
        )
        for index, row in enumerate(rows)
    ]
    result = metrics(rows, date(2026, 7, 6))
    assert result.topic_rt < 1.0
    assert result.status != "growing"


def test_synchronized_duplicate_campaign_is_not_real():
    rows = make_voices([0, 0, 0, 20], duplicate=True)
    moment = rows[0].created_at
    rows = [replace(row, created_at=moment, platform="Weibo") for row in rows]
    result = metrics(rows, date(2026, 7, 4))
    assert result.status == "not real"


def test_natural_multi_player_repetition_is_not_enough_for_not_real():
    rows = make_voices([2, 2, 2, 2, 2, 2], duplicate=True)
    rows = [replace(row, user_name=f"player-{index}") for index, row in enumerate(rows)]
    result = metrics(rows, date(2026, 7, 6))
    assert result.status_decision is not None
    assert result.status_decision.official_status != "not real"


def test_calendar_aligned_short_spike_is_one_off():
    rows = make_voices([1, 1, 15])
    event = CalendarEvent(date(2026, 7, 3), version="2.1", notes="maintenance")
    result = metrics(rows, date(2026, 7, 6), [event])
    assert result.status == "one-off"
    assert result.event_persistence < 0.45


def test_event_day_is_too_early_to_call_one_off():
    rows = make_voices([1, 1, 15])
    event = CalendarEvent(date(2026, 7, 3), version="2.1", notes="maintenance")
    result = metrics(rows, date(2026, 7, 3), [event])
    assert result.status != "one-off"
    assert result.event_persistence == 1.0


def test_event_triggered_specific_story_can_be_new_even_when_parent_topic_is_old():
    rows = make_voices([2] * 20 + [3, 4, 5])
    event = CalendarEvent(date(2026, 7, 21), version="2.0", notes="anniversary event")
    result = metrics(rows, date(2026, 7, 23), [event])
    result.top_terms = ["活动", "奖励"]
    result.mdl_gain_bits = 20.0
    result.mdl_score = 0.9
    result.event_persistence = 0.9
    result.nearby_events = [event]
    result.post_peak_observed_days = 2
    result.recent_volume = 1.2
    result.growth_ratio = 1.4
    result.topic_rt = 0.95
    result.rt_confidence = 0.0
    result.cusum_score = 0.0
    result.cusum_exceed_days = 0
    result.cusum_alarm = False
    decision = decide_status(result, date(2026, 7, 23), Settings())
    assert decision.official_status == "new"
    assert decision.evidence_reliability >= Settings().min_status_reliability


def test_high_cusum_from_one_account_has_low_reliability():
    rows = make_voices([1] * 21 + [3] * 10)
    rows = [replace(row, user_name="single-account") for row in rows]
    result = metrics(rows, date(2026, 7, 31))
    assert result.cusum_exceed_days >= 3
    assert result.status_decision is not None
    assert result.status_decision.official_status != "growing"
    assert result.status_decision.evidence_reliability < Settings().min_status_reliability


def test_fading_cannot_survive_a_rising_window_conflict():
    rows = make_voices([8, 8, 8, 7, 6, 5, 4, 3, 2, 1])
    result = metrics(rows, date(2026, 7, 14))
    result.topic_rt = 0.35
    result.rt_confidence = 0.95
    result.recent_volume = 0.20
    result.previous_volume = 1.20
    result.baseline_volume = 1.00
    result.growth_ratio = 1.30
    decision = decide_status(result, date(2026, 7, 14), Settings())
    assert decision.official_status is None
    assert "窗口声量上涨与衰退判断冲突" in decision.uncertainties


def test_new_is_not_the_default_for_an_old_ambiguous_topic():
    rows = make_voices([2] * 18)
    result = metrics(rows, date(2026, 7, 18))
    assert result.status_decision is not None
    assert result.status_decision.official_status != "new"


def test_status_is_stable_under_small_weight_changes():
    rows = make_voices([1] * 21 + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    result = metrics(rows, date(2026, 7, 31))
    settings = replace(
        Settings(),
        growing_weights=(0.44, 0.32, 0.14, 0.10),
    )
    decision = decide_status(result, date(2026, 7, 31), settings)
    assert decision.official_status == "growing"


def test_mdl_rt_and_cusum_ablation_have_separate_roles():
    rows = make_voices([1] * 21 + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    no_mdl = metrics(rows, date(2026, 7, 31))
    no_mdl.mdl_gain_bits = -1.0
    no_mdl.mdl_score = 0.0
    assert decide_status(no_mdl, date(2026, 7, 31), Settings()).official_status == "growing"

    no_rt = metrics(rows, date(2026, 7, 31))
    no_rt.topic_rt = 0.95
    no_rt.rt_confidence = 0.0
    assert decide_status(no_rt, date(2026, 7, 31), Settings()).official_status != "growing"

    no_cusum = metrics(rows, date(2026, 7, 31))
    no_cusum.cusum_score = 0.0
    no_cusum.cusum_exceed_days = 0
    no_cusum.cusum_alarm = False
    no_cusum.growth_ratio = 1.0
    assert decide_status(no_cusum, date(2026, 7, 31), Settings()).official_status != "growing"


def test_cusum_detects_a_small_sustained_shift():
    start = date(2026, 7, 1)
    baseline = [5.0, 5.2, 4.8] * 7
    monitor = [5.6, 5.8, 6.0, 6.2, 6.4, 6.5, 6.7, 6.8, 7.0, 7.2]
    rates = baseline + monitor
    daily = {start + timedelta(days=index): value for index, value in enumerate(rates)}
    traffic = {start + timedelta(days=index): 1000.0 for index in range(len(rates))}
    _, _, score, exceed_days, alarm = _slow_shift_cusum(
        daily, traffic, start + timedelta(days=len(rates) - 1)
    )
    assert score > 1.0
    assert exceed_days >= 3
    assert alarm


def test_cusum_does_not_turn_one_spike_into_a_slow_grievance():
    start = date(2026, 7, 1)
    rates = [5.0, 5.2, 4.8] * 7 + [5.0, 5.0, 5.0, 12.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    daily = {start + timedelta(days=index): value for index, value in enumerate(rates)}
    traffic = {start + timedelta(days=index): 1000.0 for index in range(len(rates))}
    _, _, _, exceed_days, alarm = _slow_shift_cusum(
        daily, traffic, start + timedelta(days=len(rates) - 1)
    )
    assert exceed_days == 1
    assert not alarm


def test_warmup_cusum_uses_available_left_edge_history():
    start = date(2026, 6, 1)
    rates = [2.0, 2.1, 1.9, 3.0, 3.5, 4.0, 4.5, 5.0]
    daily = {start + timedelta(days=index): value for index, value in enumerate(rates)}
    traffic = {start + timedelta(days=index): 1000.0 for index in range(len(rates))}

    _, _, score, exceed_days, alarm = _slow_shift_cusum(
        daily, traffic, start + timedelta(days=len(rates) - 1)
    )

    assert score > 1.0
    assert exceed_days >= 3
    assert alarm


def test_day_eight_rising_topic_transitions_from_new_to_growing():
    rows = make_voices([2, 2, 2, 2, 3, 4, 5, 7])
    result = metrics(rows, date(2026, 7, 8))

    assert result.status_decision is not None
    assert result.status_decision.official_status == "growing"
