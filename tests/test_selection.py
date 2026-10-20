from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from trend_hunter.config import Settings
from trend_hunter.models import EvidenceFeature, StatusDecision, Voice
from trend_hunter.pipeline import (
    _cross_phase_episode,
    _duplicate_story,
    _merge_duplicate_topics,
)
from trend_hunter.selection import calculate_priority, select_top_candidates
from trend_hunter.signals import calculate_topic_metrics


def _voice(
    number: int,
    *,
    text: str,
    signal_group: str,
    day: int = 24,
) -> Voice:
    return Voice(
        id=str(number),
        parent_id=None,
        platform=("QQ", "Weibo", "TapTap")[number % 3],
        channel=f"channel-{number % 4}",
        user_name=f"player-{number}",
        title=None,
        content=text,
        created_at=datetime(2026, 7, day, number % 23, tzinfo=ZoneInfo("Asia/Hong_Kong")),
        normalized=text,
        semantic_text=text,
        quality=0.9,
        signal_groups=(signal_group,),
    )


def _decision(state: str, reliability: float = 0.9) -> StatusDecision:
    return StatusDecision(
        temporal_state=state,
        authenticity_state="organic",
        official_status=state,
        evidence_scores={
            "new": 0.1,
            "growing": 0.8 if state == "growing" else 0.1,
            "fading": 0.8 if state == "fading" else 0.1,
            "one-off": 0.1,
            "not real": 0.05,
        },
        evidence_reliability=reliability,
        satisfied_conditions=("test evidence",),
        counter_evidence=(),
        uncertainties=(),
        reason="test",
    )


def _priority_metrics(indices: list[int], *, state: str, fast: bool, mdl: float):
    normalized = 0.95 if fast else 0.20
    features = {
        "organic_rt": EvidenceFeature(1.5 if fast else 0.8, normalized, 0.9),
        "cusum_persistence": EvidenceFeature(6 if fast else 1, normalized, 0.9),
        "event_adjusted_persistence": EvidenceFeature(None, None, 0.0, True),
    }
    evidence = SimpleNamespace(features=features)
    return SimpleNamespace(
        candidate_evidence=evidence,
        status_decision=_decision(state),
        message_indices=indices,
        unique_users=len(indices),
        active_days=8,
        community_breadth=0.85,
        duplicate_ratio=0.0,
        top_platform_share=0.40,
        survival_score=0.90,
        mdl_score=mdl,
        last_seen=date(2026, 7, 24),
        priority_components={},
        priority_score=0.0,
        priority_reliability=0.0,
    )


def test_high_impact_slow_issue_can_outrank_a_fast_low_impact_meme():
    voices = [
        _voice(i, text="无法登录，账号数据消失", signal_group="technical") for i in range(12)
    ] + [_voice(i, text="新皮肤这个梗笑死", signal_group="art") for i in range(12, 24)]
    outage = _priority_metrics(list(range(12)), state="fading", fast=False, mdl=0.80)
    meme = _priority_metrics(list(range(12, 24)), state="growing", fast=True, mdl=0.90)
    _, outage_score, _ = calculate_priority(outage, voices, date(2026, 7, 24), Settings())
    _, meme_score, _ = calculate_priority(meme, voices, date(2026, 7, 24), Settings())
    assert outage_score > meme_score


def test_duplicate_decision_uses_semantics_audience_and_action_together():
    organic = _decision("growing")
    left = SimpleNamespace(cluster_id=1, status_decision=organic, top_terms=["登录", "掉线"])
    right = SimpleNamespace(cluster_id=2, status_decision=organic, top_terms=["掉线", "排队"])
    profiles = {
        1: {
            "center": np.array([1.0, 0.0]),
            "users": {"a", "b", "c"},
            "actions": {"technical"},
            "mechanism": "login_and_data_loss",
            "dominance": 0.8,
        },
        2: {
            "center": np.array([0.98, 0.02]),
            "users": {"c", "d", "e"},
            "actions": {"technical"},
            "mechanism": "login_and_data_loss",
            "dominance": 0.8,
        },
    }
    assert _duplicate_story(left, right, profiles)


def test_semantically_close_but_different_product_mechanisms_do_not_merge():
    organic = _decision("growing")
    left = SimpleNamespace(cluster_id=1, status_decision=organic, top_terms=["角色", "美术"])
    right = SimpleNamespace(cluster_id=2, status_decision=organic, top_terms=["角色", "美术"])
    profiles = {
        1: {
            "center": np.array([1.0, 0.0]),
            "users": {"a", "b", "c"},
            "actions": {"art"},
            "mechanism": "overall_art_direction",
            "dominance": 0.8,
        },
        2: {
            "center": np.array([1.0, 0.0]),
            "users": {"a", "b", "d"},
            "actions": {"art"},
            "mechanism": "character_cosmetics",
            "dominance": 0.8,
        },
    }
    assert not _duplicate_story(left, right, profiles)


def test_same_issue_can_merge_across_disjoint_audiences():
    organic = _decision("fading")
    left = SimpleNamespace(cluster_id=1, status_decision=organic, top_terms=["保底", "抽卡"])
    right = SimpleNamespace(cluster_id=2, status_decision=organic, top_terms=["保底", "出金"])
    profiles = {
        1: {
            "center": np.array([1.0, 0.0]),
            "users": {"a", "b"},
            "actions": {"gacha"},
            "mechanism": "gacha_pull_experience",
            "dominance": 0.8,
        },
        2: {
            "center": np.array([0.99, 0.01]),
            "users": {"c", "d"},
            "actions": {"gacha"},
            "mechanism": "gacha_pull_experience",
            "dominance": 0.3,
        },
    }
    assert _duplicate_story(left, right, profiles)


def test_cross_phase_episode_preserves_the_emerging_signal():
    old = SimpleNamespace(cluster_id=4, status_decision=_decision("fading"))
    current = SimpleNamespace(cluster_id=9, status_decision=_decision("growing"))
    assert _cross_phase_episode(old, current) == (4, 9)
    assert _cross_phase_episode(current, old) == (4, 9)

    unresolved = SimpleNamespace(
        cluster_id=12,
        status_decision=SimpleNamespace(official_status=None),
    )
    assert _cross_phase_episode(unresolved, current) == (12, 9)


def test_merged_cluster_recomputes_metrics_from_union(monkeypatch):
    voices = [
        _voice(i, text=f"服务器登录失败 {i}", signal_group="technical", day=20 + i % 4)
        for i in range(8)
    ]
    vectors = np.ones((8, 4), dtype=np.float32) / 2
    traffic = {date(2026, 7, 1) + timedelta(days=offset): 100.0 for offset in range(24)}
    topics = [
        calculate_topic_metrics(
            1, list(range(4)), voices, vectors, date(2026, 7, 24), traffic, [], Settings()
        ),
        calculate_topic_metrics(
            2, list(range(4, 8)), voices, vectors, date(2026, 7, 24), traffic, [], Settings()
        ),
    ]
    monkeypatch.setattr("trend_hunter.pipeline._duplicate_story", lambda *_args: True)
    merged, merged_into = _merge_duplicate_topics(
        topics, voices, vectors, date(2026, 7, 24), [], Settings()
    )
    assert merged_into == {2: 1}
    assert len(merged) == 1
    assert set(merged[0].message_indices) == set(range(8))
    assert merged[0].unique_users == 8
    assert merged[0].merged_from == [1, 2]


def test_titles_and_input_order_do_not_change_priority_selection():
    def candidate(cluster_id: int, title: str):
        metrics = SimpleNamespace(cluster_id=cluster_id, status="growing")
        return SimpleNamespace(
            metrics=metrics,
            keep=True,
            title=title,
            priority_score=0.5,
            priority_reliability=0.8,
        )

    first, _ = select_top_candidates([candidate(2, "A"), candidate(1, "B")], 1)
    second, _ = select_top_candidates([candidate(1, "完全不同的标题"), candidate(2, "C")], 1)
    assert [row.metrics.cluster_id for row in first] == [1]
    assert [row.metrics.cluster_id for row in second] == [1]


def test_top_k_does_not_fill_the_report_with_historical_fading_topics():
    def candidate(cluster_id: int, status: str, score: float):
        return SimpleNamespace(
            metrics=SimpleNamespace(cluster_id=cluster_id, status=status),
            keep=True,
            priority_score=score,
            priority_reliability=0.9,
        )

    rows = [candidate(1, "growing", 0.70)] + [
        candidate(number, "fading", 0.90 - number / 100) for number in range(2, 9)
    ]
    selected, rejected = select_top_candidates(rows, 8)
    assert [row.metrics.status for row in selected].count("fading") == 3
    assert any(reason == "fading_history_lane_full" for reason in rejected.values())
