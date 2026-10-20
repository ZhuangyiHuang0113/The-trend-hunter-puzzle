from datetime import UTC, datetime

from trend_hunter.models import Voice
from trend_hunter.semantic import _local_title, _studio_interpretation


def _voice(platform: str, text: str) -> Voice:
    return Voice(
        id=platform,
        parent_id=None,
        platform=platform,
        channel="test",
        user_name=platform,
        title=None,
        content=text,
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        normalized=text,
        semantic_text=text,
    )


def test_technical_titles_follow_dominant_mechanism_not_incidental_terms():
    assert (
        _local_title(
            "性能、闪退与设备负担",
            [],
            [],
            ["登录", "闪退", "卡死"],
            "fading",
            "performance_and_crashes",
        )
        == "更新后闪退、卡死与设备发热影响稳定游玩"
    )

    assert (
        _local_title(
            "登录可用性与账号数据",
            [],
            [],
            ["闪退", "登录", "排队"],
            "fading",
            "login_and_data_loss",
        )
        == "登录转圈、排队与掉线正在阻断正常游玩"
    )


def test_gacha_title_follows_dominant_mechanism_not_incidental_paywall_terms():
    assert (
        _local_title(
            "保底挫败与付费意愿",
            [],
            [],
            ["核心材料", "通行证", "吃保底", "歪了"],
            "fading",
            "gacha_pull_frustration",
        )
        == "连续歪卡与吃保底正在削弱付费意愿"
    )


def test_unmatched_story_uses_cluster_evidence_instead_of_generic_filler():
    meaning, action = _studio_interpretation(
        "玩家体验与产品信任",
        "新地图空气墙挡住隐藏宝箱",
        "新地图空气墙阻断隐藏宝箱探索",
        top_terms=["空气墙", "隐藏宝箱", "新地图"],
        evidence=[_voice("QQ", "空气墙挡住隐藏宝箱"), _voice("TapTap", "新地图过不去")],
    )

    assert "这些评论已经形成同一个可调查的玩家故事" not in meaning
    assert "空气墙" in meaning
    assert "QQ" in meaning and "TapTap" in meaning
    assert "空气墙" in action
    assert "隐藏宝箱" in action


def test_login_story_gets_server_side_action_not_device_diagnostics():
    meaning, action = _studio_interpretation(
        "登录可用性与账号数据",
        "维护后登录转圈 排队12000人 服务器掉线",
        "维护后登录排队与掉线",
    )

    assert "服务可用性" in meaning
    assert "鉴权网关" in action
    assert "设备" not in action
