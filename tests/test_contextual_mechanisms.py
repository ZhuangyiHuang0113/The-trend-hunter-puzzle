from datetime import datetime
from zoneinfo import ZoneInfo

from trend_hunter.models import Voice
from trend_hunter.selection import voice_mechanisms


def _voice(text: str, *, flags: tuple[str, ...] = ()) -> Voice:
    return Voice(
        id="v",
        parent_id=None,
        platform="QQ",
        channel="c",
        user_name="u",
        title=None,
        content=text,
        created_at=datetime(2026, 8, 1, tzinfo=ZoneInfo("Asia/Hong_Kong")),
        normalized=text,
        semantic_text=text,
        pragmatic_flags=flags,
    )


def test_context_separates_login_failures_from_post_update_crashes():
    login = voice_mechanisms(_voice("服务器登录一直转圈，排队半天还是进不去"))
    crash = voice_mechanisms(_voice("更新后自动保存就会卡死，之后一直闪退"))
    assert "login_and_data_loss" in login
    assert "performance_and_crashes" not in login
    assert "performance_and_crashes" in crash
    assert "login_and_data_loss" not in crash


def test_context_recognises_actionable_player_mechanisms():
    assert "matchmaking" in voice_mechanisms(_voice("匹配等了五分钟，低分段总遇到高段位大佬"))
    assert "gacha_pull_frustration" in voice_mechanisms(
        _voice("这个卡池连续歪又吃保底，已经不想充了")
    )
    assert "resource_farming" in voice_mechanisms(_voice("装备副词条太随机，刷本一个月都毕业不了"))


def test_explicit_competitor_comparison_is_a_product_mechanism():
    mechanisms = voice_mechanisms(
        _voice("隔壁新版本比这个有意思", flags=("competitor_comparison",))
    )
    assert "competitor_content_gap" in mechanisms
