from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from trend_hunter.models import Voice
from trend_hunter.preprocess import preprocess_voices
from trend_hunter.threads import attach_thread_context


def voice(identifier: str, content: str, parent_id: str | None = None) -> Voice:
    return Voice(
        id=identifier,
        parent_id=parent_id,
        platform="Bilibili",
        channel="官方号",
        user_name=identifier,
        title=None,
        content=content,
        created_at=datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Hong_Kong")),
    )


def test_short_reply_inherits_parent_meaning_without_replacing_its_id():
    rows = preprocess_voices([voice("p", "这个限定池概率是不是暗改了"), voice("r", "确实", "p")])
    enriched = attach_thread_context(rows)
    reply = enriched[1]
    assert reply.id == "r"
    assert "概率" in reply.semantic_text
    assert "确实" in reply.semantic_text
    assert reply.is_low_context_reply


def test_doge_is_preserved_as_pragmatic_signal():
    row = preprocess_voices([voice("a", "白嫖党表示毫无压力 <emoji:doge>")])[0]
    assert "doge" in row.normalized.lower()
    assert row.quality > 0.3


def test_competitor_comparison_is_kept_as_multi_dimensional_player_signal():
    row = preprocess_voices([voice("a", "鸣潮至少能自选词条，这游戏刷装备真坐牢")])[0]
    assert "鸣潮" in row.competitor_games
    assert "competitor_comparison" in row.pragmatic_flags
    assert {"build", "grind"}.issubset(row.signal_groups)
    assert row.quality > 0.5


def test_short_competitor_comparison_is_not_penalised_as_off_topic():
    row = preprocess_voices([voice("a", "这游戏的动作比绝区零僵硬")])[0]
    assert "绝区零" in row.competitor_games
    assert "competitor_comparison" in row.pragmatic_flags
    assert row.quality >= 0.28


def test_irony_is_a_context_flag_not_a_sentiment_label():
    row = preprocess_voices([voice("a", "策划真是太良心了，80抽保底还歪了 <emoji:doge>")])[0]
    assert {"gacha", "trust"}.issubset(row.signal_groups)
    assert "possible_irony" in row.pragmatic_flags


def test_support_slang_inherits_parent_signal_but_stays_low_context():
    rows = preprocess_voices([voice("p", "80抽保底还歪了"), voice("r", "ZC", "p")])
    reply = attach_thread_context(rows)[1]
    assert reply.is_low_context_reply
    assert "community_slang" in reply.signal_groups
    assert "gacha" in reply.signal_groups
    assert "保底" in reply.semantic_text


def test_tui_you_is_a_retention_signal():
    row = preprocess_voices([voice("a", "再不修这个问题真的准备退游了")])[0]
    assert "retention" in row.signal_groups


def test_piracy_link_request_is_filtered_as_non_product_spam():
    row = preprocess_voices([voice("a", "破解版有吗，求链接<emoji:qq-face-178>")])[0]
    assert row.quality < 0.28
