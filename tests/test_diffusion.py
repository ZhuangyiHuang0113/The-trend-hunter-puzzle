from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from trend_hunter.diffusion import calculate_mdl_admission, estimate_topic_diffusion
from trend_hunter.models import Voice


def make_voice(
    identifier: str,
    user: str,
    day: int,
    text: str,
    *,
    quality: float = 0.8,
) -> Voice:
    return Voice(
        id=identifier,
        parent_id=None,
        platform="TapTap" if day % 2 else "Weibo",
        channel=f"channel-{day % 3}",
        user_name=user,
        title=None,
        content=text,
        created_at=datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        + timedelta(days=day - 1),
        normalized=text,
        semantic_text=text,
        quality=quality,
    )


def test_topic_rt_tracks_new_participant_generations():
    rows = [
        make_voice("1", "a", 1, "保底不对"),
        make_voice("2", "b", 2, "概率是不是改了"),
        make_voice("3", "c", 4, "我也觉得概率不对"),
        make_voice("4", "d", 5, "准备查公示概率"),
        make_voice("5", "e", 5, "保底记录有问题"),
        make_voice("6", "f", 6, "想找消协确认"),
    ]
    estimate = estimate_topic_diffusion(list(range(len(rows))), rows, rows[-1].created_at.date())
    assert estimate.organic_rt > 1.0
    assert estimate.confidence > 0.35
    assert estimate.community_breadth > 0.4


def test_copy_campaign_cannot_fake_organic_reproduction():
    rows = [
        make_voice("1", "a", 1, "概率记录不一致"),
        make_voice("2", "b", 2, "保底记录异常"),
        *[make_voice(str(index + 3), f"new-{index}", 5, "统一复制宣传文案") for index in range(10)],
    ]
    estimate = estimate_topic_diffusion(list(range(len(rows))), rows, rows[-1].created_at.date())
    assert estimate.raw_rt > estimate.organic_rt


def test_mdl_accepts_a_coherent_reusable_story():
    rows = [
        make_voice("1", "a", 1, "每天上线像打工"),
        make_voice("2", "b", 2, "日常任务太坐牢"),
        make_voice("3", "c", 3, "这不是游戏是上班"),
        make_voice("4", "d", 4, "监狱服每天打卡"),
        make_voice("5", "e", 1, "角色立绘很好看"),
        make_voice("6", "f", 2, "服务器今天维护"),
        make_voice("7", "g", 3, "竞技场遇到外挂"),
        make_voice("8", "h", 4, "新剧情有反转"),
    ]
    vectors = np.asarray(
        [
            [1.00, 0.02],
            [0.99, 0.05],
            [0.98, -0.03],
            [1.00, -0.04],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [-0.7, 0.7],
        ],
        dtype=np.float32,
    )
    admission = calculate_mdl_admission(
        [0, 1, 2, 3], rows, vectors, defining_terms=["日常", "坐牢", "上班"]
    )
    assert admission.gain_bits > 0
    assert admission.data_bits_saved > admission.model_cost_bits


def test_mdl_deduplicates_copy_volume_before_admission():
    rows = [make_voice(str(index), f"u{index}", index + 1, "同一句宣传") for index in range(5)]
    vectors = np.ones((5, 3), dtype=np.float32)
    admission = calculate_mdl_admission(list(range(5)), rows, vectors, defining_terms=["宣传"])
    assert admission.gain_bits < 0
    assert admission.score == 0.0
