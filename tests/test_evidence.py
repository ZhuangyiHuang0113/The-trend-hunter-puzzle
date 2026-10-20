from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from trend_hunter.evidence import select_evidence
from trend_hunter.models import Voice


def _voice(
    number: int,
    text: str,
    *,
    user: str,
    platform: str,
    low_reply: bool = False,
) -> Voice:
    return Voice(
        id=str(number),
        parent_id="root" if low_reply else None,
        platform=platform,
        channel=f"channel-{number}",
        user_name=user,
        title=None,
        content=text,
        created_at=datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Hong_Kong")) + timedelta(days=number),
        normalized=text,
        semantic_text=text,
        quality=0.85 if not low_reply else 0.40,
        is_low_context_reply=low_reply,
        signal_groups=("gacha",) if not low_reply else (),
    )


def test_representative_ids_prefer_clear_diverse_non_duplicate_messages():
    voices = [
        _voice(0, "限定池连续吃保底，抽卡体验很差", user="a", platform="QQ"),
        _voice(1, "限定池连续吃保底，抽卡体验很差", user="a", platform="QQ"),
        _voice(2, "武器池又歪了，保底机制让人不敢继续抽", user="b", platform="TapTap"),
        _voice(3, "确实", user="c", platform="Weibo", low_reply=True),
        _voice(4, "连续歪卡后已经不想充值", user="d", platform="Bilibili"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.98, 0.02],
            [1.0, 0.0],
            [0.95, 0.05],
        ],
        dtype=np.float32,
    )
    topic = SimpleNamespace(message_indices=list(range(5)), unique_users=4)
    selected = select_evidence(topic, voices, vectors, limit=3)
    assert 3 not in selected
    assert not ({0, 1} <= set(selected))
    assert len({voices[index].user_name for index in selected}) == len(selected)
    assert len({voices[index].platform for index in selected}) >= 2
    assert voices[selected[0]].content != "确实"
    assert voices[selected[0]].is_low_context_reply is False
