from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from trend_hunter.diagnostics import calculate_stress_test
from trend_hunter.models import Voice


def make_voice(
    number: int,
    *,
    text: str,
    user: str,
    platform: str,
    day: int,
    parent_id: str | None = None,
) -> Voice:
    return Voice(
        id=f"v{number}",
        parent_id=parent_id,
        platform=platform,
        channel="c",
        user_name=user,
        title=None,
        content=text,
        created_at=datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Hong_Kong")) + timedelta(days=day),
        normalized=text,
        semantic_text=text,
        quality=0.8,
    )


def test_stress_test_distinguishes_broad_support_from_coordinated_copying():
    organic = [
        make_voice(
            number,
            text=f"不同玩家的具体意见 {number}",
            user=f"u{number}",
            platform=f"p{number % 4}",
            day=number % 5,
        )
        for number in range(20)
    ]
    campaign = [
        make_voice(
            number,
            text="完全相同的宣传文案",
            user=f"u{number}",
            platform="p0",
            day=0,
        )
        for number in range(20)
    ]
    metrics = SimpleNamespace(message_indices=list(range(20)), nearby_events=[])
    organic_checks = calculate_stress_test(metrics, organic)
    campaign_checks = calculate_stress_test(metrics, campaign)
    assert organic_checks["largest_platform"] > 0.70
    assert organic_checks["deduplicated"] > 0.95
    assert campaign_checks["largest_platform"] == 0.0
    assert campaign_checks["deduplicated"] < 0.10
