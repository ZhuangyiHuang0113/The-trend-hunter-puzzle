from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, timedelta

from .models import TopicMetrics, Voice
from .preprocess import canonical_for_duplicates


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _message_contributions(metrics: TopicMetrics, voices: list[Voice]) -> list[tuple[int, float]]:
    indices = sorted(metrics.message_indices, key=lambda index: voices[index].created_at)
    exact_counts = Counter(canonical_for_duplicates(voices[index].normalized) for index in indices)
    user_day_seen: Counter[tuple[str, date]] = Counter()
    contributions = []
    for index in indices:
        voice = voices[index]
        duplicate_key = canonical_for_duplicates(voice.normalized)
        user_day_key = (voice.user_name, voice.created_at.date())
        user_day_seen[user_day_key] += 1
        duplicate_discount = 1.0 / math.sqrt(max(exact_counts[duplicate_key], 1))
        user_discount = 1.0 / math.sqrt(user_day_seen[user_day_key])
        reply_discount = 0.55 if voice.is_low_context_reply else 1.0
        contributions.append(
            (index, voice.quality * duplicate_discount * user_discount * reply_discount)
        )
    return contributions


def calculate_stress_test(metrics: TopicMetrics, voices: list[Voice]) -> dict[str, float]:
    """Remove each source of dependence and measure how much signal remains."""
    rows = _message_contributions(metrics, voices)
    base = sum(weight for _, weight in rows)
    if base <= 0:
        return {
            "largest_user": 0.0,
            "largest_platform": 0.0,
            "peak_day": 0.0,
            "deduplicated": 0.0,
            "roots_only": 0.0,
            "event_window": 0.0,
        }

    def retention(keep) -> float:
        return _clip(sum(weight for index, weight in rows if keep(index)) / base)

    by_user: defaultdict[str, float] = defaultdict(float)
    by_platform: defaultdict[str, float] = defaultdict(float)
    by_day: defaultdict[date, float] = defaultdict(float)
    for index, weight in rows:
        voice = voices[index]
        by_user[voice.user_name] += weight
        by_platform[voice.platform] += weight
        by_day[voice.created_at.date()] += weight
    largest_user = max(by_user, key=by_user.get)
    largest_platform = max(by_platform, key=by_platform.get)
    peak_day = max(by_day, key=by_day.get)

    seen_text: set[str] = set()
    deduplicated: set[int] = set()
    for index, _ in rows:
        key = canonical_for_duplicates(voices[index].normalized)
        if key not in seen_text:
            seen_text.add(key)
            deduplicated.add(index)

    event_days = {
        event.date + timedelta(days=offset)
        for event in metrics.nearby_events
        for offset in (-1, 0, 1)
    }
    return {
        "largest_user": retention(lambda index: voices[index].user_name != largest_user),
        "largest_platform": retention(lambda index: voices[index].platform != largest_platform),
        "peak_day": retention(lambda index: voices[index].created_at.date() != peak_day),
        "deduplicated": retention(lambda index: index in deduplicated),
        "roots_only": retention(lambda index: voices[index].parent_id is None),
        "event_window": (
            retention(lambda index: voices[index].created_at.date() not in event_days)
            if event_days
            else 1.0
        ),
    }


def _survival_score(checks: dict[str, float]) -> float:
    core = [
        checks["largest_user"],
        checks["largest_platform"],
        checks["peak_day"],
        checks["deduplicated"],
    ]
    context = [checks["roots_only"], checks["event_window"]]
    core_mean = sum(core) / len(core)
    context_mean = sum(context) / len(context)
    return _clip(0.65 * core_mean + 0.25 * min(core) + 0.10 * context_mean)


def enrich_topic_diagnostics(topics: list[TopicMetrics], voices: list[Voice]) -> None:
    for metrics in topics:
        checks = calculate_stress_test(metrics, voices)
        metrics.stress_tests = checks
        metrics.survival_score = _survival_score(checks)
