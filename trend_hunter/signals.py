from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

from .config import Settings
from .diffusion import estimate_topic_diffusion
from .models import CalendarEvent, TopicMetrics, Voice
from .preprocess import canonical_for_duplicates


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _date_range(end: date, days: int, offset: int = 0) -> list[date]:
    return [end - timedelta(days=offset + step) for step in range(days)]


def _window_rate(daily: dict[date, float], traffic: dict[date, float], dates: list[date]) -> float:
    # Per-thousand informative messages stops launch-day traffic becoming a trend by volume alone.
    return _mean([1000.0 * daily.get(day, 0.0) / max(traffic.get(day, 1.0), 1.0) for day in dates])


def _event_residual(
    daily: dict[date, float],
    traffic: dict[date, float],
    events: list[CalendarEvent],
    first_seen: date,
    peak_day: date,
    as_of: date,
) -> tuple[list[CalendarEvent], float, float, float, float, float, int]:
    """Compare an event spike with its own pre-event baseline and post-event tail."""

    relevant = [
        event
        for event in events
        if event.date <= as_of
        and (abs((event.date - peak_day).days) <= 1 or abs((event.date - first_seen).days) <= 1)
    ]
    if not relevant:
        return [], 0.0, 0.0, 0.0, 0.0, 1.0, 0

    def daily_rate(day: date) -> float:
        return 1000.0 * daily.get(day, 0.0) / max(traffic.get(day, 1.0), 1.0)

    best: tuple[float, CalendarEvent, float, float, float, float, float] | None = None
    for event in relevant:
        control_days = [event.date - timedelta(days=step) for step in range(2, 8)]
        control_rates = [daily_rate(day) for day in control_days if day in traffic]
        expected = statistics.median(control_rates) if control_rates else 0.0

        event_days = [
            event.date + timedelta(days=offset)
            for offset in (-1, 0, 1)
            if event.date + timedelta(days=offset) <= as_of
            and event.date + timedelta(days=offset) in traffic
        ]
        observed = _mean([daily_rate(day) for day in event_days])
        post_days = [
            event.date + timedelta(days=offset)
            for offset in (2, 3, 4)
            if event.date + timedelta(days=offset) <= as_of
            and event.date + timedelta(days=offset) in traffic
        ]
        post = _mean([daily_rate(day) for day in post_days]) if post_days else expected
        event_excess = max(observed - expected, 0.0)
        residual = post - expected
        # Do not call something a one-off before enough post-event time has actually passed.
        persistence = (
            _clip(max(residual, 0.0) / max(event_excess, 0.08), high=2.0)
            if len(post_days) >= 2
            else 1.0
        )
        candidate = (
            event_excess,
            event,
            expected,
            observed,
            post,
            residual,
            persistence,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    _, event, expected, observed, post, residual, persistence = best
    observed_post_days = min(max((as_of - event.date).days - 1, 0), 3)
    return [event], expected, observed, post, residual, persistence, observed_post_days


def _slow_shift_cusum(
    daily: dict[date, float],
    traffic: dict[date, float],
    as_of: date,
    *,
    baseline_days: int = 21,
    monitor_days: int = 10,
) -> tuple[float, float, float, int, bool]:
    """One-sided CUSUM for small, sustained increases in per-thousand topic share."""

    def rate(day: date) -> float:
        return 1000.0 * daily.get(day, 0.0) / max(traffic.get(day, 1.0), 1.0)

    monitor_start = as_of - timedelta(days=monitor_days - 1)
    baseline_dates = [
        monitor_start - timedelta(days=offset) for offset in range(1, baseline_days + 1)
    ]
    baseline_rates = [rate(day) for day in baseline_dates if day in traffic]
    if len(baseline_rates) < 5:
        # At the left edge of a new dataset there is no 21-day history yet. A hard zero here
        # created a status cliff exactly when topics aged out of `new`. Use the earliest
        # observed days as a low-confidence warm-up baseline, while keeping at least three
        # later days for monitoring. Very short histories still remain unclassified.
        observed = sorted(day for day in traffic if day <= as_of)
        if len(observed) < 6:
            return 0.0, 0.0, 0.0, 0, False
        split = max(3, min(len(observed) // 2, len(observed) - 3))
        baseline_dates = observed[:split]
        monitor_dates = observed[split:][-monitor_days:]
        baseline_rates = [rate(day) for day in baseline_dates]
    else:
        monitor_dates = [monitor_start + timedelta(days=offset) for offset in range(monitor_days)]

    center = statistics.median(baseline_rates)
    mad = statistics.median(abs(value - center) for value in baseline_rates)
    sigma = max(1.4826 * mad, center * 0.12, 0.05)
    allowance = 0.5 * sigma
    threshold = 4.0 * sigma
    cumulative = 0.0
    exceed_days = 0
    for day in monitor_dates:
        value = rate(day)
        cumulative = max(0.0, cumulative + value - center - allowance)
        if value > center + allowance:
            exceed_days += 1
    score = cumulative / threshold
    alarm = score >= 1.0 and exceed_days >= 3
    return center, sigma, score, exceed_days, alarm


def _effective_first_seen(selected: list[Voice], daily: dict[date, float]) -> date:
    """Ignore isolated precursors before a topic becomes a recognisable community signal."""
    raw = Counter(voice.created_at.date() for voice in selected)
    first, last = min(raw), max(raw)
    days = (last - first).days + 1
    rolling: dict[date, int] = {}
    for offset in range(days):
        day = first + timedelta(days=offset)
        rolling[day] = sum(raw.get(day + timedelta(days=step), 0) for step in range(3))
    peak = max(rolling.values(), default=1)
    threshold = max(3, math.ceil(peak * 0.25))
    for day in sorted(rolling):
        active_in_window = sum(raw.get(day + timedelta(days=step), 0) > 0 for step in range(3))
        if rolling[day] >= threshold and (active_in_window >= 2 or raw.get(day, 0) >= threshold):
            return day
    return first


def calculate_topic_metrics(
    cluster_id: int,
    indices: list[int],
    voices: list[Voice],
    vectors: np.ndarray,
    as_of: date,
    traffic_by_day: dict[date, float],
    events: list[CalendarEvent],
    settings: Settings,
) -> TopicMetrics:
    selected = [voices[index] for index in indices]
    dates = [voice.created_at.date() for voice in selected]
    first_seen, last_seen = min(dates), max(dates)
    exact_counts = Counter(canonical_for_duplicates(voice.normalized) for voice in selected)
    user_counts = Counter(voice.user_name for voice in selected)
    platform_counts = Counter(voice.platform for voice in selected)
    user_day_seen: Counter[tuple[str, date]] = Counter()
    minute_text_counts: Counter[tuple[str, str]] = Counter()
    daily: defaultdict[date, float] = defaultdict(float)

    for voice in selected:
        day = voice.created_at.date()
        duplicate_key = canonical_for_duplicates(voice.normalized)
        user_day_key = (voice.user_name, day)
        user_day_seen[user_day_key] += 1
        duplicate_discount = 1.0 / math.sqrt(max(exact_counts[duplicate_key], 1))
        user_discount = 1.0 / math.sqrt(user_day_seen[user_day_key])
        reply_discount = 0.55 if voice.is_low_context_reply else 1.0
        daily[day] += voice.quality * duplicate_discount * user_discount * reply_discount
        minute = voice.created_at.strftime("%Y-%m-%dT%H:%M")
        minute_text_counts[(minute, duplicate_key)] += 1

    first_seen = _effective_first_seen(selected, daily)

    recent_dates = _date_range(as_of, settings.recent_days)
    previous_dates = _date_range(as_of, settings.previous_days, settings.recent_days)
    baseline_dates = _date_range(
        as_of, settings.baseline_days, settings.recent_days + settings.previous_days
    )
    recent = _window_rate(daily, traffic_by_day, recent_dates)
    previous = _window_rate(daily, traffic_by_day, previous_dates)
    baseline = _window_rate(daily, traffic_by_day, baseline_dates)
    growth_ratio = (recent + 0.08) / (previous + 0.08)
    previous_ratio = (previous + 0.08) / (baseline + 0.08)
    acceleration = math.log2(max(growth_ratio, 0.01)) - math.log2(max(previous_ratio, 0.01))

    message_count = len(selected)
    unique_texts = len([key for key in exact_counts if key])
    duplicate_ratio = 1.0 - unique_texts / max(message_count, 1)
    top_user_share = max(user_counts.values(), default=0) / max(message_count, 1)
    top_platform_share = max(platform_counts.values(), default=0) / max(message_count, 1)
    burst_share = max(minute_text_counts.values(), default=0) / max(message_count, 1)
    daily_raw = Counter(dates)
    peak_day = max(daily_raw, key=daily_raw.get)
    episode_dates = [day for day in dates if day >= first_seen]
    episode_daily = Counter(episode_dates)
    peak_day_share = max(episode_daily.values(), default=0) / max(len(episode_dates), 1)
    active_days = len({day for day in daily_raw if day >= first_seen})
    unique_users = len(user_counts)
    unique_platforms = len({voice.platform for voice in selected})
    unique_channels = len({voice.channel for voice in selected})
    diffusion = estimate_topic_diffusion(indices, voices, as_of)
    cusum_baseline, cusum_sigma, cusum_score, cusum_exceed_days, cusum_alarm = _slow_shift_cusum(
        daily, traffic_by_day, as_of
    )

    (
        nearby,
        event_expected,
        event_observed,
        event_post,
        event_residual,
        event_persistence,
        event_post_days,
    ) = _event_residual(
        daily,
        traffic_by_day,
        events,
        first_seen,
        peak_day,
        as_of,
    )

    centroid = vectors[indices].mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid)) or 1.0
    coherence = float(np.mean(vectors[indices] @ (centroid / centroid_norm)))

    user_diversity = _clip(unique_users / max(min(message_count, 12), 1))
    wording_diversity = _clip(1.0 - duplicate_ratio)
    temporal_naturalness = _clip(1.0 - burst_share * 2.2)
    platform_diversity = _clip(unique_platforms / 3.0)
    concentration_penalty = _clip(1.0 - top_user_share)
    authenticity = (
        0.26 * user_diversity
        + 0.25 * wording_diversity
        + 0.20 * temporal_naturalness
        + 0.13 * platform_diversity
        + 0.16 * concentration_penalty
    )

    novelty = _clip((recent - baseline) / (recent + baseline + 0.15) * 0.5 + 0.5)
    persistence = _clip(active_days / 8.0)
    spread = _clip((unique_platforms - 1) / 3.0)
    reproduction_signal = _clip(math.log2(max(diffusion.organic_rt, 0.25)) / 3.0 + 0.35)
    slow_shift_signal = _clip(cusum_score / 2.0)
    quality = _mean([voice.quality for voice in selected])
    # This is a neutral candidate-quality score. Status and studio priority are decided later.
    score = (
        0.18 * reproduction_signal * diffusion.confidence
        + 0.12 * slow_shift_signal
        + 0.16 * novelty
        + 0.13 * persistence
        + 0.11 * user_diversity
        + 0.10 * diffusion.community_breadth
        + 0.11 * _clip((coherence - 0.20) / 0.55)
        + 0.09 * quality
    )

    return TopicMetrics(
        cluster_id=cluster_id,
        message_indices=indices,
        first_seen=first_seen,
        last_seen=last_seen,
        weighted_volume=sum(daily.values()),
        recent_volume=recent,
        previous_volume=previous,
        baseline_volume=baseline,
        growth_ratio=growth_ratio,
        acceleration=acceleration,
        unique_users=unique_users,
        unique_platforms=unique_platforms,
        unique_channels=unique_channels,
        active_days=active_days,
        duplicate_ratio=duplicate_ratio,
        top_user_share=top_user_share,
        top_platform_share=top_platform_share,
        burst_share=burst_share,
        peak_day=peak_day,
        peak_day_share=peak_day_share,
        post_peak_observed_days=max((as_of - peak_day).days, event_post_days),
        coherence=coherence,
        authenticity=authenticity,
        novelty=novelty,
        persistence=persistence,
        spread=spread,
        community_breadth=diffusion.community_breadth,
        topic_rt_raw=diffusion.raw_rt,
        topic_rt=diffusion.organic_rt,
        rt_confidence=diffusion.confidence,
        cusum_baseline=cusum_baseline,
        cusum_sigma=cusum_sigma,
        cusum_score=cusum_score,
        cusum_exceed_days=cusum_exceed_days,
        cusum_alarm=cusum_alarm,
        status="new",
        score=score,
        nearby_events=nearby,
        event_expected_rate=event_expected,
        event_observed_rate=event_observed,
        event_post_rate=event_post,
        event_residual=event_residual,
        event_persistence=event_persistence,
    )
