from __future__ import annotations

import math
import re
from datetime import date, timedelta

from .config import Settings
from .models import (
    CandidateEvidence,
    EvidenceFeature,
    StatusDecision,
    TopicMetrics,
)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _feature(
    raw: float | None,
    normalized: float | None,
    reliability: float,
    *,
    missing: bool = False,
) -> EvidenceFeature:
    return EvidenceFeature(
        raw=raw,
        normalized=None if missing else _clip(normalized or 0.0),
        reliability=_clip(reliability),
        missing=missing,
    )


def _weighted_available(
    features: dict[str, EvidenceFeature], weights: dict[str, float]
) -> tuple[float, float]:
    available = [
        (name, weight)
        for name, weight in weights.items()
        if name in features and not features[name].missing and features[name].normalized is not None
    ]
    if not available:
        return 0.0, 0.0
    available_weight = sum(weight for _, weight in available)
    score = (
        sum(weight * float(features[name].normalized) for name, weight in available)
        / available_weight
    )
    reliability = (
        sum(weight * features[name].reliability for name, weight in available) / available_weight
    )
    # Missing evidence is not silently converted to a healthy zero. The score uses only what is
    # observable, while reliability records how much of the intended evidence was present.
    reliability *= available_weight / sum(weights.values())
    return _clip(score), _clip(reliability)


def _weights(names: tuple[str, ...], values: tuple[float, ...]) -> dict[str, float]:
    if len(names) != len(values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("Evidence weights must match their features and sum to 1")
    return dict(zip(names, values, strict=True))


def build_candidate_evidence(metrics: TopicMetrics, as_of: date) -> CandidateEvidence:
    age_days = max((as_of - metrics.first_seen).days + 1, 1)
    sample_reliability = _clip(math.log1p(metrics.unique_users) / math.log(31.0))
    active_reliability = _clip(metrics.active_days / 8.0)
    history_reliability = _clip(age_days / 14.0)

    rt_growth = _clip((metrics.topic_rt - 0.95) / 0.65)
    rt_decline = _clip((1.05 - metrics.topic_rt) / 0.65)
    rt_reliability = _clip(metrics.rt_confidence * sample_reliability)
    window_growth = _clip((metrics.growth_ratio - 1.0) / 0.80)
    window_growth_reliability = _clip(active_reliability * min(age_days / 10.0, 1.0))

    cusum_strength = _clip(metrics.cusum_score / 2.0)
    cusum_persistence = _clip(metrics.cusum_exceed_days / 6.0)
    cusum_reliability = _clip(active_reliability * history_reliability)

    has_event = bool(metrics.nearby_events)
    event_observed_days = (
        min(
            metrics.post_peak_observed_days,
            max((as_of - metrics.nearby_events[-1].date).days, 0),
        )
        if has_event
        else 0
    )
    event_reliability = _clip(event_observed_days / 3.0) if has_event else 0.0
    event_persistence = _clip(metrics.event_persistence / 1.1) if has_event else None
    post_event_decay = _clip(1.0 - metrics.event_persistence) if has_event else None
    event_lift = (
        (metrics.event_observed_rate + 0.08) / (metrics.event_expected_rate + 0.08)
        if has_event
        else None
    )
    event_explanation = (
        _clip(((event_lift or 1.0) - 1.0) / 2.0) * float(post_event_decay or 0.0)
        if has_event
        else None
    )

    historical = max(metrics.previous_volume, metrics.baseline_volume)
    history_missing = historical < 0.08 or age_days < 7
    recent_retention = metrics.recent_volume / max(historical, 0.08)
    sustained_decline = _clip((0.95 - recent_retention) / 0.70)
    decline_reliability = _clip(active_reliability * history_reliability)

    stress_robustness = _clip(metrics.survival_score)
    decline_robustness = sustained_decline * stress_robustness
    source_concentration = max(metrics.top_user_share, metrics.top_platform_share)
    duplicate_anomaly = _clip((metrics.duplicate_ratio - 0.42) / 0.48)
    synchronisation_anomaly = _clip((metrics.burst_share - 0.04) / 0.36)
    source_anomaly = _clip((source_concentration - 0.45) / 0.50)
    copy_collapse = _clip(
        1.0
        - min(
            metrics.stress_tests.get("deduplicated", 1.0),
            metrics.survival_score,
        )
    )
    authenticity_risk = _clip(
        0.35 * duplicate_anomaly
        + 0.25 * synchronisation_anomaly
        + 0.20 * source_anomaly
        + 0.20 * copy_collapse
    )

    features = {
        "mdl_gain": _feature(
            metrics.mdl_gain_bits,
            metrics.mdl_score,
            _clip(math.log1p(max(metrics.unique_users, 0)) / math.log(21.0)),
        ),
        "organic_rt": _feature(metrics.topic_rt, rt_growth, rt_reliability),
        "window_growth": _feature(
            metrics.growth_ratio,
            window_growth,
            window_growth_reliability,
        ),
        "rt_decline": _feature(metrics.topic_rt, rt_decline, rt_reliability),
        "absence_of_new_generations": _feature(metrics.topic_rt, rt_decline, rt_reliability),
        "cusum_strength": _feature(metrics.cusum_score, cusum_strength, cusum_reliability),
        "cusum_persistence": _feature(
            float(metrics.cusum_exceed_days), cusum_persistence, cusum_reliability
        ),
        "event_adjusted_persistence": _feature(
            metrics.event_persistence if has_event else None,
            event_persistence,
            event_reliability,
            missing=not has_event,
        ),
        "post_event_decay": _feature(
            metrics.event_persistence if has_event else None,
            post_event_decay,
            event_reliability,
            missing=not has_event,
        ),
        "event_explanation": _feature(
            event_lift,
            event_explanation,
            event_reliability,
            missing=not has_event,
        ),
        "peak_concentration": _feature(
            metrics.peak_day_share,
            _clip((metrics.peak_day_share - 0.20) / 0.55),
            _clip(metrics.post_peak_observed_days / 3.0),
        ),
        "participant_breadth": _feature(
            metrics.community_breadth,
            metrics.community_breadth,
            sample_reliability,
        ),
        "stress_robustness": _feature(
            metrics.survival_score,
            stress_robustness,
            sample_reliability,
        ),
        "sustained_activity_decline": _feature(
            recent_retention if not history_missing else None,
            sustained_decline,
            decline_reliability,
            missing=history_missing,
        ),
        "decline_robustness": _feature(
            recent_retention if not history_missing else None,
            decline_robustness,
            decline_reliability,
            missing=history_missing,
        ),
        "duplicate_anomaly": _feature(
            metrics.duplicate_ratio,
            duplicate_anomaly,
            sample_reliability,
        ),
        "synchronisation_anomaly": _feature(
            metrics.burst_share,
            synchronisation_anomaly,
            sample_reliability,
        ),
        "temporal_synchronisation": _feature(
            metrics.burst_share,
            synchronisation_anomaly,
            sample_reliability,
        ),
        "source_concentration_anomaly": _feature(
            source_concentration,
            source_anomaly,
            sample_reliability,
        ),
        "stress_collapse": _feature(
            metrics.survival_score,
            copy_collapse,
            sample_reliability,
        ),
        "authenticity_risk": _feature(
            authenticity_risk,
            authenticity_risk,
            sample_reliability,
        ),
    }
    missing = tuple(name for name, feature in features.items() if feature.missing)
    return CandidateEvidence(
        topic_id=metrics.cluster_id,
        independent_player_count=metrics.unique_users,
        topic_age_days=age_days,
        active_days=metrics.active_days,
        evidence_window={
            "first_seen": metrics.first_seen.isoformat(),
            "last_seen": metrics.last_seen.isoformat(),
            "as_of": as_of.isoformat(),
            "recent_start": (as_of - timedelta(days=3)).isoformat(),
        },
        features=features,
        missing_features=missing,
    )


def decide_status(
    metrics: TopicMetrics,
    as_of: date,
    settings: Settings,
) -> StatusDecision:
    evidence = build_candidate_evidence(metrics, as_of)
    features = evidence.features

    growing, growing_rel = _weighted_available(
        features,
        _weights(
            (
                "organic_rt",
                "cusum_persistence",
                "event_adjusted_persistence",
                "participant_breadth",
            ),
            settings.growing_weights,
        ),
    )
    # Bridge the transition out of `new`: a robust multi-day rise in normalized share can
    # support growing before Topic Rt and a 21-day CUSUM have both matured. This path still
    # requires breadth and stress-test survival, so one loud account or one spike cannot pass.
    window_growth = float(features["window_growth"].normalized or 0.0)
    stress_robustness = float(features["stress_robustness"].normalized or 0.0)
    participant_breadth = float(features["participant_breadth"].normalized or 0.0)
    volume_growth = _clip(
        0.45 * window_growth
        + 0.25 * stress_robustness
        + 0.20 * participant_breadth
        + 0.10 * _clip(evidence.active_days / 8.0)
    )
    volume_growth_rel = _clip(
        0.45 * features["window_growth"].reliability
        + 0.30 * features["stress_robustness"].reliability
        + 0.25 * features["participant_breadth"].reliability
    )
    volume_growth_supported = (
        evidence.active_days >= 4
        and evidence.topic_age_days <= 14
        and metrics.rt_confidence >= 0.35
        and window_growth >= 0.35
        and stress_robustness >= 0.50
        and participant_breadth >= 0.45
    )
    if volume_growth_supported and volume_growth > growing:
        growing, growing_rel = volume_growth, volume_growth_rel
    fading, fading_rel = _weighted_available(
        features,
        _weights(
            ("rt_decline", "sustained_activity_decline", "decline_robustness"),
            settings.fading_weights,
        ),
    )
    # A mature topic can fade by failing to recruit a new participant generation even when
    # raw post volume is roughly flat. Requiring both volume decline and Rt decline caused
    # weeks of otherwise valid candidates to receive no status at all.
    generation_fading = _clip(
        0.55 * float(features["rt_decline"].normalized or 0.0)
        + 0.30 * stress_robustness
        + 0.15 * _clip(evidence.topic_age_days / 14.0)
    )
    generation_fading_rel = _clip(
        0.60 * features["rt_decline"].reliability
        + 0.25 * features["stress_robustness"].reliability
        + 0.15 * _clip(evidence.topic_age_days / 14.0)
    )
    historical = max(metrics.previous_volume, metrics.baseline_volume)
    generation_fading_supported = (
        evidence.topic_age_days >= 14
        and historical >= 0.08
        and metrics.recent_volume <= historical * 0.90
        and float(features["rt_decline"].normalized or 0.0) >= 0.50
        and stress_robustness >= 0.55
    )
    if generation_fading_supported and generation_fading > fading:
        fading, fading_rel = generation_fading, generation_fading_rel
    one_off, one_off_rel = _weighted_available(
        features,
        _weights(
            (
                "peak_concentration",
                "post_event_decay",
                "event_explanation",
                "absence_of_new_generations",
            ),
            settings.one_off_weights,
        ),
    )
    authenticity_risk, authenticity_rel = _weighted_available(
        features,
        _weights(
            (
                "duplicate_anomaly",
                "synchronisation_anomaly",
                "source_concentration_anomaly",
                "stress_collapse",
            ),
            settings.authenticity_weights,
        ),
    )

    mdl = features["mdl_gain"]
    observation_short = evidence.topic_age_days <= 7 or evidence.active_days < 4
    event_terms = {
        "周年",
        "活动",
        "活動",
        "奖励",
        "獎勵",
        "福利",
        "礼包",
        "禮包",
        "补偿",
        "補償",
    }
    calendar_entities = {
        entity.strip()
        for event in metrics.nearby_events
        for entity in re.findall(r"「([^」]+)」", event.notes)
        if entity.strip()
    }
    event_specific = bool((event_terms | calendar_entities).intersection(metrics.top_terms))
    event_age = (
        min((as_of - event.date).days for event in metrics.nearby_events)
        if metrics.nearby_events
        else 999
    )
    event_triggered_emergence = (
        bool(metrics.nearby_events)
        and 0 <= event_age <= 4
        and metrics.event_persistence >= 0.55
        and metrics.recent_volume >= 0.50
        and event_specific
    )
    new_semantics = metrics.mdl_gain_bits > 0 and float(mdl.normalized or 0.0) >= 0.52
    new_score = _clip(
        0.40 * float(mdl.normalized or 0.0)
        + 0.25 * _clip(metrics.novelty)
        + 0.20 * _clip(metrics.unique_users / 8.0)
        + 0.15 * (1.0 if observation_short else 0.0)
    )
    new_rel = _clip(
        0.45 * mdl.reliability
        + 0.35 * _clip(metrics.unique_users / 10.0)
        + 0.20 * _clip(metrics.active_days / 5.0)
    )

    scores = {
        "new": round(new_score, 6),
        "growing": round(growing, 6),
        "fading": round(fading, 6),
        "one-off": round(one_off, 6),
        "not real": round(authenticity_risk, 6),
    }
    reliabilities = {
        "new": new_rel,
        "growing": growing_rel,
        "fading": fading_rel,
        "one-off": one_off_rel,
        "not real": authenticity_rel,
    }

    copy_family = max(
        float(features["duplicate_anomaly"].normalized or 0.0),
        float(features["stress_collapse"].normalized or 0.0),
    )
    suspicious_families = sum(
        value >= 0.60
        for value in (
            copy_family,
            float(features["synchronisation_anomaly"].normalized or 0.0),
            float(features["source_concentration_anomaly"].normalized or 0.0),
        )
    )
    authenticity_state = (
        "suspicious"
        if authenticity_risk >= settings.min_status_score
        and suspicious_families >= 2
        and authenticity_rel >= settings.min_status_reliability
        else "organic"
        if authenticity_risk < 0.50
        else "undetermined"
    )

    participant_support = max(
        float(features["organic_rt"].normalized or 0.0),
        float(features["participant_breadth"].normalized or 0.0),
    )
    persistence_support = max(
        float(features["cusum_persistence"].normalized or 0.0),
        float(features["event_adjusted_persistence"].normalized or 0.0),
    )
    historical = max(metrics.previous_volume, metrics.baseline_volume)
    one_off_condition = (
        metrics.post_peak_observed_days >= 2
        and float(features["peak_concentration"].normalized or 0.0) >= 0.50
        and (
            float(features["post_event_decay"].normalized or 0.0) >= 0.50
            or metrics.recent_volume < max(metrics.previous_volume * 0.60, 0.20)
        )
    )
    conditions = {
        "growing": (
            evidence.active_days >= 3
            and participant_support >= 0.50
            and (persistence_support >= 0.50 or volume_growth_supported)
        ),
        "fading": (
            generation_fading_supported
            or (
                evidence.topic_age_days >= 7
                and historical >= 0.08
                and metrics.recent_volume < historical
                and float(features["rt_decline"].normalized or 0.0) >= 0.45
                and float(features["sustained_activity_decline"].normalized or 0.0) >= 0.45
            )
        ),
        "one-off": one_off_condition,
        "new": (
            new_semantics
            and (observation_short or event_triggered_emergence)
            and not one_off_condition
            and metrics.unique_users >= settings.min_topic_users
            and authenticity_state != "suspicious"
        ),
    }
    if conditions["new"] and (observation_short or event_triggered_emergence):
        # During the explicit onset window, `new` has semantic precedence. The same early
        # rise must not create a near-tie with the warm-up growing bridge; after day seven,
        # the bridge can take over naturally.
        conditions["growing"] = False

    temporal_scores = {
        name: scores[name] for name in ("new", "growing", "fading", "one-off") if conditions[name]
    }
    temporal_state = "undetermined"
    official_status = None
    satisfied: list[str] = []
    counter: list[str] = []
    uncertainties: list[str] = []
    chosen_reliability = 0.0

    if authenticity_state == "suspicious":
        official_status = "not real"
        chosen_reliability = authenticity_rel
        satisfied.append("至少两类相对独立的异常传播证据同时成立")
        if metrics.burst_share < 0.10:
            counter.append("没有明显的分钟级同步爆发")
    elif temporal_scores:
        ordered = sorted(temporal_scores.items(), key=lambda item: (-item[1], item[0]))
        best_name, best_score = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        chosen_reliability = reliabilities[best_name]
        if (
            best_score >= settings.min_status_score
            and best_score - runner_up >= settings.min_status_margin
            and chosen_reliability >= settings.min_status_reliability
        ):
            temporal_state = best_name  # type: ignore[assignment]
            official_status = best_name  # type: ignore[assignment]
            satisfied.append(f"{best_name} 证据强度和领先幅度均达到配置门槛")
            if best_name == "new" and event_triggered_emergence and not observation_short:
                satisfied.append("已知事件触发了新一轮具体玩家讨论")
        else:
            if best_score < settings.min_status_score:
                uncertainties.append("最高状态证据不足")
            if best_score - runner_up < settings.min_status_margin:
                uncertainties.append("前两种状态的证据过于接近")
            if chosen_reliability < settings.min_status_reliability:
                uncertainties.append("样本或观察窗口不足")
    else:
        uncertainties.append("没有状态同时满足必要条件")

    if evidence.missing_features:
        uncertainties.append("部分事件证据不适用或尚不可观察")
    if metrics.rt_confidence < 0.35:
        counter.append("Topic Rt 的样本可靠性偏低")
    if metrics.cusum_alarm and metrics.topic_rt < 0.90:
        counter.append("讨论占比上移，但新参与者扩散较弱")
    if metrics.growth_ratio > 1.20 and official_status == "fading":
        uncertainties.append("窗口声量上涨与衰退判断冲突")
        temporal_state = "undetermined"
        official_status = None

    reason = (
        f"最高时间状态为 {temporal_state}；状态证据可靠性 {chosen_reliability:.2f}。"
        if official_status != "not real"
        else f"传播真实性风险为 {authenticity_risk:.2f}，且异常来自多个证据组。"
    )
    metrics.candidate_evidence = evidence
    decision = StatusDecision(
        temporal_state=temporal_state,  # type: ignore[arg-type]
        authenticity_state=authenticity_state,  # type: ignore[arg-type]
        official_status=official_status,  # type: ignore[arg-type]
        evidence_scores=scores,
        evidence_reliability=round(chosen_reliability, 6),
        satisfied_conditions=tuple(satisfied),
        counter_evidence=tuple(counter),
        uncertainties=tuple(dict.fromkeys(uncertainties)),
        reason=reason,
    )
    metrics.status_decision = decision
    if official_status is not None:
        metrics.status = official_status
    return decision


def classify_topics(topics: list[TopicMetrics], as_of: date, settings: Settings) -> None:
    for topic in topics:
        decide_status(topic, as_of, settings)
