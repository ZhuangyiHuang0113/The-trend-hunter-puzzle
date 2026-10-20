from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date

import numpy as np

from .clustering import _calendar_terms, cluster_topics, tokenize_for_topics, top_terms_for_indices
from .config import Settings
from .decision import classify_topics
from .diagnostics import enrich_topic_diagnostics
from .diffusion import enrich_mdl_admission
from .evidence import select_evidence
from .features import build_features
from .models import CalendarEvent, TopicCandidate, TopicMetrics, Trend, Voice
from .preprocess import preprocess_voices
from .selection import (
    action_lenses,
    score_and_qualify_topics,
    select_top_candidates,
    topic_mechanism_profile,
)
from .semantic import refine_labels
from .signals import calculate_topic_metrics
from .threads import attach_thread_context

LOGGER = logging.getLogger(__name__)


def _calendar_entity_signals(voice: Voice, events: list[CalendarEvent]) -> tuple[str, ...]:
    """Return weak, as-of-safe hints for entities explicitly named by the calendar."""
    entities = {
        entity.strip()
        for event in events
        for entity in re.findall(r"「([^」]+)」", event.notes)
        if entity.strip()
    }
    return tuple(
        f"calendar_entity:{entity}" for entity in sorted(entities) if entity in voice.semantic_text
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _cross_phase_episode(left: TopicMetrics, right: TopicMetrics) -> tuple[int, int] | None:
    """Return (older/declining, emerging) ids when one story has two lifecycle episodes."""
    left_status = left.status_decision.official_status if left.status_decision else None
    right_status = right.status_decision.official_status if right.status_decision else None
    emerging = {"new", "growing"}
    non_emerging = {"fading", "one-off", None}
    if left_status in emerging and right_status in non_emerging:
        return right.cluster_id, left.cluster_id
    if right_status in emerging and left_status in non_emerging:
        return left.cluster_id, right.cluster_id
    return None


def _duplicate_story(
    left: TopicMetrics,
    right: TopicMetrics,
    profiles: dict[int, dict[str, object]],
) -> bool:
    left_decision = left.status_decision
    right_decision = right.status_decision
    if left_decision is None or right_decision is None:
        return False
    if left_decision.authenticity_state != right_decision.authenticity_state:
        return False

    left_profile = profiles[left.cluster_id]
    right_profile = profiles[right.cluster_id]
    semantic = float(left_profile["center"] @ right_profile["center"])
    left_users = left_profile["users"]
    right_users = right_profile["users"]
    assert isinstance(left_users, set) and isinstance(right_users, set)
    participant_overlap = _jaccard(left_users, right_users)
    left_actions = left_profile["actions"]
    right_actions = right_profile["actions"]
    assert isinstance(left_actions, set) and isinstance(right_actions, set)
    action_overlap = _jaccard(left_actions, right_actions)
    left_mechanism = str(left_profile["mechanism"])
    right_mechanism = str(right_profile["mechanism"])
    left_dominance = float(left_profile["dominance"])
    right_dominance = float(right_profile["dominance"])
    same_mechanism = bool(left_mechanism) and left_mechanism == right_mechanism
    different_known_mechanisms = (
        bool(left_mechanism)
        and bool(right_mechanism)
        and left_mechanism != right_mechanism
        and min(left_dominance, right_dominance) >= 0.55
    )
    term_overlap = _jaccard(set(left.top_terms), set(right.top_terms))
    # A shared product mechanism is necessary but not sufficient. Two art topics, for example,
    # can require different studio actions even when both contain the word "美术".
    if same_mechanism:
        return action_overlap >= 0.50 and (
            participant_overlap >= 0.20
            or (semantic >= 0.62 and (participant_overlap >= 0.06 or term_overlap >= 0.10))
            or (semantic >= 0.80 and term_overlap >= 0.18)
        )
    if different_known_mechanisms:
        return False
    if (
        left_decision.authenticity_state == "suspicious"
        and semantic >= 0.76
        and term_overlap >= 0.20
    ):
        return True
    return action_overlap >= 0.50 and (
        (semantic >= 0.86 and participant_overlap >= 0.14)
        or (semantic >= 0.94 and participant_overlap >= 0.06)
    )


def _merge_duplicate_topics(
    topics: list[TopicMetrics],
    voices: list[Voice],
    vectors: np.ndarray,
    as_of: date,
    events: list[CalendarEvent],
    settings: Settings,
) -> tuple[list[TopicMetrics], dict[int, int]]:
    parent = {topic.cluster_id: topic.cluster_id for topic in topics}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        root, child = sorted((left_root, right_root))
        parent[child] = root

    ordered = sorted(topics, key=lambda topic: topic.cluster_id)
    suppressed_episode: dict[int, int] = {}
    profiles: dict[int, dict[str, object]] = {}
    for topic in ordered:
        center = vectors[topic.message_indices].mean(axis=0)
        center /= float(np.linalg.norm(center)) or 1.0
        mechanism, dominance, _, _ = topic_mechanism_profile(topic, voices)
        profiles[topic.cluster_id] = {
            "center": center,
            "users": {voices[index].user_name for index in topic.message_indices},
            "actions": action_lenses(topic, voices),
            "mechanism": mechanism,
            "dominance": dominance,
        }
    for position, left in enumerate(ordered):
        for right in ordered[position + 1 :]:
            if _duplicate_story(left, right, profiles):
                episode = _cross_phase_episode(left, right)
                if episode is not None:
                    declining_id, emerging_id = episode
                    suppressed_episode[declining_id] = emerging_id
                    continue
                union(left.cluster_id, right.cluster_id)

    groups: defaultdict[int, list[TopicMetrics]] = defaultdict(list)
    for topic in ordered:
        if topic.cluster_id in suppressed_episode:
            continue
        groups[find(topic.cluster_id)].append(topic)
    merged_into = {
        topic.cluster_id: root
        for root, members in groups.items()
        for topic in members
        if topic.cluster_id != root
    }
    merged_into.update(suppressed_episode)
    if not merged_into:
        return topics, {}

    traffic: defaultdict[date, float] = defaultdict(float)
    for voice in voices:
        traffic[voice.created_at.date()] += max(voice.quality, 0.1)
    global_df = Counter(
        token for voice in voices for token in set(tokenize_for_topics(voice.semantic_text))
    )
    rebuilt: list[TopicMetrics] = []
    for root, members in sorted(groups.items()):
        if len(members) == 1:
            rebuilt.append(members[0])
            continue
        indices = sorted(
            {index for member in members for index in member.message_indices},
            key=lambda index: (voices[index].created_at, voices[index].id),
        )
        metrics = calculate_topic_metrics(
            root,
            indices,
            voices,
            vectors,
            as_of,
            dict(traffic),
            events,
            settings,
        )
        metrics.top_terms = top_terms_for_indices(indices, voices, global_df)
        if not metrics.nearby_events:
            contextual_events = [
                event
                for event in events
                if 0 <= (as_of - event.date).days <= 4
                and _calendar_terms(event).intersection(metrics.top_terms)
            ]
            if contextual_events:
                metrics.nearby_events = [contextual_events[-1]]
        metrics.merged_from = sorted(member.cluster_id for member in members)
        rebuilt.append(metrics)

    enrich_mdl_admission(rebuilt, voices, vectors)
    enrich_topic_diagnostics(rebuilt, voices)
    classify_topics(rebuilt, as_of, settings)
    score_and_qualify_topics(rebuilt, voices, as_of, settings)
    return rebuilt, merged_into


def _merge_final_story_duplicates(
    candidates: list[TopicCandidate],
    voices: list[Voice],
    vectors: np.ndarray,
    as_of: date,
    events: list[CalendarEvent],
    settings: Settings,
) -> tuple[list[TopicCandidate], dict[int, int]]:
    """Final guard for the same labelled player story split across temporal clusters."""
    groups: defaultdict[str, list[TopicCandidate]] = defaultdict(list)
    for candidate in candidates:
        # Only an explicit semantic merge key may join candidates here. Local fallback titles
        # are deliberately templated and therefore unsafe as identity keys.
        key = candidate.merge_key.strip()
        groups[key or f"__candidate_{candidate.metrics.cluster_id}"].append(candidate)

    traffic: defaultdict[date, float] = defaultdict(float)
    for voice in voices:
        traffic[voice.created_at.date()] += max(voice.quality, 0.1)
    global_df = Counter(
        token for voice in voices for token in set(tokenize_for_topics(voice.semantic_text))
    )
    result: list[TopicCandidate] = []
    merged_into: dict[int, int] = {}
    for members in groups.values():
        if len(members) == 1:
            result.append(members[0])
            continue
        root = min(member.metrics.cluster_id for member in members)
        indices = sorted(
            {index for member in members for index in member.metrics.message_indices},
            key=lambda index: (voices[index].created_at, voices[index].id),
        )
        metrics = calculate_topic_metrics(
            root, indices, voices, vectors, as_of, dict(traffic), events, settings
        )
        metrics.top_terms = top_terms_for_indices(indices, voices, global_df)
        metrics.merged_from = sorted(member.metrics.cluster_id for member in members)
        enrich_mdl_admission([metrics], voices, vectors)
        enrich_topic_diagnostics([metrics], voices)
        classify_topics([metrics], as_of, settings)
        score_and_qualify_topics([metrics], voices, as_of, settings)
        if not metrics.eligibility_passed:
            continue
        evidence = select_evidence(metrics, voices, vectors, settings.max_evidence)
        refreshed = TopicCandidate(
            metrics=metrics,
            evidence_indices=evidence,
            title="",
            title_en="",
            category="",
            why="",
            priority_components=dict(metrics.priority_components),
            priority_score=metrics.priority_score,
            priority_reliability=metrics.priority_reliability,
        )
        result.extend(
            refine_labels([refreshed], voices, replace(settings, semantic_mode="offline"))
        )
        merged_into.update(
            {
                member.metrics.cluster_id: root
                for member in members
                if member.metrics.cluster_id != root
            }
        )
    return result, merged_into


def _audit_entry(
    topic: TopicMetrics,
    voices: list[Voice],
    *,
    rank_before: int | None,
    rank_after: int | None,
    selected: bool,
    rejection_reason: str,
    representative_indices: list[int] | None = None,
) -> dict[str, object]:
    mechanism, dominance, mixed_share, mechanisms = topic_mechanism_profile(topic, voices)
    return {
        "candidate_id": topic.cluster_id,
        "eligibility_passed": topic.eligibility_passed,
        "eligibility_reasons": topic.eligibility_reasons,
        "merged_from": topic.merged_from,
        "merged_into": None,
        "top_terms": topic.top_terms,
        "dominant_mechanism": mechanism or None,
        "mechanism_dominance": round(dominance, 6),
        "mixed_mechanism_share": round(mixed_share, 6),
        "observed_mechanisms": sorted(mechanisms),
        "window_metrics": {
            "recent_volume": round(topic.recent_volume, 6),
            "previous_volume": round(topic.previous_volume, 6),
            "baseline_volume": round(topic.baseline_volume, 6),
            "growth_ratio": round(topic.growth_ratio, 6),
            "topic_rt": round(topic.topic_rt, 6),
            "rt_reliability": round(topic.rt_confidence, 6),
            "cusum_score": round(topic.cusum_score, 6),
            "cusum_exceed_days": topic.cusum_exceed_days,
        },
        "candidate_evidence": (
            topic.candidate_evidence.to_dict() if topic.candidate_evidence else None
        ),
        "status_decision": topic.status_decision.to_dict() if topic.status_decision else None,
        "selected_status": (
            topic.status_decision.official_status if topic.status_decision else None
        ),
        "priority_components": topic.priority_components,
        "priority_score": topic.priority_score,
        "priority_reliability": topic.priority_reliability,
        "rank_before_deduplication": rank_before,
        "rank_after_deduplication": rank_after,
        "selected": selected,
        "selection_reason": ("eligible_then_ranked_by_studio_priority" if selected else ""),
        "rejection_reason": rejection_reason,
        "evidence_voice_ids": [voices[index].id for index in (representative_indices or [])],
        "representative_voice": (
            {
                "voice_id": voices[representative_indices[0]].id,
                "text": voices[representative_indices[0]].content,
                "created_at": voices[representative_indices[0]].created_at.isoformat(),
            }
            if representative_indices
            else None
        ),
        "source_voice_ids": [voices[index].id for index in topic.message_indices],
    }


def discover_trends(
    voices: list[Voice],
    events: list[CalendarEvent],
    as_of: date,
    settings: Settings,
) -> tuple[list[Trend], dict[str, object]]:
    processed = attach_thread_context(preprocess_voices(voices))
    eligible = [
        voice
        for voice in processed
        if voice.quality >= settings.min_quality and bool(voice.semantic_text.strip())
    ]
    if len(eligible) < settings.min_topic_messages:
        return [], {"loaded": len(voices), "eligible": len(eligible), "feature_mode": "none"}

    features = build_features(
        (voice.semantic_text for voice in eligible),
        settings,
        weak_signals=(
            (
                *voice.signal_groups,
                *(
                    ("competitor_comparison",)
                    if "competitor_comparison" in voice.pragmatic_flags
                    else ()
                ),
                *(("possible_irony",) if "possible_irony" in voice.pragmatic_flags else ()),
                *_calendar_entity_signals(voice, events),
            )
            for voice in eligible
        ),
    )
    raw_topics = cluster_topics(eligible, features.vectors, as_of, events, settings)
    enrich_mdl_admission(raw_topics, eligible, features.vectors)
    enrich_topic_diagnostics(raw_topics, eligible)
    classify_topics(raw_topics, as_of, settings)
    score_and_qualify_topics(raw_topics, eligible, as_of, settings)

    before_order = sorted(
        raw_topics,
        key=lambda topic: (-topic.priority_score, topic.cluster_id),
    )
    rank_before = {topic.cluster_id: rank for rank, topic in enumerate(before_order, 1)}

    merged_topics, merged_into = _merge_duplicate_topics(
        raw_topics,
        eligible,
        features.vectors,
        as_of,
        events,
        settings,
    )
    if not merged_into:
        # The merge path recalculates these values. Keep the singletons on the same explicit path.
        classify_topics(merged_topics, as_of, settings)
        score_and_qualify_topics(merged_topics, eligible, as_of, settings)

    after_order = sorted(
        merged_topics,
        key=lambda topic: (-topic.priority_score, topic.cluster_id),
    )
    rank_after = {topic.cluster_id: rank for rank, topic in enumerate(after_order, 1)}

    qualified = [topic for topic in after_order if topic.eligibility_passed]
    candidate_rows: list[TopicCandidate] = []
    for topic in qualified:
        evidence = select_evidence(topic, eligible, features.vectors, settings.max_evidence)
        candidate_rows.append(
            TopicCandidate(
                metrics=topic,
                evidence_indices=evidence,
                title="",
                title_en="",
                category="",
                why="",
                priority_components=dict(topic.priority_components),
                priority_score=topic.priority_score,
                priority_reliability=topic.priority_reliability,
            )
        )

    head = candidate_rows[: settings.llm_candidates]
    tail = candidate_rows[settings.llm_candidates :]
    semantic_trace: dict[str, object] = {}
    labelled = refine_labels(head, eligible, settings, trace=semantic_trace)
    if tail:
        labelled.extend(refine_labels(tail, eligible, replace(settings, semantic_mode="offline")))
    if bool(semantic_trace.get("used_llm")):
        labelled, final_merged_into = _merge_final_story_duplicates(
            labelled, eligible, features.vectors, as_of, events, settings
        )
    else:
        final_merged_into = {}
    selected, top_k_rejections = select_top_candidates(labelled, settings.max_trends)
    selected_ids = {candidate.metrics.cluster_id for candidate in selected}
    labelled_ids = {candidate.metrics.cluster_id for candidate in labelled}
    representative_by_id = {
        candidate.metrics.cluster_id: candidate.evidence_indices for candidate in labelled
    }
    merged_into.update(final_merged_into)

    trends = [
        Trend(
            title=candidate.title,
            title_en=candidate.title_en,
            category=candidate.category,
            status=candidate.metrics.status,
            first_seen=candidate.metrics.first_seen.isoformat(),
            voice_ids=[eligible[index].id for index in candidate.evidence_indices],
            why=candidate.why,
            representative_voice={
                "voice_id": eligible[candidate.evidence_indices[0]].id,
                "text": eligible[candidate.evidence_indices[0]].content,
            },
        )
        for candidate in selected
    ]

    audit: list[dict[str, object]] = []
    for topic in after_order:
        rejection = topic.rejection_reason
        if topic.cluster_id in labelled_ids and topic.cluster_id not in selected_ids:
            rejection = top_k_rejections.get(topic.cluster_id, "semantic_review_rejected")
        audit.append(
            _audit_entry(
                topic,
                eligible,
                rank_before=rank_before.get(topic.cluster_id),
                rank_after=rank_after.get(topic.cluster_id),
                selected=topic.cluster_id in selected_ids,
                rejection_reason=rejection,
                representative_indices=representative_by_id.get(topic.cluster_id),
            )
        )
    for child, root in sorted(merged_into.items()):
        child_topic = next(topic for topic in raw_topics if topic.cluster_id == child)
        entry = _audit_entry(
            child_topic,
            eligible,
            rank_before=rank_before.get(child),
            rank_after=None,
            selected=False,
            rejection_reason="duplicate_topic",
        )
        entry["merged_into"] = root
        audit.append(entry)

    selected_slots: list[int | None] = [candidate.metrics.cluster_id for candidate in selected]
    selected_slots.extend([None] * max(settings.max_trends - len(selected_slots), 0))
    diagnostics = {
        "loaded": len(voices),
        "eligible": len(eligible),
        "feature_mode": features.mode,
        "semantic_runtime": {
            "requested_mode": settings.semantic_mode,
            **semantic_trace,
        },
        "lexicon_runtime": {
            "matched_messages": sum(bool(voice.signal_groups) for voice in eligible),
            "matched_share": round(
                sum(bool(voice.signal_groups) for voice in eligible) / max(len(eligible), 1), 6
            ),
            "group_counts": dict(
                sorted(
                    Counter(group for voice in eligible for group in voice.signal_groups).items()
                )
            ),
            "feature_weight": 0.16,
        },
        "raw_topic_candidates": len(raw_topics),
        "deduplicated_topic_candidates": len(merged_topics),
        "selected": len(trends),
        "empty_output_slots": max(settings.max_trends - len(trends), 0),
        "selected_slots": selected_slots,
        "selection_audit": sorted(
            audit,
            key=lambda row: (not bool(row["selected"]), row["candidate_id"]),
        ),
        "statuses": {
            status: sum(trend.status == status for trend in trends)
            for status in ("new", "growing", "one-off", "fading", "not real")
        },
    }
    diagnostics["stage_counts"] = {
        "after_clustering": len(raw_topics),
        "after_mdl": sum(topic.mdl_gain_bits > 0 for topic in merged_topics),
        "after_deduplication": len(merged_topics),
        "after_status_decision": sum(
            topic.status_decision is not None and topic.status_decision.official_status is not None
            for topic in merged_topics
        ),
        "after_eligibility": len(qualified),
        "after_final_story_deduplication": len(labelled),
        "final_selected": len(trends),
    }
    LOGGER.info(
        "Pipeline diagnostics: loaded=%s eligible=%s raw=%s deduplicated=%s selected=%s "
        "statuses=%s",
        diagnostics["loaded"],
        diagnostics["eligible"],
        diagnostics["raw_topic_candidates"],
        diagnostics["deduplicated_topic_candidates"],
        diagnostics["selected"],
        diagnostics["statuses"],
    )
    return trends, diagnostics
