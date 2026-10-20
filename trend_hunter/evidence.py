from __future__ import annotations

from collections import Counter
from math import ceil

import numpy as np

from .models import TopicMetrics, Voice
from .preprocess import canonical_for_duplicates
from .selection import PRODUCT_GROUPS, topic_mechanism_profile, voice_mechanisms

EXPLANATORY_MARKERS = (
    "因为",
    "所以",
    "导致",
    "结果",
    "问题是",
    "不是",
    "而是",
    "相比",
    "不如",
    "希望",
    "建议",
    "应该",
    "改成",
    "如果",
    "每次",
    "每天",
    "每周",
    "之后",
)
GENERIC_MARKERS = ("有人吗", "怎么说", "怎么看", "求问", "萌新问", "确实", "同意", "+1")


def select_evidence(
    topic: TopicMetrics,
    voices: list[Voice],
    vectors: np.ndarray,
    limit: int,
) -> list[int]:
    indices = topic.message_indices
    centroid = vectors[indices].mean(axis=0)
    centroid /= float(np.linalg.norm(centroid)) or 1.0
    similarities = {index: float(vectors[index] @ centroid) for index in indices}
    exact_counts = Counter(canonical_for_duplicates(voices[index].normalized) for index in indices)
    dominant_mechanism, mechanism_dominance, _, _ = topic_mechanism_profile(topic, voices)

    def base_score(index: int) -> float:
        voice = voices[index]
        specificity = min(len(voice.normalized) / 80.0, 1.0)
        duplicate_penalty = min(
            (exact_counts[canonical_for_duplicates(voice.normalized)] - 1) * 0.08, 0.40
        )
        low_reply_penalty = 0.35 if voice.is_low_context_reply else 0.0
        structured_signal = min(len(voice.signal_groups), 3) / 3.0
        comparison_bonus = 1.0 if "competitor_comparison" in voice.pragmatic_flags else 0.0
        mechanisms = voice_mechanisms(voice)
        mechanism_alignment = (
            1.0 if dominant_mechanism and dominant_mechanism in mechanisms else 0.0
        )
        mechanism_conflict = min(
            len(mechanisms - {dominant_mechanism}) * 0.25 if dominant_mechanism else 0.0,
            0.50,
        )
        concrete_product_signal = bool(set(voice.signal_groups) & PRODUCT_GROUPS)
        independent_context = 1.0 if voice.parent_id is None else 0.0
        # The first exported id is what a reviewer is most likely to read. Prefer a voice
        # that explains a trigger, consequence or desired change over a merely typical line.
        explanatory = min(
            sum(marker in voice.semantic_text for marker in EXPLANATORY_MARKERS) / 2.0,
            1.0,
        )
        generic = 1.0 if any(marker in voice.semantic_text for marker in GENERIC_MARKERS) else 0.0
        return (
            0.32 * similarities[index]
            + 0.23 * voice.quality
            + 0.15 * specificity
            + 0.08 * mechanism_alignment * max(mechanism_dominance, 0.5)
            + 0.04 * structured_signal
            + 0.03 * comparison_bonus
            + 0.05 * independent_context
            + 0.10 * explanatory
            + (0.03 if concrete_product_signal else -0.03)
            - duplicate_penalty
            - low_reply_penalty
            - 0.08 * generic
            - mechanism_conflict
        )

    ranked = sorted(indices, key=base_score, reverse=True)
    chosen: list[int] = []
    used_users: Counter[str] = Counter()
    used_platforms: Counter[str] = Counter()
    used_channels: Counter[str] = Counter()
    used_texts: set[str] = set()

    # Preserve the first clear signal, not merely the chronologically first short reply.
    early_count = min(len(indices), max(3, min(8, ceil(len(indices) / 5))))
    early_pool = sorted(indices, key=lambda index: voices[index].created_at)[:early_count]
    early = max(early_pool, key=base_score)
    chosen.append(early)
    used_users[voices[early].user_name] += 1
    used_platforms[voices[early].platform] += 1
    used_channels[voices[early].channel] += 1
    used_texts.add(canonical_for_duplicates(voices[early].normalized))

    while len(chosen) < min(limit, len(indices)):
        best_index = None
        best_value = float("-inf")
        for index in ranked:
            if index in chosen:
                continue
            voice = voices[index]
            text_key = canonical_for_duplicates(voice.normalized)
            if text_key in used_texts:
                continue
            diversity = 0.16 * (used_users[voice.user_name] == 0) + 0.12 * (
                used_platforms[voice.platform] == 0
            )
            diversity += 0.05 * (used_channels[voice.channel] == 0)
            temporal = 0.06 * all(
                voices[item].created_at.date() != voice.created_at.date() for item in chosen
            )
            redundancy = max(float(vectors[index] @ vectors[item]) for item in chosen)
            value = base_score(index) + diversity + temporal - 0.20 * max(redundancy - 0.82, 0)
            if value > best_value:
                best_index, best_value = index, value
        if best_index is None:
            break
        chosen.append(best_index)
        voice = voices[best_index]
        used_users[voice.user_name] += 1
        used_platforms[voice.platform] += 1
        used_channels[voice.channel] += 1
        used_texts.add(canonical_for_duplicates(voice.normalized))

    aligned = [
        index
        for index in chosen
        if dominant_mechanism and dominant_mechanism in voice_mechanisms(voices[index])
    ]
    if not aligned:
        aligned = [
            index
            for index in chosen
            if set(voices[index].signal_groups) & PRODUCT_GROUPS
            or "competitor_comparison" in voices[index].pragmatic_flags
        ]
    representative = max(aligned or chosen, key=base_score)
    remainder = sorted(
        (index for index in chosen if index != representative),
        key=lambda index: voices[index].created_at,
    )
    return [representative, *remainder]
