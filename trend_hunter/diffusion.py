from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .models import TopicMetrics, Voice
from .preprocess import canonical_for_duplicates


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class DiffusionEstimate:
    raw_rt: float
    organic_rt: float
    confidence: float
    community_breadth: float


@dataclass(frozen=True, slots=True)
class MDLAdmission:
    data_bits_saved: float
    model_cost_bits: float
    gain_bits: float
    score: float


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.prod(values) ** (1.0 / len(values))


def estimate_topic_diffusion(
    indices: list[int],
    voices: list[Voice],
    as_of: date,
    *,
    generation_days: int = 3,
) -> DiffusionEstimate:
    """Estimate whether one generation of participants is followed by another.

    This is an observational proxy, not a causal infection estimate: no exposure or share
    data exists. Exact-copy and low-context activity is discounted in the organic estimate.
    """
    selected = [voices[index] for index in indices]
    if not selected:
        return DiffusionEstimate(0.0, 0.0, 0.0, 0.0)

    current_start = as_of - timedelta(days=generation_days - 1)
    previous_start = current_start - timedelta(days=generation_days)
    previous_end = current_start - timedelta(days=1)

    first_by_user: dict[str, Voice] = {}
    for voice in sorted(selected, key=lambda item: (item.created_at, item.id)):
        first_by_user.setdefault(voice.user_name, voice)

    previous_active = {
        voice.user_name
        for voice in selected
        if previous_start <= voice.created_at.date() <= previous_end
    }
    current_new = {
        user
        for user, voice in first_by_user.items()
        if current_start <= voice.created_at.date() <= as_of
    }
    raw_rt = (len(current_new) + 1.5) / (len(previous_active) + 1.5)

    exact_counts = Counter(canonical_for_duplicates(voice.normalized) for voice in selected)

    def organic_weight(voice: Voice) -> float:
        duplicate_key = canonical_for_duplicates(voice.normalized)
        duplicate_discount = 1.0 / math.sqrt(max(exact_counts[duplicate_key], 1))
        reply_discount = 0.55 if voice.is_low_context_reply else 1.0
        return max(voice.quality, 0.05) * duplicate_discount * reply_discount

    new_weight = sum(organic_weight(first_by_user[user]) for user in current_new)
    previous_weight = sum(
        max(
            organic_weight(voice)
            for voice in selected
            if voice.user_name == user and previous_start <= voice.created_at.date() <= previous_end
        )
        for user in previous_active
    )
    organic_rt = (new_weight + 1.5) / (previous_weight + 1.5)

    unique_users = len({voice.user_name for voice in selected})
    unique_platforms = len({voice.platform for voice in selected})
    unique_channels = len({voice.channel for voice in selected})
    active_days = len({voice.created_at.date() for voice in selected})
    community_breadth = _geometric_mean(
        [
            _clip(unique_users / 12.0),
            _clip(unique_platforms / 3.0),
            _clip(unique_channels / 5.0),
            _clip(active_days / 7.0),
        ]
    )

    observed_generations = len(
        {max(0, (as_of - voice.created_at.date()).days // generation_days) for voice in selected}
    )
    sample_confidence = _clip((len(current_new) + len(previous_active)) / 16.0)
    history_confidence = _clip(observed_generations / 3.0)
    confidence = _clip(
        0.52 * sample_confidence + 0.23 * history_confidence + 0.25 * community_breadth
    )
    return DiffusionEstimate(raw_rt, organic_rt, confidence, community_breadth)


def calculate_mdl_admission(
    indices: list[int],
    voices: list[Voice],
    vectors: np.ndarray,
    *,
    defining_terms: list[str] | None = None,
) -> MDLAdmission:
    """Approximate two-part MDL gain for inventing one reusable topic code.

    The null model encodes each distinct message against the conversation background. The
    topic model pays to describe a centroid and a short label, then encodes residuals around
    that centroid. Exact-copy campaigns get only one representative and cannot manufacture
    compression gain through repetition.
    """
    if not indices or vectors.size == 0:
        return MDLAdmission(0.0, 0.0, 0.0, 0.0)

    representative_indices: list[int] = []
    seen_texts: set[str] = set()
    for index in sorted(indices, key=lambda item: voices[item].created_at):
        key = canonical_for_duplicates(voices[index].normalized)
        if not key or key in seen_texts:
            continue
        seen_texts.add(key)
        representative_indices.append(index)
    if len(representative_indices) < 3:
        model_cost = math.log2(max(len(voices), 2))
        return MDLAdmission(0.0, model_cost, -model_cost, 0.0)

    sample = vectors[representative_indices].astype(np.float64)
    background_center = vectors.mean(axis=0).astype(np.float64)
    topic_center = sample.mean(axis=0)
    null_variance = float(np.mean(np.sum((sample - background_center) ** 2, axis=1)))
    topic_variance = float(np.mean(np.sum((sample - topic_center) ** 2, axis=1)))
    effective_dimensions = min(12, vectors.shape[1])
    ratio = max((null_variance + 1e-5) / (topic_variance + 1e-5), 1.0)
    data_bits = 0.5 * len(sample) * effective_dimensions * math.log2(ratio)

    label_terms = max(1, min(len(defining_terms or ()), 5))
    model_bits = (
        0.5 * effective_dimensions * math.log2(len(sample) + 1)
        + label_terms * math.log2(len(sample) + 1)
        + math.log2(max(len(voices) / len(sample), 1.0) + 1.0)
    )
    gain = data_bits - model_bits
    score = _clip(0.5 + 0.5 * math.tanh(gain / max(model_bits, 1.0)))
    return MDLAdmission(data_bits, model_bits, gain, score)


def enrich_mdl_admission(
    topics: list[TopicMetrics],
    voices: list[Voice],
    vectors: np.ndarray,
) -> None:
    for topic in topics:
        admission = calculate_mdl_admission(
            topic.message_indices,
            voices,
            vectors,
            defining_terms=topic.top_terms,
        )
        topic.mdl_data_bits_saved = admission.data_bits_saved
        topic.mdl_model_cost_bits = admission.model_cost_bits
        topic.mdl_gain_bits = admission.gain_bits
        topic.mdl_score = admission.score
        # MDL is an admission argument. It does not decide temporal status or studio priority.
