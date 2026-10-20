from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

Status = Literal["new", "growing", "one-off", "fading", "not real"]
TemporalState = Literal["new", "growing", "one-off", "fading", "undetermined"]
AuthenticityState = Literal["organic", "suspicious", "undetermined"]


@dataclass(frozen=True, slots=True)
class EvidenceFeature:
    """One auditable signal, kept separate from the decision that consumes it."""

    raw: float | None
    normalized: float | None
    reliability: float
    missing: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "reliability": self.reliability,
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    topic_id: int
    independent_player_count: int
    topic_age_days: int
    active_days: int
    evidence_window: dict[str, str]
    features: dict[str, EvidenceFeature]
    missing_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "topic_id": self.topic_id,
            "independent_player_count": self.independent_player_count,
            "topic_age_days": self.topic_age_days,
            "active_days": self.active_days,
            "evidence_window": self.evidence_window,
            "missing_features": list(self.missing_features),
            "features": {name: feature.to_dict() for name, feature in self.features.items()},
        }


@dataclass(frozen=True, slots=True)
class StatusDecision:
    temporal_state: TemporalState
    authenticity_state: AuthenticityState
    official_status: Status | None
    evidence_scores: dict[str, float]
    evidence_reliability: float
    satisfied_conditions: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    uncertainties: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "temporal_state": self.temporal_state,
            "authenticity_state": self.authenticity_state,
            "official_status": self.official_status,
            "evidence_scores": self.evidence_scores,
            "evidence_reliability": self.evidence_reliability,
            "satisfied_conditions": list(self.satisfied_conditions),
            "counter_evidence": list(self.counter_evidence),
            "uncertainties": list(self.uncertainties),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Voice:
    id: str
    parent_id: str | None
    platform: str
    channel: str
    user_name: str
    title: str | None
    content: str
    created_at: datetime
    normalized: str = ""
    semantic_text: str = ""
    quality: float = 1.0
    is_low_context_reply: bool = False
    signal_groups: tuple[str, ...] = ()
    competitor_games: tuple[str, ...] = ()
    pragmatic_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    date: date
    version: str
    notes: str


@dataclass(slots=True)
class TopicMetrics:
    cluster_id: int
    message_indices: list[int]
    first_seen: date
    last_seen: date
    weighted_volume: float
    recent_volume: float
    previous_volume: float
    baseline_volume: float
    growth_ratio: float
    acceleration: float
    unique_users: int
    unique_platforms: int
    unique_channels: int
    active_days: int
    duplicate_ratio: float
    top_user_share: float
    top_platform_share: float
    burst_share: float
    peak_day: date
    peak_day_share: float
    post_peak_observed_days: int
    coherence: float
    authenticity: float
    novelty: float
    persistence: float
    spread: float
    community_breadth: float
    topic_rt_raw: float
    topic_rt: float
    rt_confidence: float
    cusum_baseline: float
    cusum_sigma: float
    cusum_score: float
    cusum_exceed_days: int
    cusum_alarm: bool
    status: Status
    score: float
    top_terms: list[str] = field(default_factory=list)
    nearby_events: list[CalendarEvent] = field(default_factory=list)
    event_expected_rate: float = 0.0
    event_observed_rate: float = 0.0
    event_post_rate: float = 0.0
    event_residual: float = 0.0
    event_persistence: float = 1.0
    survival_score: float = 1.0
    stress_tests: dict[str, float] = field(default_factory=dict)
    mdl_data_bits_saved: float = 0.0
    mdl_model_cost_bits: float = 0.0
    mdl_gain_bits: float = 0.0
    mdl_score: float = 0.0
    candidate_evidence: CandidateEvidence | None = None
    status_decision: StatusDecision | None = None
    priority_components: dict[str, float] = field(default_factory=dict)
    priority_score: float = 0.0
    priority_reliability: float = 0.0
    eligibility_passed: bool = False
    eligibility_reasons: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    merged_from: list[int] = field(default_factory=list)


@dataclass(slots=True)
class TopicCandidate:
    metrics: TopicMetrics
    evidence_indices: list[int]
    title: str
    title_en: str
    category: str
    why: str
    player_meaning: str = ""
    suggested_action: str = ""
    merge_key: str = ""
    keep: bool = True
    priority_components: dict[str, float] = field(default_factory=dict)
    priority_score: float = 0.0
    priority_reliability: float = 0.0


@dataclass(frozen=True, slots=True)
class Trend:
    title: str
    title_en: str
    category: str
    status: Status
    first_seen: str
    voice_ids: list[str]
    why: str
    representative_voice: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        if not self.voice_ids:
            raise ValueError("a trend must contain at least one voice_id")
        if self.representative_voice.get("voice_id") != self.voice_ids[0]:
            raise ValueError("representative_voice.voice_id must equal voice_ids[0]")
        return {
            "title": self.title,
            "title_en": self.title_en,
            "category": self.category,
            "status": self.status,
            "first_seen": self.first_seen,
            "voice_ids": self.voice_ids,
            "representative_voice": self.representative_voice,
            "why": self.why,
        }
