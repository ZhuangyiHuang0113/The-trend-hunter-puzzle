from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    timezone: str = "Asia/Hong_Kong"
    max_trends: int = 8
    random_seed: int = 42
    min_quality: float = 0.28
    min_topic_messages: int = 5
    min_topic_users: int = 3
    max_evidence: int = 7
    recent_days: int = 4
    previous_days: int = 7
    baseline_days: int = 21
    max_features: int = 18_000
    svd_dimensions: int = 96
    target_cluster_size: int = 100
    min_clusters: int = 24
    max_clusters: int = 280
    semantic_mode: str = "hybrid"
    openai_model: str = "gpt-6-astra"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 256
    typesafe_model: str = "jev-latest"
    typesafe_api_url: str = "https://api.typesafe.ai/v1/systemone"
    semantic_confidence: float = 0.65
    llm_candidates: int = 32
    min_status_score: float = 0.65
    min_status_margin: float = 0.15
    min_status_reliability: float = 0.40
    min_priority_score: float = 0.16
    growing_weights: tuple[float, float, float, float] = (0.40, 0.35, 0.15, 0.10)
    fading_weights: tuple[float, float, float] = (0.40, 0.35, 0.25)
    one_off_weights: tuple[float, float, float, float] = (0.30, 0.30, 0.20, 0.20)
    authenticity_weights: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
    priority_weights: tuple[float, float, float, float, float, float, float] = (
        0.25,
        0.20,
        0.15,
        0.15,
        0.10,
        0.10,
        0.05,
    )
    cache_dir: Path = Path(".trend_cache")

    @classmethod
    def from_env(cls, *, semantic_mode: str | None = None) -> Settings:
        return cls(
            semantic_mode=semantic_mode or os.getenv("TREND_SEMANTIC_MODE", "hybrid"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-6-astra"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            typesafe_model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
            typesafe_api_url=os.getenv("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"),
            cache_dir=Path(os.getenv("TREND_CACHE_DIR", ".trend_cache")),
        )

    @property
    def has_openai_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    @property
    def has_typesafe_key(self) -> bool:
        return bool(os.getenv("TYPESAFE_API_KEY"))
