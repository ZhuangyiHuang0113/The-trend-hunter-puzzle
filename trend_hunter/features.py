from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureResult:
    vectors: np.ndarray
    mode: str


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _append_weak_signals(
    vectors: np.ndarray,
    weak_signals: Sequence[Sequence[str]] | None,
) -> np.ndarray:
    """Add low-weight categorical hints without turning a lexicon into hard labels."""
    if not weak_signals:
        return vectors
    vocabulary = sorted({signal for row in weak_signals for signal in row})
    if not vocabulary:
        return vectors
    positions = {signal: index for index, signal in enumerate(vocabulary)}
    indicators = np.zeros((len(weak_signals), len(vocabulary)), dtype=np.float32)
    for row_index, row in enumerate(weak_signals):
        for signal in row:
            indicators[row_index, positions[signal]] = 0.16
    return _normalize_rows(np.hstack((vectors, indicators)))


def _cache_key(text: str, model: str, dimensions: int) -> str:
    raw = f"{model}\0{dimensions}\0{text}".encode()
    return hashlib.sha256(raw).hexdigest()


def _openai_embeddings(texts: list[str], settings: Settings) -> np.ndarray:
    from openai import OpenAI

    cache_path = settings.cache_dir / "embeddings.jsonl"
    cached: dict[str, list[float]] = {}
    try:
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    cached[row["key"]] = row["vector"]
    except Exception as exc:
        LOGGER.warning("Ignoring unreadable embedding cache: %s", exc)

    keys = [
        _cache_key(text, settings.embedding_model, settings.embedding_dimensions) for text in texts
    ]
    missing_positions = [index for index, key in enumerate(keys) if key not in cached]
    client = OpenAI(max_retries=3, timeout=60.0)
    new_rows: list[dict[str, object]] = []
    for start in range(0, len(missing_positions), 256):
        positions = missing_positions[start : start + 256]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=[texts[position][:6000] for position in positions],
            dimensions=settings.embedding_dimensions,
        )
        for position, item in zip(positions, response.data, strict=True):
            vector = list(item.embedding)
            cached[keys[position]] = vector
            new_rows.append({"key": keys[position], "vector": vector})
        LOGGER.info(
            "Embedded %s/%s uncached messages",
            min(start + 256, len(missing_positions)),
            len(missing_positions),
        )

    if new_rows:
        try:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as handle:
                for row in new_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            LOGGER.warning("Could not write embedding cache: %s", exc)
    return _normalize_rows(np.asarray([cached[key] for key in keys], dtype=np.float32))


def _jieba_tokens(text: str) -> list[str]:
    import jieba

    return [
        token.strip().lower()
        for token in jieba.lcut(text, cut_all=False)
        if len(token.strip()) >= 2 and not token.isspace()
    ]


def _offline_features(texts: list[str], settings: Settings) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    word = TfidfVectorizer(
        tokenizer=_jieba_tokens,
        token_pattern=None,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.72,
        sublinear_tf=True,
        max_features=settings.max_features // 2,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.72,
        sublinear_tf=True,
        max_features=settings.max_features // 2,
    )
    matrix = FeatureUnion([("word", word), ("char", char)]).fit_transform(texts)
    if matrix.shape[1] <= 2:
        return _normalize_rows(matrix.toarray().astype(np.float32))
    dimensions = min(settings.svd_dimensions, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if dimensions < 2:
        return _normalize_rows(matrix.toarray().astype(np.float32))
    reduced = TruncatedSVD(
        n_components=dimensions, random_state=settings.random_seed
    ).fit_transform(matrix)
    return _normalize_rows(reduced.astype(np.float32))


def build_features(
    texts: Iterable[str],
    settings: Settings,
    *,
    weak_signals: Iterable[Sequence[str]] | None = None,
) -> FeatureResult:
    materialized = list(texts)
    materialized_signals = list(weak_signals) if weak_signals is not None else None
    if materialized_signals is not None and len(materialized_signals) != len(materialized):
        raise ValueError("weak_signals must have one row per text")
    # Hybrid is intentionally local at the high-volume stage. This keeps cost and latency
    # proportional to the candidate shortlist instead of the 48k-message corpus.
    wants_openai = settings.semantic_mode == "openai"
    if wants_openai:
        try:
            vectors = _openai_embeddings(materialized, settings)
            return FeatureResult(
                _append_weak_signals(vectors, materialized_signals),
                "openai+weak-signals" if materialized_signals else "openai",
            )
        except Exception as exc:
            if settings.semantic_mode == "openai":
                raise
            LOGGER.warning("OpenAI embeddings unavailable; using offline features: %s", exc)
    vectors = _offline_features(materialized, settings)
    return FeatureResult(
        _append_weak_signals(vectors, materialized_signals),
        "offline+weak-signals" if materialized_signals else "offline",
    )
