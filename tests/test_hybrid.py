from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trend_hunter.config import Settings
from trend_hunter.features import build_features
from trend_hunter.semantic import _answer_choice, _jev_questions


def test_hybrid_keeps_high_volume_feature_stage_local(monkeypatch):
    def fail_if_called(_texts, _settings):
        raise AssertionError("hybrid mode must not embed the full corpus through OpenAI")

    monkeypatch.setattr("trend_hunter.features._openai_embeddings", fail_if_called)
    monkeypatch.setattr(
        "trend_hunter.features._offline_features",
        lambda texts, _settings: np.ones((len(texts), 2), dtype=np.float32),
    )
    result = build_features(["抽卡概率", "服务器掉线"], Settings(semantic_mode="hybrid"))
    assert result.mode == "offline"
    assert result.vectors.shape == (2, 2)


def test_settings_detect_typesafe_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    assert Settings().has_typesafe_key


def test_jev_contract_uses_bounded_choice_questions():
    candidate = SimpleNamespace(metrics=SimpleNamespace(cluster_id=17))
    questions = _jev_questions([candidate])
    assert set(questions) == {
        "cpos17_coherence",
        "cpos17_actionability",
        "cpos17_lens",
    }
    assert all(question["type"] == "choice" for question in questions.values())
    assert "coordinated_manipulation" in questions["cpos17_actionability"]["criteria"]


def test_jev_answer_parser_is_conservative_on_missing_values():
    assert _answer_choice({}, "missing") == ("", 0.0)
    assert _answer_choice({"x": {"choice": "coherent", "confidence": 0.81}}, "x") == (
        "coherent",
        0.81,
    )
