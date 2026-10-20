from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date

import numpy as np

from .config import Settings
from .models import CalendarEvent, TopicMetrics, Voice
from .signals import calculate_topic_metrics

STOPWORDS = {
    "一个",
    "这个",
    "那个",
    "什么",
    "怎么",
    "可以",
    "就是",
    "不是",
    "真的",
    "感觉",
    "觉得",
    "还是",
    "已经",
    "没有",
    "玩家",
    "上文",
    "帖子",
    "标题",
    "回复",
    "游戏",
    "星环守望",
    "mention",
    "emoji",
    "image",
    "详情",
    "见图",
    "哈哈",
    "哈哈哈",
    "啊啊啊",
    "今天",
    "一天",
    "一下",
    "快乐",
    "离谱",
    "说真的",
    "不住",
    "萌新",
    "萌新问",
    "藏狐",
    "发错",
    "orz",
    "qaq",
    "hhhh",
}


def _calendar_terms(event: CalendarEvent) -> set[str]:
    terms = set(re.findall(r"「([^」]+)」", event.notes))
    lowered = event.notes.lower()
    if "anniversary" in lowered or "周年" in event.notes:
        terms.update({"周年", "活动", "活動", "奖励", "獎勵", "福利", "礼包", "禮包"})
    if "maintenance" in lowered or "维护" in event.notes or "維護" in event.notes:
        terms.update({"维护", "維護", "停服", "炸服", "排队", "排隊", "登录", "登入"})
    return terms


def tokenize_for_topics(text: str) -> list[str]:
    import jieba

    tokens = []
    for token in jieba.lcut(text.lower(), cut_all=False):
        token = token.strip(" \t\r\n,，。.!！?？:：;；()（）[]【】<>《》\"'")
        if len(token) < 2 or token in STOPWORDS or token.isdigit():
            continue
        tokens.append(token)
    return tokens


def top_terms_for_indices(
    indices: list[int], voices: list[Voice], global_df: Counter[str]
) -> list[str]:
    local = Counter(
        token
        for index in indices
        for token in set(tokenize_for_topics(voices[index].semantic_text))
    )
    scored = []
    for term, count in local.items():
        inverse = math.log(2.0 + len(voices) / (1.0 + global_df[term]))
        scored.append((count * inverse, term))
    return [term for _, term in sorted(scored, reverse=True)[:8]]


def cluster_topics(
    voices: list[Voice],
    vectors: np.ndarray,
    as_of: date,
    events: list[CalendarEvent],
    settings: Settings,
) -> list[TopicMetrics]:
    from sklearn.cluster import MiniBatchKMeans

    n_samples = len(voices)
    if n_samples < settings.min_topic_messages:
        return []
    n_clusters = round(n_samples / settings.target_cluster_size)
    n_clusters = max(settings.min_clusters, min(settings.max_clusters, n_clusters))
    n_clusters = min(n_clusters, max(2, n_samples // settings.min_topic_messages))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=settings.random_seed,
        batch_size=min(2048, max(256, n_samples // 8)),
        n_init=5,
        max_iter=160,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(vectors)
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)

    traffic: defaultdict[date, float] = defaultdict(float)
    for voice in voices:
        traffic[voice.created_at.date()] += max(voice.quality, 0.1)
    global_df = Counter(
        token for voice in voices for token in set(tokenize_for_topics(voice.semantic_text))
    )

    topics: list[TopicMetrics] = []
    for cluster_id, indices in groups.items():
        unique_users = len({voices[index].user_name for index in indices})
        if len(indices) < settings.min_topic_messages or unique_users < settings.min_topic_users:
            continue
        metrics = calculate_topic_metrics(
            cluster_id,
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
                # This only adds known calendar context. Event residual remains unobserved;
                # the decision layer therefore keeps reduced reliability.
                metrics.nearby_events = [contextual_events[-1]]
        if metrics.coherence < 0.18:
            continue
        average_length = sum(len(voices[index].normalized) for index in indices) / len(indices)
        if average_length < 4:
            continue
        topics.append(metrics)
    return sorted(topics, key=lambda topic: topic.score, reverse=True)
