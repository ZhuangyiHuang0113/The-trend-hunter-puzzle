from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .config import Settings
from .models import TopicCandidate, TopicMetrics, Voice

PRIORITY_COMPONENTS = (
    "impact",
    "momentum",
    "breadth",
    "robustness",
    "novelty",
    "timeliness",
    "actionability",
)

HIGH_IMPACT_TERMS = {
    "无法登录",
    "無法登入",
    "登录不了",
    "登入不了",
    "掉线",
    "掉線",
    "回档",
    "回檔",
    "丢失",
    "消失",
    "误封",
    "誤封",
    "退款",
    "投诉",
    "投訴",
    "退坑",
    "退游",
    "棄坑",
    "卸载",
    "卸載",
    "外挂",
    "外掛",
    "骗氪",
    "騙氪",
    "暗改",
}

PRODUCT_GROUPS = {
    "gacha",
    "character_power",
    "balance",
    "grind",
    "build",
    "combat",
    "technical",
    "story",
    "art",
    "event",
    "retention",
    "fairness",
    "game_modes",
    "trust",
}

MECHANISM_TERMS = {
    "cheating_and_reports": {"外挂", "外掛", "锁头", "鎖頭", "开挂", "開掛", "举报", "舉報"},
    "gacha_rate_trust": {"暗改", "概率", "機率", "爆率", "公示", "出金怪怪"},
    "gacha_pull_experience": {
        "抽卡",
        "十连",
        "十連",
        "单抽",
        "單抽",
        "出金",
        "保底",
        "歪了",
        "没歪",
    },
    "gacha_pull_showcase": {
        "欧皇",
        "歐皇",
        "双金",
        "雙金",
        "三金",
        "抽卡记录",
        "抽卡記錄",
        "晒一下",
        "曬一下",
    },
    "gacha_pull_frustration": {"非酋", "沉船", "吃保底", "歪了", "一个金都", "一個金都", "井票"},
    "gacha_paywall": {"核心材料", "通行证", "通行證", "礼包", "禮包", "月卡", "逼氪"},
    "login_and_data_loss": {
        "无法登录",
        "無法登入",
        "登录一直",
        "登入一直",
        "排队",
        "排隊",
        "回档",
        "回檔",
    },
    "performance_and_crashes": {
        "卡顿",
        "卡頓",
        "闪退",
        "閃退",
        "掉帧",
        "掉幀",
        "发热",
        "發熱",
        "卡死",
    },
    "visual_glitches": {"贴图", "貼圖", "穿模", "模型错误", "模型錯誤", "黑屏", "白屏"},
    "loading_performance": {"加载", "加載", "高画质", "高畫質", "团战", "團戰", "ppt"},
    "character_balance": {
        "强度",
        "強度",
        "超模",
        "什么时候削",
        "什麼時候削",
        "老角色",
        "伤害",
        "傷害",
    },
    "pvp_meta_balance": {"PVP平衡", "pvp平衡", "竞技场环境", "競技場環境", "环境好单一"},
    "team_build_viability": {
        "配队",
        "配隊",
        "还能玩吗",
        "還能玩嗎",
        "深渊",
        "深淵",
        "哪个强",
        "哪個強",
    },
    "combat_controls": {"锁敌", "鎖敵", "镜头", "鏡頭", "手感", "闪避", "閃避", "弹反", "彈反"},
    "matchmaking": {"匹配", "单排", "單排", "车队", "車隊", "段位", "排不到人"},
    "competitor_content_gap": {"比这个有意思", "比介个有意思", "比這個有意思"},
    "daily_grind": {"坐牢", "上班", "日常", "周常", "刷材料", "清体力", "清體力"},
    "daily_chore": {"日常", "周常", "每日", "每周", "打卡", "上班"},
    "resource_farming": {"刷材料", "刷本", "刷词条", "刷詞條", "毕业", "畢業"},
    "stamina_friction": {"清体力", "清體力", "体力", "體力", "树脂", "掃蕩", "扫荡"},
    "event_value": {"活动奖励", "活動獎勵", "补偿", "補償", "周年", "福利"},
    "story_and_character": {"剧情", "劇情", "人设", "人設", "塑造", "文案"},
    "character_affinity": {"老婆", "老公", "角色厨", "角色廚", "本命", "我推"},
    "overall_art_direction": {"美术", "美術", "画风", "畫風", "场景", "場景", "光影"},
    "character_cosmetics": {"立绘", "立繪", "皮肤", "皮膚", "时装", "時裝", "衣装", "臉模", "脸模"},
    "character_visual_meme": {
        "铁锅",
        "鐵鍋",
        "大招动画",
        "大招動畫",
        "同人图",
        "同人圖",
        "脸模",
        "臉模",
    },
}


@dataclass(frozen=True, slots=True)
class ContextualMechanismRule:
    """Require a product anchor and a symptom/intent cue, with optional exclusions."""

    mechanism: str
    anchors: frozenset[str]
    cues: frozenset[str]
    excludes: frozenset[str] = frozenset()

    def matches(self, text: str) -> bool:
        return (
            any(term in text for term in self.anchors)
            and any(term in text for term in self.cues)
            and not any(term in text for term in self.excludes)
        )


CONTEXTUAL_MECHANISM_RULES = (
    ContextualMechanismRule(
        "login_and_data_loss",
        frozenset({"登录", "登入", "服务器", "伺服器", "账号", "帳號"}),
        frozenset(
            {
                "转圈",
                "轉圈",
                "登不上",
                "进不去",
                "進不去",
                "排队",
                "排隊",
                "掉线",
                "斷線",
                "回档",
                "回檔",
            }
        ),
    ),
    ContextualMechanismRule(
        "performance_and_crashes",
        frozenset(
            {
                "更新后",
                "更新後",
                "自动保存",
                "自動保存",
                "手机",
                "手機",
                "电脑",
                "電腦",
                "设备",
                "裝置",
            }
        ),
        frozenset({"闪退", "閃退", "卡死", "崩溃", "崩潰", "发热", "發熱", "掉帧", "掉幀"}),
    ),
    ContextualMechanismRule(
        "matchmaking",
        frozenset({"匹配", "排位", "单排", "單排"}),
        frozenset(
            {
                "等了",
                "分钟",
                "分鐘",
                "排不到",
                "高段位",
                "低分段",
                "大佬",
                "萌新被虐",
                "实力差",
                "實力差",
            }
        ),
    ),
    ContextualMechanismRule(
        "gacha_pull_frustration",
        frozenset({"抽", "卡池", "保底", "歪"}),
        frozenset(
            {"吃保底", "连续歪", "連續歪", "不想充", "不敢抽", "沉船", "一个金都", "一個金都"}
        ),
    ),
    ContextualMechanismRule(
        "gacha_rate_trust",
        frozenset({"概率", "機率", "爆率", "出金率", "卡池"}),
        frozenset({"暗改", "偷改", "没公示", "沒公示", "不透明", "怪怪", "变低", "變低"}),
    ),
    ContextualMechanismRule(
        "stamina_friction",
        frozenset({"体力", "體力", "树脂", "樹脂"}),
        frozenset({"不够", "不夠", "溢出", "清不完", "卡进度", "卡進度", "恢复慢", "恢復慢"}),
    ),
    ContextualMechanismRule(
        "resource_farming",
        frozenset({"材料", "词条", "詞條", "装备", "裝備", "刷本"}),
        frozenset({"刷不出", "随机", "隨機", "毕业", "畢業", "养不起", "養不起", "重复", "重複"}),
    ),
    ContextualMechanismRule(
        "pvp_meta_balance",
        frozenset({"pvp", "PVP", "竞技场", "競技場", "排位"}),
        frozenset({"环境", "環境", "阵容", "陣容", "十把九把", "必带", "必帶", "单一", "單一"}),
    ),
)

GROUP_IMPACT = {
    "technical": 0.90,
    "fairness": 0.90,
    "retention": 0.88,
    "trust": 0.84,
    "gacha": 0.78,
    "character_power": 0.72,
    "balance": 0.72,
    "grind": 0.66,
    "combat": 0.62,
    "build": 0.58,
    "event": 0.55,
    "game_modes": 0.52,
    "story": 0.45,
    "art": 0.40,
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def action_lenses(metrics: TopicMetrics, voices: list[Voice]) -> set[str]:
    counts: dict[str, set[str]] = defaultdict(set)
    for index in metrics.message_indices:
        voice = voices[index]
        for group in voice.signal_groups:
            if group in PRODUCT_GROUPS:
                counts[group].add(voice.user_name)
    threshold = max(2, math.ceil(metrics.unique_users * 0.12))
    return {group for group, users in counts.items() if len(users) >= threshold}


def voice_mechanisms(voice: Voice) -> set[str]:
    text = voice.semantic_text
    mechanisms = {
        mechanism
        for mechanism, terms in MECHANISM_TERMS.items()
        if any(term in text for term in terms)
    }
    contextual_matches = {
        rule.mechanism for rule in CONTEXTUAL_MECHANISM_RULES if rule.matches(text)
    }
    mechanisms.update(contextual_matches)
    if "competitor_comparison" in voice.pragmatic_flags:
        mechanisms.add("competitor_content_gap")

    # Resolve broad keyword collisions using the more specific contextual reading.
    if "login_and_data_loss" in contextual_matches and not any(
        term in text for term in ("闪退", "閃退", "卡死", "崩溃", "崩潰", "掉帧", "掉幀")
    ):
        mechanisms.discard("performance_and_crashes")
    if "performance_and_crashes" in contextual_matches and not any(
        term in text for term in ("登不上", "进不去", "進不去", "排队", "排隊", "回档", "回檔")
    ):
        mechanisms.discard("login_and_data_loss")
    if "gacha_pull_showcase" in mechanisms or "gacha_pull_frustration" in mechanisms:
        mechanisms.discard("gacha_pull_experience")
    if "gacha_rate_trust" in mechanisms or "gacha_paywall" in mechanisms:
        mechanisms.discard("gacha_pull_experience")
        mechanisms.discard("gacha_pull_showcase")
        mechanisms.discard("gacha_pull_frustration")
    if {"visual_glitches", "loading_performance"} & mechanisms:
        mechanisms.discard("performance_and_crashes")
    if {"daily_chore", "resource_farming", "stamina_friction"} & mechanisms:
        mechanisms.discard("daily_grind")
    if "character_visual_meme" in mechanisms:
        mechanisms.discard("team_build_viability")
    return mechanisms


def topic_mechanism_profile(
    metrics: TopicMetrics, voices: list[Voice]
) -> tuple[str, float, float, set[str]]:
    users_by_mechanism: dict[str, set[str]] = defaultdict(set)
    mixed_users: set[str] = set()
    all_users = {voices[index].user_name for index in metrics.message_indices}
    for index in metrics.message_indices:
        voice = voices[index]
        mechanisms = voice_mechanisms(voice)
        if len(mechanisms) > 1:
            mixed_users.add(voice.user_name)
        for mechanism in mechanisms:
            users_by_mechanism[mechanism].add(voice.user_name)
    ranked = sorted(users_by_mechanism.items(), key=lambda item: (-len(item[1]), item[0]))
    if not ranked:
        return "", 0.0, 0.0, set()
    dominant, users = ranked[0]
    total = max(len(all_users), 1)
    return (
        dominant,
        len(users) / total,
        len(mixed_users) / total,
        set(users_by_mechanism),
    )


def _independent_user_share(
    metrics: TopicMetrics,
    voices: list[Voice],
    predicate,
) -> float:
    all_users = {voices[index].user_name for index in metrics.message_indices}
    matching = {
        voices[index].user_name for index in metrics.message_indices if predicate(voices[index])
    }
    return len(matching) / max(len(all_users), 1)


def calculate_priority(
    metrics: TopicMetrics,
    voices: list[Voice],
    as_of: date,
    settings: Settings,
) -> tuple[dict[str, float], float, float]:
    evidence = metrics.candidate_evidence
    decision = metrics.status_decision
    if evidence is None or decision is None:
        return {}, 0.0, 0.0

    lenses = action_lenses(metrics, voices)
    severity_prior = max((GROUP_IMPACT.get(group, 0.35) for group in lenses), default=0.35)
    affected_share = _independent_user_share(
        metrics,
        voices,
        lambda voice: any(term in voice.semantic_text for term in HIGH_IMPACT_TERMS),
    )
    broad_severity = _clip(affected_share / 0.35)
    impact = _clip(0.58 * severity_prior + 0.42 * broad_severity)

    f = evidence.features
    rt = float(f["organic_rt"].normalized or 0.0)
    cusum = float(f["cusum_persistence"].normalized or 0.0)
    event = (
        float(f["event_adjusted_persistence"].normalized or 0.0)
        if not f["event_adjusted_persistence"].missing
        else 0.0
    )
    # Rt and CUSUM share some underlying activity. The max carries the main momentum signal;
    # the min can only add a small corroboration bonus rather than double-counting growth.
    momentum = _clip(0.62 * max(rt, cusum) + 0.18 * min(rt, cusum) + 0.20 * event)
    if decision.temporal_state == "fading":
        momentum = _clip(0.55 * decision.evidence_scores["fading"])
    elif decision.temporal_state == "one-off":
        momentum = _clip(0.45 * decision.evidence_scores["one-off"])

    breadth = _clip(
        metrics.community_breadth
        * (1.0 - 0.35 * metrics.duplicate_ratio)
        * (1.0 - 0.20 * metrics.top_platform_share)
    )
    robustness = _clip(metrics.survival_score)
    novelty = _clip(metrics.mdl_score)
    days_since_last = max((as_of - metrics.last_seen).days, 0)
    timeliness = _clip(math.exp(-days_since_last / 12.0))
    concrete_share = _independent_user_share(
        metrics,
        voices,
        lambda voice: bool(set(voice.signal_groups) & PRODUCT_GROUPS),
    )
    actionability = _clip(0.35 * min(len(lenses) / 2.0, 1.0) + 0.65 * concrete_share)

    components = {
        "impact": impact,
        "momentum": momentum,
        "breadth": breadth,
        "robustness": robustness,
        "novelty": novelty,
        "timeliness": timeliness,
        "actionability": actionability,
    }
    if len(settings.priority_weights) != len(PRIORITY_COMPONENTS) or not math.isclose(
        sum(settings.priority_weights), 1.0, abs_tol=1e-9
    ):
        raise ValueError("Priority weights must match their components and sum to 1")
    weights = dict(zip(PRIORITY_COMPONENTS, settings.priority_weights, strict=True))
    base = sum(weights[name] * value for name, value in components.items())

    feature_reliability = _clip(
        0.55 * decision.evidence_reliability
        + 0.25 * min(math.log1p(metrics.unique_users) / math.log(21.0), 1.0)
        + 0.20 * min(metrics.active_days / 8.0, 1.0)
    )
    authenticity_risk = decision.evidence_scores["not real"]
    authenticity_adjustment = (
        1.0 if decision.authenticity_state == "suspicious" else 1.0 - 0.35 * authenticity_risk
    )
    reliability_adjustment = 0.62 + 0.38 * feature_reliability
    priority = _clip(base * reliability_adjustment * authenticity_adjustment)

    if decision.authenticity_state == "suspicious":
        # Coordinated propagation is a separate studio risk. Rank it by the strength and reach
        # of the anomaly, not by pretending it is an organic player trend.
        priority = _clip(
            (0.50 * authenticity_risk + 0.25 * breadth + 0.15 * timeliness + 0.10 * impact)
            * reliability_adjustment
        )

    rounded = {name: round(value, 6) for name, value in components.items()}
    metrics.priority_components = rounded
    metrics.priority_score = round(priority, 6)
    metrics.priority_reliability = round(feature_reliability, 6)
    return rounded, metrics.priority_score, metrics.priority_reliability


def evaluate_eligibility(
    metrics: TopicMetrics,
    voices: list[Voice],
    settings: Settings,
) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    decision = metrics.status_decision
    if decision is None or decision.official_status is None:
        return False, reasons, "undetermined_status"
    if metrics.mdl_gain_bits <= 0 and decision.authenticity_state != "suspicious":
        return False, reasons, "mdl_did_not_support_independent_topic"
    if metrics.unique_users < settings.min_topic_users:
        return False, reasons, "too_few_independent_players"
    if metrics.active_days < 2 and decision.official_status not in {"new", "not real"}:
        return False, reasons, "insufficient_time_evidence"
    invalid_indices = any(index < 0 or index >= len(voices) for index in metrics.message_indices)
    if not metrics.message_indices or invalid_indices:
        return False, reasons, "untraceable_voice_ids"
    if metrics.priority_reliability < 0.22:
        return False, reasons, "low_evidence_reliability"
    if metrics.priority_score < settings.min_priority_score:
        return False, reasons, "below_priority_floor"
    _, dominance, mixed_share, mechanisms = topic_mechanism_profile(metrics, voices)
    if not mechanisms:
        return False, reasons, "no_specific_product_mechanism"
    event_new = decision.official_status == "new" and bool(metrics.nearby_events)
    mixed_limit = 0.55 if event_new else 0.28
    if len(mechanisms) >= 3 and (dominance < 0.62 or mixed_share >= mixed_limit):
        return False, reasons, "mixed_product_mechanisms"
    if len(mechanisms) >= 2 and dominance < 0.55 and mixed_share >= 0.25:
        return False, reasons, "mixed_product_mechanisms"
    reasons.extend(
        [
            "traceable_voice_ids",
            "independent_topic_or_actionable_authenticity_risk",
            "status_has_sufficient_evidence",
        ]
    )
    return True, reasons, ""


def score_and_qualify_topics(
    topics: list[TopicMetrics],
    voices: list[Voice],
    as_of: date,
    settings: Settings,
) -> None:
    for topic in topics:
        calculate_priority(topic, voices, as_of, settings)
        passed, reasons, rejection = evaluate_eligibility(topic, voices, settings)
        topic.eligibility_passed = passed
        topic.eligibility_reasons = reasons
        topic.rejection_reason = rejection


def select_top_candidates(
    candidates: list[TopicCandidate],
    limit: int,
) -> tuple[list[TopicCandidate], dict[int, str]]:
    ordered = sorted(
        (candidate for candidate in candidates if candidate.keep),
        key=lambda candidate: (
            -candidate.priority_score,
            -candidate.priority_reliability,
            candidate.metrics.cluster_id,
        ),
    )
    selected: list[TopicCandidate] = []
    rejected: dict[int, str] = {}
    risk_lane = [candidate for candidate in ordered if candidate.metrics.status == "not real"][:2]
    emerging_lane = [
        candidate
        for candidate in ordered
        if candidate.metrics.status in {"new", "growing", "one-off"}
    ]
    fading_lane = [candidate for candidate in ordered if candidate.metrics.status == "fading"]
    selected.extend(risk_lane)
    selected.extend(emerging_lane[: max(limit - len(selected), 0)])
    remaining = max(limit - len(selected), 0)
    selected.extend(fading_lane[: min(3, remaining)])
    selected = sorted(
        selected,
        key=lambda candidate: (
            -candidate.priority_score,
            -candidate.priority_reliability,
            candidate.metrics.cluster_id,
        ),
    )
    selected_ids = {candidate.metrics.cluster_id for candidate in selected}
    risk_ids = {candidate.metrics.cluster_id for candidate in risk_lane}
    for candidate in ordered:
        if candidate.metrics.cluster_id in selected_ids:
            continue
        rejected[candidate.metrics.cluster_id] = (
            "suspicious_signal_lane_full"
            if candidate.metrics.status == "not real"
            and candidate.metrics.cluster_id not in risk_ids
            else "fading_history_lane_full"
            if candidate.metrics.status == "fading"
            else "below_top_k_cutoff"
        )
    return selected, rejected
