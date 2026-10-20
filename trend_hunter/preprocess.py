from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .io import replace_voice
from .lexicons import SignalProfile, analyse_player_signals
from .models import Voice

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[^\s,:：，。！？!?]{1,30}")
IMAGE_RE = re.compile(r"<image:[^>]*>|\[图片\]|\[圖片\]", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}")
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

LOW_CONTEXT_REPLIES = {
    "+1",
    "1",
    "确实",
    "確實",
    "同意",
    "顶",
    "頂",
    "是的",
    "真的",
    "笑死",
    "草",
    "哈哈",
    "666",
    "点了",
    "支持",
    "zc",
    "？",
    "?",
}

NOISE_PHRASES = {
    "签到",
    "簽到",
    "打卡",
    "早",
    "晚安",
    "沙发",
    "沙發",
    "前排",
    "已三连",
    "已三連",
    "求个好友",
    "求個好友",
    "有人一起玩吗",
    "有人一起玩嗎",
    "日常水一贴",
    "日常水一貼",
    "第一次发帖多多关照",
    "第一次發帖多多關照",
}

NON_PRODUCT_SPAM_PHRASES = {
    "破解版",
    "求链接",
    "求连结",
    "出号",
    "带价私聊",
    "代练",
    "代肝",
}


def _converter():
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:
        return None


def normalize_text(text: str, converter: object | None = None) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = URL_RE.sub(" <URL> ", text)
    text = MENTION_RE.sub(" <MENTION> ", text)
    text = IMAGE_RE.sub(" <IMAGE> ", text)
    text = text.replace("<emoji:", "<EMOJI:")
    text = REPEATED_CHAR_RE.sub(lambda match: match.group(1) * 3, text)
    text = SPACE_RE.sub(" ", text).strip()
    if converter is not None:
        try:
            text = converter.convert(text)
        except Exception:
            pass
    return text


def canonical_for_duplicates(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<(?:mention|url|image|emoji:[^>]*)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\u3400-\u9fff]+", "", text)
    return text


def quality_score(
    text: str,
    *,
    has_parent: bool,
    title: str | None,
    profile: SignalProfile | None = None,
) -> float:
    stripped = text.strip()
    profile = profile or analyse_player_signals(stripped)
    duplicate_key = canonical_for_duplicates(stripped)
    if not duplicate_key or PUNCT_RE.match(stripped):
        return 0.0
    if any(phrase in stripped for phrase in NON_PRODUCT_SPAM_PHRASES):
        return 0.02
    if duplicate_key in {canonical_for_duplicates(item) for item in NOISE_PHRASES}:
        return 0.06

    chinese = len(re.findall(r"[\u3400-\u9fff]", stripped))
    unique_chinese = set(re.findall(r"[\u3400-\u9fff]", stripped))
    alphanumeric = len(re.findall(r"[A-Za-z0-9\u3400-\u9fff]", stripped))
    if len(unique_chinese) <= 2 and alphanumeric <= 10 and not profile.groups:
        return 0.06
    length_signal = min(1.0, math.log1p(alphanumeric) / math.log(45))
    # Several independent player dimensions are more informative than a single keyword hit,
    # but remain a small bonus so the lexicon cannot decide topic membership by itself.
    signal_breadth = min(len(profile.groups), 3) / 3.0
    lexical_signal = 1.0 if profile.groups else 0.0
    sentence_signal = (
        1.0 if re.search(r"[，。！？!?;；]|因为|但是|怎么|為什麼|觉得|感覺", stripped) else 0.0
    )
    low_context = stripped.lower() in LOW_CONTEXT_REPLIES

    score = (
        0.12
        + 0.48 * length_signal
        + 0.10 * lexical_signal
        + 0.05 * signal_breadth
        + 0.12 * sentence_signal
    )
    score += 0.08 if title else 0.0
    score += 0.08 if has_parent and low_context else 0.0
    comparison = "competitor_comparison" in profile.flags
    useful_competitor_context = bool(profile.groups) or comparison
    score += 0.06 if profile.competitors and useful_competitor_context else 0.0
    score += 0.02 if "possible_irony" in profile.flags else 0.0
    score -= 0.28 if low_context and not has_parent else 0.0
    score -= 0.12 if chinese == 0 and alphanumeric < 10 else 0.0
    return max(0.0, min(1.0, score))


def preprocess_voices(voices: list[Voice]) -> list[Voice]:
    converter = _converter()
    result: list[Voice] = []
    for voice in voices:
        normalized = normalize_text(voice.content, converter)
        title = normalize_text(voice.title or "", converter) or None
        profile = analyse_player_signals(" ".join(item for item in (title, normalized) if item))
        quality = quality_score(
            normalized,
            has_parent=bool(voice.parent_id),
            title=title,
            profile=profile,
        )
        result.append(
            replace_voice(
                voice,
                title=title,
                normalized=normalized,
                semantic_text=normalized,
                quality=quality,
                is_low_context_reply=normalized.lower() in LOW_CONTEXT_REPLIES,
                signal_groups=profile.groups,
                competitor_games=profile.competitors,
                pragmatic_flags=profile.flags,
            )
        )
    return result


def duplicate_frequencies(voices: list[Voice]) -> Counter[str]:
    return Counter(
        canonical_for_duplicates(voice.normalized) for voice in voices if voice.normalized
    )
