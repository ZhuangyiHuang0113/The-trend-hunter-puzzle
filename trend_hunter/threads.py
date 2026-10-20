from __future__ import annotations

from .io import replace_voice
from .models import Voice


def attach_thread_context(voices: list[Voice], max_parent_chars: int = 240) -> list[Voice]:
    """Attach parent meaning without duplicating the parent as a new observation."""
    by_id = {voice.id: voice for voice in voices}
    enriched: list[Voice] = []
    for voice in voices:
        pieces: list[str] = []
        if voice.title:
            pieces.append(f"帖子标题: {voice.title}")
        parent = by_id.get(voice.parent_id) if voice.parent_id else None
        if parent and parent.normalized:
            pieces.append(f"上文: {parent.normalized[:max_parent_chars]}")
        if voice.normalized:
            pieces.append(f"玩家: {voice.normalized}")
        semantic_text = "\n".join(pieces).strip()
        # Short replies such as “确实 / 点了 / ZC” inherit what they endorse, while their own
        # voice id and low-context penalty remain intact.
        signal_groups = tuple(
            dict.fromkeys((*((parent.signal_groups) if parent else ()), *voice.signal_groups))
        )
        competitor_games = tuple(
            dict.fromkeys((*((parent.competitor_games) if parent else ()), *voice.competitor_games))
        )
        pragmatic_flags = tuple(
            dict.fromkeys((*((parent.pragmatic_flags) if parent else ()), *voice.pragmatic_flags))
        )
        enriched.append(
            replace_voice(
                voice,
                semantic_text=semantic_text,
                signal_groups=signal_groups,
                competitor_games=competitor_games,
                pragmatic_flags=pragmatic_flags,
            )
        )
    return enriched
