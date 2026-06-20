from __future__ import annotations

import re
from .normalize import split_sentences, normalize_space


def text_dedupe_backend(text: str) -> tuple[str, list[dict]]:
    traces: list[dict] = []
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text, traces

    seen: set[str] = set()
    kept: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence.lower()).strip()
        if key in seen and not sentence.startswith("__PROTECTED_"):
            traces.append({
                "backend": "text_dedupe_backend",
                "reason": "exact repeated sentence removed",
                "removed": sentence,
            })
            continue
        seen.add(key)
        kept.append(sentence)
    return normalize_space(" ".join(kept)), traces
