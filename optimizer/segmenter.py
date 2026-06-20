from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class Segment:
    text: str
    start: int
    end: int
    kind: str = "natural_language"


class Segmenter:
    def split(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        pos = 0
        for match in re.finditer(r"__PROTECTED_\d+__", text):
            if match.start() > pos:
                segments.append(Segment(text[pos:match.start()], pos, match.start()))
            segments.append(Segment(match.group(0), match.start(), match.end(), "protected"))
            pos = match.end()
        if pos < len(text):
            segments.append(Segment(text[pos:], pos, len(text)))
        return [segment for segment in segments if segment.text]
