from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class InformationUnit:
    id: str
    source_text: str
    start: int
    end: int
    subject: str
    relation: str
    object: str
    theme: str
    concepts: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    modality: str = "neutral"
    polarity: str = "positive"
    quantities: tuple[str, ...] = ()
    time: str | None = None
    condition: str | None = None
    source_label: str | None = None
    risk_level: str = "normal"
    importance: float = 0.5

    @property
    def signature(self) -> tuple:
        return (
            self.theme,
            self.subject.lower(),
            self.relation,
            self.object.lower(),
            self.modality,
            self.polarity,
            self.quantities,
            self.concepts,
        )

    def to_trace(self) -> dict:
        return asdict(self)


@dataclass
class InformationCluster:
    theme: str
    units: list[InformationUnit] = field(default_factory=list)

    @property
    def start(self) -> int:
        return min(unit.start for unit in self.units)

    @property
    def end(self) -> int:
        return max(unit.end for unit in self.units)

    @property
    def source_span(self) -> str:
        return " ".join(unit.source_text for unit in self.units)

    @property
    def concepts(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for unit in self.units:
            for concept in unit.concepts:
                if concept not in seen:
                    seen.add(concept)
                    result.append(concept)
        return result
