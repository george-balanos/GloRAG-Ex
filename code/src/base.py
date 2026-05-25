from dataclasses import dataclass, field
from typing import Any

@dataclass
class Entity:
    name: str
    type: str = ""
    description: str = ""
    rank: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    src: str
    tgt: str
    keywords: str = ""
    description: str = ""
    weight: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class Subgraph:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    raw_context: str = ""