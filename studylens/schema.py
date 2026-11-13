"""The intermediate representation shared by every component: the Diagram Scene Graph (DSG).

Nodes  = parts / entities (labelled boxes, icons, or regions a label points to). ``bbox`` is the
         region to highlight; ``label_bbox`` is where the label text is (from OCR when available);
         ``region_source`` says how the region was obtained ("shape" = enclosing shape found by
         image analysis, "grounded" = located by the VLM, "label" = only the text box is known).
Edges  = directed relationships (arrows), source -> target.
Texts  = every OCR text item (ids t1..tn) with its role: label, title, caption or other.
Groups = containers that enclose several parts (e.g. "MICRO SERVICES").

Both extractors (Qwen2.5-VL and the ground-truth annotation fallback) produce a
``DiagramGraph``; the QA engine and the UI only ever consume a ``DiagramGraph``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx


@dataclass
class Node:
    id: str
    label: str
    bbox: list[float] | None = None  # [x1, y1, x2, y2] in image pixels: the region to highlight
    confidence: float | None = None
    label_bbox: list[float] | None = None  # where the label text is
    region_source: str | None = None  # "shape" | "grounded" | "label" | None (legacy)
    text_ids: list[str] = field(default_factory=list)  # OCR text items forming the label


@dataclass
class Edge:
    source: str  # node id
    target: str  # node id
    relation: str = "points_to"
    confidence: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.target)


@dataclass
class DiagramGraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    # Provenance: which backend produced this graph ("qwen2.5-vl-3b", "annotation", "user-edited", ...)
    source: str = "unknown"
    image_size: list[int] | None = None  # [width, height]
    warnings: list[str] = field(default_factory=list)
    texts: list[dict[str, Any]] = field(default_factory=list)  # {"id", "text", "box", "role"}
    groups: list[dict[str, Any]] = field(default_factory=list)  # {"id", "label", "members", "bbox"}
    title: str | None = None

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "source": self.source,
            "image_size": self.image_size,
            "warnings": list(self.warnings),
            "texts": list(self.texts),
            "groups": list(self.groups),
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiagramGraph":
        nodes = [
            Node(
                id=str(n["id"]),
                label=str(n.get("label", n["id"])),
                bbox=_clean_bbox(n.get("bbox")),
                confidence=n.get("confidence"),
                label_bbox=_clean_bbox(n.get("label_bbox")),
                region_source=n.get("region_source"),
                text_ids=[str(t) for t in n.get("text_ids", [])],
            )
            for n in d.get("nodes", [])
        ]
        edges = [
            Edge(
                source=str(e["source"]),
                target=str(e["target"]),
                relation=str(e.get("relation") or "points_to"),
                confidence=e.get("confidence"),
            )
            for e in d.get("edges", [])
        ]
        g = cls(
            nodes=nodes,
            edges=edges,
            source=str(d.get("source", "unknown")),
            image_size=d.get("image_size"),
            warnings=list(d.get("warnings", [])),
            texts=[t for t in d.get("texts", []) if isinstance(t, dict)][:500],
            groups=[x for x in d.get("groups", []) if isinstance(x, dict)][:100],
            title=d.get("title"),
        )
        g.validate()
        return g

    # ---- helpers -------------------------------------------------------
    def validate(self) -> None:
        """Drop edges whose endpoints are unknown and duplicate edges; record a warning for each."""
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate node ids in graph")
        seen: set[tuple[str, str]] = set()
        kept: list[Edge] = []
        for e in self.edges:
            if e.source not in ids or e.target not in ids:
                self.warnings.append(f"dropped edge {e.source}->{e.target}: unknown endpoint")
                continue
            if e.source == e.target:
                self.warnings.append(f"dropped self-loop on {e.source}")
                continue
            if e.key in seen:
                continue
            seen.add(e.key)
            kept.append(e)
        self.edges = kept

    def label_of(self, node_id: str) -> str:
        for n in self.nodes:
            if n.id == node_id:
                return n.label
        return node_id

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.id, label=n.label)
        for e in self.edges:
            g.add_edge(e.source, e.target, relation=e.relation)
        return g

    def edge_text(self, e: Edge) -> str:
        return f"{self.label_of(e.source)} -> {self.label_of(e.target)}"

    def as_text(self) -> str:
        """Compact textual form, used when giving the graph to a language model."""
        lines = ["Entities: " + ", ".join(n.label for n in self.nodes), "Arrows:"]
        lines += [f"- {self.edge_text(e)}" for e in self.edges] or ["- (none)"]
        return "\n".join(lines)


def _clean_bbox(b: Any) -> list[float] | None:
    if not b or not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        return [float(v) for v in b]
    except (TypeError, ValueError):
        return None
