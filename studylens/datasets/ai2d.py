"""AI2D (Kembhavi et al., ECCV 2016) loader: gold diagram graphs + human-written questions.

The dataset is NOT redistributed with this repository (AI2 non-commercial licence: no
redistribution, no public release of modified data). Download it with
``python scripts/get_ai2d.py`` into ``external/ai2d/`` (git-ignored).

Gold graph construction from the annotations
--------------------------------------------
* Nodes  = text elements (``text`` entries): label = ``value`` (stripped), bbox = ``rectangle``.
  Labels that are identical after normalisation are merged into one node.
* Each image blob (``B*``) is mapped to a text label via ``intraObjectLabel`` /
  ``intraObjectRegionLabel`` relationships. When a blob has several label texts (annotation
  noise, e.g. a column header linked to a blob), the label whose rectangle centre is nearest to
  the blob centroid is used.
* Edges  = ``interObjectLinkage`` relationships with ``hasDirectionality == True``:
  origin -> destination (tail -> arrowhead; verified visually on food-web diagrams).
  Endpoints that are unlabelled blobs are dropped (counted in ``dropped_edges``).
  Undirected linkages are not used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ROOT
from ..schema import DiagramGraph, Edge, Node

AI2D_ROOT = ROOT / "external" / "ai2d"
AI2D_DIR = AI2D_ROOT / "ai2d"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


@dataclass
class Question:
    qid: str
    text: str
    choices: list[str]
    answer: int  # index into choices
    abc_label: bool


@dataclass
class Diagram:
    image_id: str  # e.g. "16.png"
    category: str
    split: str  # "train" | "test"
    graph: DiagramGraph
    questions: list[Question] = field(default_factory=list)
    dropped_edges: int = 0

    @property
    def image_path(self) -> Path:
        return AI2D_DIR / "images" / self.image_id


def available() -> bool:
    return (AI2D_DIR / "annotations").is_dir() and (AI2D_DIR / "images").is_dir()


def _centroid(poly) -> tuple[float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def build_graph(ann: dict) -> tuple[DiagramGraph, int]:
    texts = ann.get("text", {})
    nodes: list[Node] = []
    by_norm: dict[str, str] = {}
    tid_to_node: dict[str, str] = {}
    centres: dict[str, tuple[float, float]] = {}
    for tid, t in sorted(texts.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 0):
        label = str(t.get("value", "")).strip()
        key = norm(label)
        if not key:
            continue
        (x1, y1), (x2, y2) = t["rectangle"]
        centres[tid] = ((x1 + x2) / 2, (y1 + y2) / 2)
        if key in by_norm:
            tid_to_node[tid] = by_norm[key]
            continue
        nid = f"n{len(nodes) + 1}"
        nodes.append(Node(id=nid, label=label, bbox=[float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]))
        by_norm[key] = nid
        tid_to_node[tid] = nid

    blobs = ann.get("blobs", {})
    blob_labels: dict[str, list[str]] = {}
    for r in ann.get("relationships", {}).values():
        if r["category"] not in ("intraObjectLabel", "intraObjectRegionLabel"):
            continue
        o, d = r["origin"], r["destination"]
        if o.startswith("T") and d.startswith("B"):
            blob_labels.setdefault(d, []).append(o)
        elif d.startswith("T") and o.startswith("B"):
            blob_labels.setdefault(o, []).append(d)
    blob_to_node: dict[str, str] = {}
    for b, tids in blob_labels.items():
        tids = [t for t in tids if t in tid_to_node]
        if not tids:
            continue
        if len(tids) > 1 and b in blobs:
            cx, cy = _centroid(blobs[b]["polygon"])
            tids.sort(key=lambda t: (centres[t][0] - cx) ** 2 + (centres[t][1] - cy) ** 2)
        blob_to_node[b] = tid_to_node[tids[0]]

    def resolve(x: str) -> str | None:
        if x.startswith("T"):
            return tid_to_node.get(x)
        if x.startswith("B"):
            return blob_to_node.get(x)
        return None

    edges: list[Edge] = []
    dropped = 0
    for r in ann.get("relationships", {}).values():
        if r["category"] != "interObjectLinkage" or not r.get("hasDirectionality"):
            continue
        s, t = resolve(r["origin"]), resolve(r["destination"])
        if s is None or t is None or s == t:
            dropped += 1
            continue
        edges.append(Edge(source=s, target=t))
    g = DiagramGraph(nodes=nodes, edges=edges, source="ai2d-gold")
    g.validate()
    g.warnings = []
    return g, dropped


def load(split: str | None = None, with_abc: bool = False) -> list[Diagram]:
    """Load AI2D diagrams (optionally one split). Requires the dataset in external/ai2d/."""
    if not available():
        raise FileNotFoundError("AI2D not found; run `python scripts/get_ai2d.py` first.")
    cats = json.loads((AI2D_DIR / "categories.json").read_text())
    test_ids = {l.strip() for l in (AI2D_ROOT / "ai2d_test_ids.csv").read_text().split() if l.strip()}
    out: list[Diagram] = []
    for img_id in sorted(cats, key=lambda s: int(s.split(".")[0])):
        sp = "test" if img_id.split(".")[0] in test_ids else "train"
        if split and sp != split:
            continue
        ann = json.loads((AI2D_DIR / "annotations" / f"{img_id}.json").read_text())
        graph, dropped = build_graph(ann)
        d = Diagram(image_id=img_id, category=cats[img_id], split=sp, graph=graph, dropped_edges=dropped)
        qpath = AI2D_DIR / "questions" / f"{img_id}.json"
        if qpath.exists():
            for text, q in json.loads(qpath.read_text())["questions"].items():
                if q["abcLabel"] and not with_abc:
                    continue
                d.questions.append(Question(q["questionId"], text, list(q["answerTexts"]), int(q["correctAnswer"]), bool(q["abcLabel"])))
        out.append(d)
    return out


def gold_locations(ann: dict) -> dict[str, list[dict]]:
    """Where each label really points, from AI2D's annotations (for grounding evaluation).

    {normalised label: [{"type": "point", "xy": (x, y)} | {"type": "box", "box": [x1, y1, x2, y2]}]}
    * ``intraObjectLinkage`` (label --line--> object): the tip of the connector line, i.e. the end
      of the line farthest from the label;
    * ``intraObjectLabel`` / ``intraObjectRegionLabel`` (label names a blob): the blob's bounding box.
    Labels without such a relationship have no gold location and are not scored.
    """
    import math

    texts, blobs, arrows = ann.get("text", {}), ann.get("blobs", {}), ann.get("arrows", {})
    out: dict[str, list[dict]] = {}
    for r in ann.get("relationships", {}).values():
        cat = r.get("category")
        o, d = r.get("origin", ""), r.get("destination", "")
        tid = o if o.startswith("T") else d if d.startswith("T") else None
        if tid is None or tid not in texts:
            continue
        key = norm(str(texts[tid].get("value", "")))
        if not key:
            continue
        (x1, y1), (x2, y2) = texts[tid]["rectangle"]
        tc = ((x1 + x2) / 2, (y1 + y2) / 2)
        if cat == "intraObjectLinkage" and r.get("connector") in arrows:
            poly = arrows[r["connector"]]["polygon"]
            tip = max(poly, key=lambda p: math.dist(p, tc))
            out.setdefault(key, []).append({"type": "point", "xy": (float(tip[0]), float(tip[1]))})
        elif cat in ("intraObjectLabel", "intraObjectRegionLabel"):
            bid = d if d.startswith("B") else o if o.startswith("B") else None
            if bid in blobs:
                xs = [p[0] for p in blobs[bid]["polygon"]]
                ys = [p[1] for p in blobs[bid]["polygon"]]
                out.setdefault(key, []).append({"type": "box", "box": [min(xs), min(ys), max(xs), max(ys)]})
    return out


def location_hit(box: list[float], golds: list[dict], image_size: tuple[int, int], pad_frac: float = 0.03) -> bool:
    """Does a highlighted box point at the gold location? Point gold: the (padded) box contains the
    point. Box gold: the highlight's centre lies inside the gold box."""
    import math

    if not box:
        return False
    pad = pad_frac * math.dist((0, 0), image_size)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for g in golds:
        if g["type"] == "point":
            x, y = g["xy"]
            if box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad:
                return True
        else:
            b = g["box"]
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
    return False
