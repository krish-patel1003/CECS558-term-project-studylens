"""Diagram Scene Graph (DSG) extraction: OCR-anchored, so the model never writes coordinates.

1. **OCR** (PaddleOCR, ``studylens.ocr``) gives every text item with an exact box: t1 .. tn.
2. **Parts (no AI)**: OCR lines are grouped into part labels deterministically (lines inside the
   same enclosing shape, or stacked lines); the title is the topmost text outside any shape.
   Arrowheads misread as text ("V", "-") are filtered by OCR confidence.
3. **Arrows (AI)**: Qwen2.5-VL sees the image and the list of part NAMES and lists the drawn arrows
   by name; names are matched back to parts deterministically. No coordinates are generated.
4. **Regions (mostly no AI)**: for each part, the shape enclosing its label is found by image
   analysis (``regions.enclosing_box``). Only parts that are not inside a closed shape (e.g. an
   anatomy label connected to its part by a leader line) are located by a focused Qwen grounding
   call. Otherwise the label box itself is used. Each node records how its region was obtained.

The result is a ``DiagramGraph`` (nodes with ``bbox`` = region to highlight and ``label_bbox``),
so the deterministic graph query engine works on it unchanged.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

from PIL import Image

from .regions import binarise, enclosing_box
from .schema import DiagramGraph, Edge, Node

# Arrows are asked for BY NAME: asking for numbered part ids made the 3B model chain the list in
# order (p1->p2->p3...), while names gave arrow F1 0.67-0.91 on pilot diagrams (see docs).
ARROW_PROMPT = """You are reading a diagram. These are the names of its labelled parts: {names}.

List every arrow that is drawn in the diagram. An arrow goes FROM the part at its tail TO the part its arrowhead touches; look carefully at which end has the arrowhead. Only arrows that are actually drawn, not connections you would expect. Use the part names exactly as given.

Reply with ONLY JSON: {{"arrows": [{{"from": "<part name>", "to": "<part name>"}}]}}"""

GROUND_PROMPT = ("For each label below, locate the part of the diagram it names and give its bounding box. If a label is "
                 "connected by a line to a drawing, give the box of the PART the line points to, not of the label text. "
                 "Otherwise give the box of the object or region the label names.\n"
                 "Labels: {labels}\nReply with ONLY a JSON list: [{{\"label\": \"...\", \"bbox_2d\": [x1, y1, x2, y2]}}]")
_GROUNDED = re.compile(r'"label"\s*:\s*"([^"]*)"\s*,\s*"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)')

_BOX = re.compile(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")


def number_texts(lines: list[dict]) -> list[dict]:
    """OCR lines -> text items with ids t1..tn, in reading order (top-to-bottom, left-to-right)."""
    # drop OCR noise: arrowheads read as "V" / "-" etc. Single characters (e.g. AI2D letter labels
    # "A", "D") are kept only when read with high confidence.
    def keep(r):
        t = str(r["text"]).strip()
        alnum = sum(c.isalnum() for c in t)
        score = float(r.get("score", 1.0))
        return alnum >= 1 and score >= 0.5 and (alnum > 1 or score >= 0.95)
    lines = [r for r in lines if keep(r)]
    order = sorted(lines, key=lambda r: (round(((r["box"][1] + r["box"][3]) / 2) / 12), r["box"][0]))
    return [{"id": f"t{i}", "text": r["text"], "box": [float(v) for v in r["box"]], "role": "other"}
            for i, r in enumerate(order, start=1)]


def _union(boxes: list[list[float]]) -> list[float] | None:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def group_parts(texts: list[dict], ink, size: tuple[int, int]) -> tuple[list[dict], str | None]:
    """Deterministic grouping of OCR lines into part labels (no AI).

    * lines inside the same enclosing shape form one part (multi-line box labels);
    * remaining lines that are stacked (small vertical gap, overlapping horizontally) are merged;
    * a wide line in the top 12% of the image that is not inside a shape is the title.
    Returns parts [{"text_ids", "label", "label_box", "shape"}] and the title text id (or None).
    """
    W, H = size
    shapes = {t["id"]: enclosing_box(ink, t["box"]) for t in texts}
    heights = sorted(t["box"][3] - t["box"][1] for t in texts) or [10]
    med_h = heights[len(heights) // 2]
    title = None
    free = [t for t in texts if shapes[t["id"]] is None]
    tops = sorted([t for t in free if (t["box"][1] + t["box"][3]) / 2 < 0.10 * H], key=lambda t: t["box"][1])
    if tops and len(tops) <= 2:
        title = tops[0]["id"]
    groups: list[list[dict]] = []
    # 1) same enclosing shape
    by_shape: list[tuple[list[float], list[dict]]] = []
    for t in texts:
        sb = shapes[t["id"]]
        if sb is None or t["id"] == title:
            continue
        for box, members in by_shape:
            if _iou(box, sb) > 0.8:
                members.append(t)
                break
        else:
            by_shape.append((sb, [t]))
    groups += [m for _, m in by_shape]
    # 2) stacked free lines
    rest = sorted([t for t in free if t["id"] != title], key=lambda t: (t["box"][1], t["box"][0]))
    used: set[str] = set()
    for t in rest:
        if t["id"] in used:
            continue
        g, used_now = [t], {t["id"]}
        cur = t
        for u in rest:
            if u["id"] in used or u["id"] in used_now:
                continue
            gap = u["box"][1] - cur["box"][3]
            overlap = min(cur["box"][2], u["box"][2]) - max(cur["box"][0], u["box"][0])
            width = min(cur["box"][2] - cur["box"][0], u["box"][2] - u["box"][0])
            if -2 <= gap <= 0.6 * med_h and overlap > 0.5 * width:
                g.append(u)
                used_now.add(u["id"])
                cur = u
        used |= used_now
        groups.append(g)
    parts = []
    for g in groups:
        g = sorted(g, key=lambda t: (t["box"][1], t["box"][0]))
        box = _union([t["box"] for t in g])
        parts.append({"text_ids": [t["id"] for t in g], "label": " ".join(t["text"] for t in g).strip(),
                      "label_box": box, "shape": shapes[g[0]["id"]]})
    parts.sort(key=lambda q: (round(((q["label_box"][1] + q["label_box"][3]) / 2) / 12), q["label_box"][0]))
    return parts, title


def build_arrow_prompt(parts: list[dict], size: tuple[int, int] | None = None) -> str:
    return ARROW_PROMPT.format(names=json.dumps([q["label"] for q in parts[:120]], ensure_ascii=False))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def parse_arrows(raw: str, parts: list[dict]) -> tuple[list[tuple[int, int]], list[str]]:
    """[(tail index, head index)] (0-based). Names are matched to parts exactly, then fuzzily;
    unknown names are dropped with a warning. Tolerant of truncated output."""
    import difflib

    warnings = []
    names = {_norm(q["label"]): i for i, q in enumerate(parts)}

    def find(name):
        k = _norm(name)
        if k in names:
            return names[k]
        close = difflib.get_close_matches(k, names.keys(), n=1, cutoff=0.8)
        return names[close[0]] if close else None

    out, seen = [], set()
    for a, b in re.findall(r'"from"\s*:\s*"([^"]+)"\s*,\s*"to"\s*:\s*"([^"]+)"', raw or ""):
        s_, t_ = find(a), find(b)
        if s_ is None or t_ is None:
            warnings.append(f"dropped arrow {a!r} -> {b!r}: not a listed part")
        elif s_ != t_ and (s_, t_) not in seen:
            seen.add((s_, t_))
            out.append((s_, t_))
    if raw and raw.count("{") > raw.count("}"):
        warnings.append("arrow list was cut off; kept the complete arrows")
    return out, warnings


class DSGExtractor:
    """Builds a DSG with OCR + one structure call + regions. ``vlm`` is a QwenVLExtractor."""

    name = "dsg (PaddleOCR + Qwen2.5-VL structure + regions)"

    def __init__(self, vlm, ocr, max_grounding: int = 24):
        self.vlm, self.ocr, self.max_grounding = vlm, ocr, max_grounding
        self.last_raw_output: str | None = None
        self.last_latency_s: float | None = None

    def ground(self, image: Image.Image, labels: list[str]) -> dict[str, list[float]]:
        """One grounding call for several labels (measured on AI2D parts diagrams: as accurate as one
        call per label and ~2x faster). Returns {label: box in original pixels}."""
        if not labels:
            return {}
        raw, (sx, sy) = self.vlm._generate(image, GROUND_PROMPT.format(labels=json.dumps(labels)), 30 * len(labels) + 40)
        W, H = image.size
        out = {}
        for m in _GROUNDED.finditer(raw or ""):
            b = [float(m.group(2)) / sx, float(m.group(3)) / sy, float(m.group(4)) / sx, float(m.group(5)) / sy]
            b = [max(0.0, min(W, b[0])), max(0.0, min(H, b[1])), max(0.0, min(W, b[2])), max(0.0, min(H, b[3]))]
            if b[2] - b[0] > 2 and b[3] - b[1] > 2:
                out[m.group(1).strip().lower()] = b
        return out

    def extract_stream(self, image: Image.Image, ocr_lines: list[dict] | None = None) -> Iterator[dict]:
        t0 = time.perf_counter()
        yield {"stage": "ocr"}
        lines = ocr_lines if ocr_lines is not None else self.ocr.read(image)
        texts = number_texts(lines)
        ink = binarise(image)
        parts, title = group_parts(texts, ink, image.size)
        yield {"stage": "ocr_done", "n_texts": len(texts), "entities": len(parts), "elapsed_s": round(time.perf_counter() - t0, 1)}
        yield {"stage": "structure"}
        raw = self.vlm.generate(image, build_arrow_prompt(parts), max_new_tokens=min(3072, 18 * len(parts) + 80))
        self.last_raw_output = raw
        arrows, warnings = parse_arrows(raw, parts)
        yield {"stage": "structure_done", "entities": len(parts), "arrows": len(arrows), "elapsed_s": round(time.perf_counter() - t0, 1)}
        need = [q["label"] for q in parts if q["shape"] is None][: self.max_grounding]
        located = {}
        if need:
            yield {"stage": "locating", "n": len(need)}
            located = self.ground(image, need)
        byid = {t["id"]: t for t in texts}
        nodes = []
        for k, q in enumerate(parts, start=1):
            if q["shape"] is not None:
                region, src = q["shape"], "shape"
            elif q["label"].lower() in located:
                region, src = located[q["label"].lower()], "grounded"
            else:
                region, src = q["label_box"], "label"
            nodes.append(Node(id=f"n{k}", label=q["label"], bbox=region, label_bbox=q["label_box"], region_source=src,
                              text_ids=list(q["text_ids"])))
            for t in q["text_ids"]:
                byid[t]["role"] = "label"
        if title:
            byid[title]["role"] = "title"
        edges = [Edge(source=f"n{s_ + 1}", target=f"n{t_ + 1}") for s_, t_ in arrows]
        g = DiagramGraph(nodes=nodes, edges=edges, source="dsg: PaddleOCR + Qwen2.5-VL-3B arrows + shape/grounded regions",
                         image_size=list(image.size), warnings=warnings, texts=texts, title=byid[title]["text"] if title else None)
        g.validate()
        self.last_latency_s = time.perf_counter() - t0
        yield {"stage": "done", "graph": g}

    def extract(self, image: Image.Image, ocr_lines: list[dict] | None = None) -> DiagramGraph:
        g = None
        for ev in self.extract_stream(image, ocr_lines):
            if ev["stage"] == "done":
                g = ev["graph"]
        return g


def region_counts(g: DiagramGraph) -> dict[str, Any]:
    out: dict[str, int] = {}
    for n in g.nodes:
        out[n.region_source or "none"] = out.get(n.region_source or "none", 0) + 1
    return out
