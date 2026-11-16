"""Explanations with references: the main StudyLens answer.

Pipeline for one question (image + Diagram Scene Graph):
1. **Facts (no AI where possible).** The question goes through the graph-query engine
   (``query.structured_answer``). For a structural question NetworkX computes the answer and the
   arrows that support it; these facts are given to the model to narrate.
2. **Explanation (AI).** Qwen2.5-VL sees the image, the part names and the extracted arrows and
   writes a short explanation that names parts exactly as they are labelled.
3. **Reference linking (no AI).** Every mention of a part name in the explanation is linked to that
   part of the scene graph (exact, case-insensitive, simple plurals); consecutive mentions joined by
   graph arrows become a *flow* reference; the arrows behind a NetworkX answer become a flow too.
   Links are computed, not generated, so a reference can never point at something that is not in
   the graph. (A 3B model asked to write ``[n7]``-style ids itself ignored the instruction; see
   docs/model_and_prompts.md.)
4. Optional: ``{locate: X}`` requests for things the graph lacks are resolved by one grounding call
   and marked approximate.
"""

from __future__ import annotations

import re
from typing import Any

from .query import structured_answer
from .schema import DiagramGraph

EXPLAIN_PROMPT = """You are a tutor helping a student understand this diagram.
Parts labelled in the diagram: {names}
Arrows extracted from the diagram (A -> B means an arrow from A to B; the extraction may contain mistakes, the image is the final authority):
{arrows}
{facts}
Student's question: {question}

Answer in 3 to 6 sentences (use numbered steps if you describe a process or a flow). Mention parts by their names exactly as labelled in the diagram. Explain what the diagram shows; if the diagram does not show what is asked, say so."""

_LOCATE = re.compile(r"\{\s*locate\s*:\s*([^}]{1,60})\}", re.I)

# Boxes come from a SECOND call: the model's native "explain and box the parts" answer (the B2 baseline
# prompt of scripts/evaluate_grounding.py), of which only the boxes are used. On 100 AI2D dev diagrams
# these in-answer boxes were right 69% of the time. Asking for the boxes inside the explanation call
# instead (one JSON answer) cut explanations from ~170 to ~40 words and left raw JSON in 23 of 100
# answers (docs/evaluation.md), so the explanation keeps its own plain-text call (EXPLAIN_PROMPT).
BOX_PROMPT = """Explain this diagram to a student: go through its labelled parts one by one and, if there are arrows, describe the flow they show.
Then list every part you mentioned with its bounding box in this image.
Reply with ONLY JSON: {"explanation": "...", "parts": [{"name": "...", "bbox_2d": [x1, y1, x2, y2]}]}"""

_JSON_TAIL = re.compile(r"(?:\n[^\n]*bounding box[^\n]*\n)?\s*(?:```(?:json)?\s*)?\{\s*\"(?:parts|explanation)\"[\s\S]*$", re.I)


def clean_explanation(text: str) -> str:
    """Remove a JSON block (and the heading introducing it) that the model sometimes appends."""
    text = _JSON_TAIL.sub("", text or "")
    return re.sub(r"```(?:json)?|```", "", text).strip()


_EXPL = re.compile(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)
_PART_BOX = re.compile(r'"name"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)')


def parse_explain_json(raw: str, scale: tuple[float, float] = (1.0, 1.0)) -> tuple[str, list[dict]]:
    """(explanation text, [{"name", "bbox"} in original pixels]); tolerant of truncated JSON and of
    plain-text answers (then there are no boxes)."""
    import json as _json

    raw = raw or ""
    sx, sy = scale
    parts = [{"name": m.group(1).strip(), "bbox": [float(m.group(2)) / sx, float(m.group(3)) / sy,
                                                   float(m.group(4)) / sx, float(m.group(5)) / sy]}
             for m in _PART_BOX.finditer(raw)]
    m = _EXPL.search(raw)
    if m:
        try:
            text = _json.loads('"' + m.group(1) + '"')
        except ValueError:
            text = m.group(1).replace('\\n', '\n').replace('\\"', '"')
    else:
        text = re.sub(r"```(?:json)?|```", "", raw).strip()
    return text.strip(), parts


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def _on_label_text(box, label_box, size) -> bool:
    """The box is centred on the part's own label text (a common grounding mistake), not on the part."""
    if not label_box:
        return False
    W, H = size or (1000, 1000)
    pad = 0.01 * max(W, H)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return label_box[0] - pad <= cx <= label_box[2] + pad and label_box[1] - pad <= cy <= label_box[3] + pad


def augment_with_answer_boxes(graph: DiagramGraph, parts: list[dict]) -> tuple[DiagramGraph, dict]:
    """Use the model's in-answer boxes for scene-graph parts (when not centred on the label text), and add
    parts the model named that the scene graph lacks but whose name is text read in the diagram
    (verified by OCR; never invented). Returns (graph copy, stats)."""
    import difflib
    import copy

    g = copy.deepcopy(graph)
    size = tuple(g.image_size) if g.image_size else None
    by_name = {_norm_name(n.label): n for n in g.nodes}
    texts = [_norm_name(t.get("text", "")) for t in (g.texts or [])]
    stats = {"answer_boxes": 0, "rejected_on_label": 0, "added_from_text": 0, "unverified": 0}
    seen = set()
    for p in parts:
        k = _norm_name(p["name"])
        if not k or k in seen:
            continue
        seen.add(k)
        n = by_name.get(k)
        if n is None:
            close = difflib.get_close_matches(k, list(by_name), n=1, cutoff=0.85)
            n = by_name[close[0]] if close else None
        b = p["bbox"]
        if size:
            b = [max(0.0, min(size[0], b[0])), max(0.0, min(size[1], b[1])), max(0.0, min(size[0], b[2])), max(0.0, min(size[1], b[3]))]
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            continue
        if n is not None:
            if _on_label_text(b, getattr(n, "label_bbox", None), size):
                stats["rejected_on_label"] += 1
                continue
            n.bbox, n.region_source = b, "answer"
            stats["answer_boxes"] += 1
        elif any(k == t or (len(k) > 3 and (k in t or t in k)) for t in texts if t):
            from .schema import Node
            g.nodes.append(Node(id=f"a{len(g.nodes) + 1}", label=p["name"].strip(), bbox=b, region_source="answer"))
            by_name[k] = g.nodes[-1]
            stats["added_from_text"] += 1
        else:
            stats["unverified"] += 1  # named something not in the diagram's text: no reference
    return g, stats


def serialise(graph: DiagramGraph) -> tuple[str, str, dict[str, Any]]:
    names = ", ".join(f'"{n.label}"' for n in graph.nodes) or "(none found)"
    arrows, emap = [], {}
    for i, e in enumerate(graph.edges, start=1):
        emap[f"e{i}"] = e
        arrows.append(f"- {graph.label_of(e.source)} -> {graph.label_of(e.target)}")
    return names, "\n".join(arrows) or "(no arrows found)", emap


def facts_block(s: dict, graph: DiagramGraph, emap: dict | None = None) -> str:
    if not s or s.get("status") != "answered":
        return ""
    ev = [x["text"] for x in s.get("evidence", [])]
    return ("Facts computed exactly from the extracted arrows (use them):\n"
            f"- {s['answer']}\n" + ("- supporting arrows: " + "; ".join(ev) + "\n" if ev else ""))


def _patterns(graph: DiagramGraph) -> list[tuple[re.Pattern, str]]:
    """Regex per part label (longest first): case-insensitive, flexible whitespace, optional plural."""
    pats = []
    for n in sorted(graph.nodes, key=lambda n: -len(n.label)):
        lab = n.label.strip()
        words = [re.escape(w) for w in re.split(r"\s+", lab) if w]
        if not words:
            continue
        if len(lab) == 1:
            # one-letter labels ("A", "D") vs the article "A": only after a cue word, in brackets,
            # or right before punctuation
            L = re.escape(lab)
            pats.append((re.compile(rf"\b(?i:label|letter|part|point|stage|structure|labelled|marked)\s+({L})(?![\w-])"
                                    rf"|\(({L})\)|(?<![\w-])({L})(?=[.,;:!?])"), n.id))
            continue
        body = r"\s+".join(words)
        exact_case = len(lab) <= 2
        pats.append((re.compile(rf"(?<![\w-])({body})(?:e?s)?(?![\w-])", 0 if exact_case else re.I), n.id))
    return pats


def link_mentions(text: str, graph: DiagramGraph) -> list[tuple[int, int, str]]:
    """Non-overlapping (start, end, node id) mentions of part names in the text, in order."""
    taken = [False] * len(text)
    found = []
    for pat, nid in _patterns(graph):
        for m in pat.finditer(text):
            # one-letter patterns capture the letter in one of several groups: use that group's span
            groups = [i for i in range(1, (pat.groups or 0) + 1) if m.group(i) is not None]
            st, en = (m.span(groups[0]) if pat.groups > 1 else m.span())
            if any(taken[st:en]):
                continue
            for i in range(st, en):
                taken[i] = True
            found.append((st, en, nid))
    return sorted(found)


def build_references(text: str, graph: DiagramGraph, s: dict | None = None) -> tuple[str, list[dict], dict]:
    """Insert [[k]] markers after linked mentions; add flow references from graph arrows."""
    nodes = {n.id: n for n in graph.nodes}
    succ = {(e.source, e.target) for e in graph.edges}
    mentions = link_mentions(text, graph)
    refs: list[dict] = []
    num_of: dict[Any, int] = {}

    def ref(key, entry) -> int:
        if key not in num_of:
            entry["num"] = len(refs) + 1
            refs.append(entry)
            num_of[key] = entry["num"]
        return num_of[key]

    inserts: dict[int, list[int]] = {}
    for start, end, nid in mentions:
        n = nodes[nid]
        k = ref(("n", nid), {"kind": "part", "provenance": "graph", "node": nid, "label": n.label, "bbox": n.bbox,
                             "region_source": n.region_source})
        inserts.setdefault(end, [])
        if k not in inserts[end]:
            inserts[end].append(k)
    # flows: maximal runs of consecutive mentions where each step is a graph arrow
    run = [mentions[0]] if mentions else []
    flows = []
    for m in mentions[1:]:
        if m[2] == run[-1][2]:
            continue
        if (run[-1][2], m[2]) in succ:
            run.append(m)
        else:
            if len(run) >= 2:
                flows.append(run)
            run = [m]
    if len(run) >= 2:
        flows.append(run)
    for fl in flows:
        chain = [(a[2], b[2]) for a, b in zip(fl, fl[1:])]
        label = " → ".join(nodes[x[2]].label for x in fl)
        k = ref(("f", tuple(chain)), {"kind": "flow" if len(chain) > 1 else "arrow", "provenance": "graph",
                                       "arrows": [{"source": a, "target": b} for a, b in chain], "label": label,
                                       "edges": [f"{a}->{b}" for a, b in chain], "connected": True})
        inserts.setdefault(fl[-1][1], []).append(k)
    # the arrows behind a NetworkX answer
    facts_ref = None
    if s and s.get("status") == "answered" and s.get("evidence"):
        chain = [(x["source"], x["target"]) for x in s["evidence"]]
        connected = all(a[1] == b[0] for a, b in zip(chain, chain[1:]))
        label = (" → ".join([nodes[chain[0][0]].label] + [nodes[b].label for _, b in chain]) if connected
                 else "; ".join(f"{nodes[a].label} → {nodes[b].label}" for a, b in chain))
        facts_ref = ref(("facts", tuple(chain)), {"kind": "flow" if len(chain) > 1 else "arrow", "provenance": "graph",
                                                 "arrows": [{"source": a, "target": b} for a, b in chain], "label": label,
                                                 "edges": [f"{a}->{b}" for a, b in chain], "connected": connected,
                                                 "from_query": True})
    out, last = [], 0
    for pos in sorted(inserts):
        out.append(text[last:pos])
        out.append("".join(f"[[{k}]]" for k in inserts[pos]))
        last = pos
    out.append(text[last:])
    stats = {"n_refs": len(refs), "n_mentions": len(mentions), "n_flows": len(flows), "facts_ref": facts_ref,
             "invalid_ids": [], "broken_flows": 0}
    return "".join(out), refs, stats


def resolve_locates(text: str, refs: list[dict], locator, image) -> str:
    names = list(dict.fromkeys(m.strip() for m in _LOCATE.findall(text)))[:8]
    boxes = locator(image, names) if (names and locator is not None) else {}

    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        b = boxes.get(name.lower())
        if not b:
            return ""
        refs.append({"num": len(refs) + 1, "kind": "part", "provenance": "located", "label": name, "bbox": b,
                     "region_source": "grounded"})
        return f" [[{refs[-1]['num']}]]"

    return _LOCATE.sub(repl, text)


def explain(vlm, image, question: str, graph: DiagramGraph, choices: list[str] | None = None, locator=None) -> dict:
    names, arrows, emap = serialise(graph)
    s = structured_answer(question, graph, choices, parser=vlm)
    prompt = EXPLAIN_PROMPT.format(names=names, arrows=arrows, facts=facts_block(s, graph), question=question)
    raw = vlm.generate(image, prompt, max_new_tokens=250)
    text, refs, stats = build_references(raw, graph, s)
    text = resolve_locates(text, refs, locator, image)
    return {"text": text.strip(), "references": refs, "raw": raw, "query": s, "checks": stats,
            "latency_s": getattr(vlm, "last_latency_s", None)}
