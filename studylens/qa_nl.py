"""Natural-language question answering over a diagram (AI) with deterministic evidence checks.

Two answerers, both Qwen2.5-VL-3B with BASE weights (the extraction LoRA is disabled):

* ``direct_answer``   (condition E2): image + question [+ options] -> answer. No graph.
* ``grounded_answer`` (condition E3): image + extracted graph with numbered arrows + question
                       [+ options] -> JSON {answer|choice, evidence: [arrow ids], graph_supported}.

``verify`` then checks the model's citations against the graph WITHOUT any AI:
* every cited id must exist in the graph (otherwise it is a hallucinated citation and is removed);
* the answer should mention an endpoint of a cited arrow (``answer_in_evidence``);
* status:  "grounded"     - graph_supported and >= 1 valid cited arrow
           "unsupported"  - the model answered but no valid arrow supports it (image-only answer,
                            shown to the user with a warning)
           "abstained"    - the model said it cannot answer
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from .schema import DiagramGraph

LETTERS = "abcdefgh"

GROUNDED_PROMPT = """Answer a question about this diagram. You are also given a graph that was extracted automatically from the diagram (it may contain mistakes). Arrow "A -> B" means the diagram shows an arrow from A to B.

Entities: {entities}
Arrows:
{arrows}

Question: {question}
{options}
Use the diagram and the arrows. Reply with ONLY a JSON object:
{{"{answer_key}": {answer_hint}, "evidence": [ids of the arrows your answer relies on, e.g. "e1"], "graph_supported": true or false}}
Set "graph_supported" to false and "evidence" to [] if none of the listed arrows supports your answer. If the question cannot be answered from the diagram, set "{answer_key}" to "cannot answer"."""

DIRECT_PROMPT = """Answer a question about this diagram using what the diagram shows.

Question: {question}
{options}
{instruction}"""


OCR_PROMPT = """You cannot see the diagram. An OCR system read the following pieces of text from it. Each line gives the text and the position of its centre in pixels from the top-left corner of a {w}x{h} image.

{lines}

Question: {question}
{options}
{instruction}"""


def build_ocr_prompt(lines: list[dict], size: tuple[int, int], question: str, choices: list[str] | None = None) -> str:
    rows = []
    for ln in sorted(lines, key=lambda r: ((r["box"][1] + r["box"][3]) / 2, r["box"][0]))[:80]:
        b = ln["box"]
        rows.append(f'"{ln["text"]}" at ({round((b[0] + b[2]) / 2)}, {round((b[1] + b[3]) / 2)})')
    if choices:
        instr = "Reply with ONLY the letter of the correct option."
    else:
        instr = "Answer in one short sentence. If the text does not give enough information, reply exactly: Not shown in the diagram."
    return OCR_PROMPT.format(w=size[0], h=size[1], lines="\n".join(rows) or "(no text was found)", question=question,
                             options=_options_block(choices), instruction=instr)


def _options_block(choices: list[str] | None) -> str:
    if not choices:
        return ""
    return "Options:\n" + "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices)) + "\n"


def graph_context(graph: DiagramGraph) -> tuple[str, str, dict[str, Any]]:
    """Entities line, numbered arrows block, and the id -> edge map."""
    ids = {}
    lines = []
    for i, e in enumerate(graph.edges, start=1):
        eid = f"e{i}"
        ids[eid] = e
        lines.append(f"{eid}: {graph.edge_text(e)}")
    ents = ", ".join(n.label for n in graph.nodes) or "(none)"
    return ents, "\n".join(lines) or "(no arrows were extracted)", ids


def build_grounded_prompt(question: str, graph: DiagramGraph, choices: list[str] | None = None) -> tuple[str, dict]:
    ents, arrows, ids = graph_context(graph)
    if choices:
        key, hint = "choice", '"<letter of the correct option>"'
    else:
        key, hint = "answer", '"<short answer>"'
    prompt = GROUNDED_PROMPT.format(entities=ents, arrows=arrows, question=question,
                                    options=_options_block(choices), answer_key=key, answer_hint=hint)
    return prompt, ids


def build_direct_prompt(question: str, choices: list[str] | None = None) -> str:
    if choices:
        instr = "Reply with ONLY the letter of the correct option."
    else:
        instr = "Answer in one short sentence. If the diagram does not show enough to answer, reply exactly: Not shown in the diagram."
    return DIRECT_PROMPT.format(question=question, options=_options_block(choices), instruction=instr)


# ---------------------------------------------------------------------------- parsing
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def parse_json_obj(raw: str) -> dict | None:
    txt = re.sub(r"```(?:json)?", "", raw or "").strip()
    a, b = txt.find("{"), txt.rfind("}")
    if a == -1 or b == -1:
        return None
    blob = txt[a : b + 1]
    # Qwen often writes ids unquoted ("evidence": [e1, e2]) or leaves trailing commas
    quoted = re.sub(r'([\[,]\s*)(e\d+)(\s*)(?=[,\]])', r'\1"\2"\3', blob)
    no_trail = re.sub(r",\s*([}\]])", r"\1", quoted)
    for cand in (blob, quoted, no_trail, no_trail.replace("'", '"')):
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    # last resort: pull the fields out individually
    obj: dict[str, Any] = {}
    m = re.search(r'"(answer|choice)"\s*:\s*"([^"]*)"', blob)
    if m:
        obj[m.group(1)] = m.group(2)
    m = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', blob)
    if m:
        obj["evidence"] = re.findall(r"e\d+", m.group(1))
    m = re.search(r'"graph_supported"\s*:\s*(true|false)', blob, re.I)
    if m:
        obj["graph_supported"] = m.group(1).lower() == "true"
    return obj if ("answer" in obj or "choice" in obj) else None


def match_choice(text: str, choices: list[str]) -> int | None:
    """Map a model reply ("b", "b) Fox", "Fox") to an option index."""
    if not choices or text is None:
        return None
    t = str(text).strip()
    m = re.match(r"^\(?([a-hA-H])\)?(?:[).:\s]|$)", t)
    if m and LETTERS.index(m.group(1).lower()) < len(choices):
        return LETTERS.index(m.group(1).lower())
    nt = _norm(t)
    norms = [_norm(c) for c in choices]
    if nt in norms:
        return norms.index(nt)
    hits = [i for i, c in enumerate(norms) if c and re.search(rf"\b{re.escape(c)}\b", nt)]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(nt, norms, n=1, cutoff=0.75)
    return norms.index(close[0]) if close else None


_CANNOT = re.compile(r"cannot answer|not shown|can t answer|cannot be determined|unknown")


def verify(obj: dict | None, raw: str, graph: DiagramGraph, ids: dict, choices: list[str] | None = None) -> dict:
    """Deterministic post-check of a grounded answer. No AI involved."""
    res: dict[str, Any] = {"raw": raw, "parse_ok": obj is not None}
    if obj is None:
        # fall back to the raw text as an answer, but it can never be "grounded"
        obj = {"answer": raw.strip(), "evidence": [], "graph_supported": False}
    if choices:
        ci = match_choice(str(obj.get("choice", obj.get("answer", ""))), choices)
        res["choice_index"] = ci
        res["answer"] = choices[ci] if ci is not None else str(obj.get("choice", obj.get("answer", ""))).strip()
    else:
        res["choice_index"] = None
        res["answer"] = str(obj.get("answer", "")).strip()
    cited = obj.get("evidence") or []
    if isinstance(cited, str):
        cited = re.findall(r"e\d+", cited)
    cited = [str(c).strip() for c in cited]
    valid = [c for c in cited if c in ids]
    res["cited"] = cited
    res["invalid_citations"] = [c for c in cited if c not in ids]
    res["evidence"] = [
        {"id": c, "source": ids[c].source, "target": ids[c].target, "text": graph.edge_text(ids[c])} for c in valid
    ]
    ans_n = _norm(res["answer"])
    endpoints = {_norm(graph.label_of(ids[c].source)) for c in valid} | {_norm(graph.label_of(ids[c].target)) for c in valid}
    res["answer_in_evidence"] = bool(ans_n) and any(ep and (ep in ans_n or ans_n in ep) for ep in endpoints)
    is_yes_no = bool(re.match(r"^(yes|no|true|false)\b", ans_n))
    supported = bool(obj.get("graph_supported")) and bool(valid)
    res["grounding_note"] = None
    if not ans_n or _CANNOT.search(ans_n):
        res["status"] = "abstained"
    elif supported and (res["answer_in_evidence"] or is_yes_no):
        res["status"] = "grounded"
    else:
        res["status"] = "unsupported"
        if supported:
            # the model claimed support, but its cited arrows do not involve the answer
            res["grounding_note"] = "The model cited arrows that do not involve its answer, so the citation was rejected."
    res["answer_nodes"] = sorted({ids[c].source for c in valid} | {ids[c].target for c in valid})
    return res


# ---------------------------------------------------------------------------- model calls
def grounded_answer(vlm, image, question: str, graph: DiagramGraph, choices: list[str] | None = None) -> dict:
    prompt, ids = build_grounded_prompt(question, graph, choices)
    raw = vlm.generate(image, prompt, max_new_tokens=96)
    out = verify(parse_json_obj(raw), raw, graph, ids, choices)
    out["latency_s"] = vlm.last_latency_s
    out["method"] = "Qwen2.5-VL-3B answers from image + extracted graph; citations checked deterministically (E3)"
    return out


EXPLAIN_PROMPT = """A student asked about a diagram: {question}
The answer, computed from the arrows extracted from the diagram, is: {answer}
The arrows used as evidence:
{arrows}

Explain this answer to the student in one or two sentences, using only these arrows (an arrow A -> B means the diagram draws an arrow from A to B). Do not add facts the arrows do not show."""


def explain(vlm, question: str, answer: str, evidence: list[dict]) -> dict:
    """The language model only EXPLAINS an answer the graph query already computed (no new facts)."""
    arrows = "\n".join(f"- {e['text']}" for e in evidence) or "(none)"
    raw = vlm.generate_text(EXPLAIN_PROMPT.format(question=question, answer=answer, arrows=arrows), max_new_tokens=90)
    return {"text": raw.strip(), "latency_s": vlm.last_latency_s}


def ocr_answer(vlm, lines: list[dict], size: tuple[int, int], question: str, choices: list[str] | None = None) -> dict:
    """Condition E1: OCR text only -> Qwen (text-only, base weights). The model never sees the image."""
    raw = vlm.generate_text(build_ocr_prompt(lines, size, question, choices), max_new_tokens=64)
    out: dict[str, Any] = {"raw": raw, "latency_s": vlm.last_latency_s,
                           "method": "PaddleOCR text + positions -> Qwen2.5-VL-3B text-only (E1, no image, no graph)"}
    if choices:
        ci = match_choice(raw, choices)
        out["choice_index"] = ci
        out["answer"] = choices[ci] if ci is not None else raw.strip()
    else:
        out["choice_index"] = None
        out["answer"] = raw.strip()
    return out


def direct_answer(vlm, image, question: str, choices: list[str] | None = None) -> dict:
    raw = vlm.generate(image, build_direct_prompt(question, choices), max_new_tokens=64)
    out: dict[str, Any] = {"raw": raw, "latency_s": vlm.last_latency_s,
                           "method": "Qwen2.5-VL-3B answers from the image only (E2, no graph, no evidence)"}
    if choices:
        ci = match_choice(raw, choices)
        out["choice_index"] = ci
        out["answer"] = choices[ci] if ci is not None else raw.strip()
    else:
        out["choice_index"] = None
        out["answer"] = raw.strip()
    return out
