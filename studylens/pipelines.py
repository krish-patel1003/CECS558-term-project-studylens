"""The three answering pipelines compared in the project, as event generators.

E1  OCR baseline     image -> PaddleOCR text + positions -> Qwen (text only)            -> answer
E2  direct VLM       image + question -> Qwen2.5-VL                                    -> answer
E3  graph-grounded   image -> extracted graph -> question parsed into a graph query
                     (rules, else Qwen as parser only) -> NetworkX -> answer + cited arrows;
                     if the graph query cannot answer, Qwen answers from image + graph and
                     must cite arrow ids, which are checked deterministically (fallback)

Each ``run_*`` function yields progress events
    {"pipeline": "E3", "stage": "query", "state": "running" | "done" | "skipped" | "error", ...}
and finishes with {"pipeline": ..., "result": {...}}. The web UI streams these events to draw the
live pipeline diagrams; the evaluation scripts use ``collect`` to get just the result. The stage
names match the boxes drawn in the UI.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from . import explain as explain_mod
from . import qa_nl
from .query import structured_answer
from .schema import DiagramGraph

STAGES = {
    "EX": ["image", "graph", "facts", "explain", "check", "locate", "answer"],
    "E1": ["image", "ocr", "llm_text", "answer"],
    "E2": ["image", "vlm", "answer"],
    "E3": ["image", "extract", "parse", "query", "explain", "fallback", "answer"],
}


def _ev(p: str, stage: str, state: str, **detail) -> dict:
    return {"pipeline": p, "stage": stage, "state": state, **detail}


def collect(gen: Iterator[dict]) -> dict:
    """Run a pipeline generator to completion and return its result."""
    result = None
    for ev in gen:
        if "result" in ev:
            result = ev["result"]
    return result


def run_explain(vlm, image, graph: DiagramGraph, question: str, choices: list[str] | None = None, locator=None) -> Iterator[dict]:
    """StudyLens's main answer: explanation with checked, clickable references (studylens/explain.py)."""
    t0 = time.perf_counter()
    yield _ev("EX", "image", "done")
    yield _ev("EX", "graph", "done", entities=len(graph.nodes), arrows=len(graph.edges), source=graph.source)
    if vlm is None:
        yield _ev("EX", "explain", "error", detail="No language model loaded (EXTRACTOR_BACKEND=qwen needed).")
        yield {"pipeline": "EX", "result": {"text": None, "status": "error", "error": "no model"}}
        return
    yield _ev("EX", "facts", "running")
    names, arrows, emap = explain_mod.serialise(graph)
    s = structured_answer(question, graph, choices, parser=vlm)
    yield _ev("EX", "facts", "done" if s["status"] == "answered" else "skipped", status=s["status"], query=s["query"],
              parser=s["parser"], answer=s["answer"] if s["status"] == "answered" else None)
    yield _ev("EX", "explain", "running")
    prompt = explain_mod.EXPLAIN_PROMPT.format(names=names, arrows=arrows, facts=explain_mod.facts_block(s, graph),
                                               question=question)
    raw = vlm.generate(image, prompt, max_new_tokens=250)
    yield _ev("EX", "explain", "done", latency_s=vlm.last_latency_s)
    yield _ev("EX", "check", "running")
    text, refs, stats = explain_mod.build_references(raw, graph, s)
    yield _ev("EX", "check", "done", n_refs=stats["n_refs"], mentions=stats["n_mentions"], flows=stats["n_flows"])
    if explain_mod._LOCATE.search(text) and locator is not None:
        yield _ev("EX", "locate", "running")
        text = explain_mod.resolve_locates(text, refs, locator, image)
        yield _ev("EX", "locate", "done", n=sum(r["provenance"] == "located" for r in refs))
    else:
        text = explain_mod._LOCATE.sub("", text)
        yield _ev("EX", "locate", "skipped")
    yield _ev("EX", "answer", "done")
    yield {"pipeline": "EX", "result": {"text": text.strip(), "references": refs, "checks": stats, "query": s, "raw": raw,
                                        "status": "explained", "total_s": round(time.perf_counter() - t0, 2)}}


def run_e1(vlm, ocr, image, question: str, choices: list[str] | None = None, ocr_lines: list[dict] | None = None) -> Iterator[dict]:
    t0 = time.perf_counter()
    yield _ev("E1", "image", "done")
    if ocr_lines is None:
        yield _ev("E1", "ocr", "running")
        try:
            ocr_lines = ocr.read(image)
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            yield _ev("E1", "ocr", "error", detail=str(exc))
            yield {"pipeline": "E1", "result": {"answer": None, "status": "error", "error": str(exc)}}
            return
        yield _ev("E1", "ocr", "done", n_lines=len(ocr_lines), latency_s=ocr.last_latency_s,
                  preview=[ln["text"] for ln in ocr_lines[:12]])
    else:
        yield _ev("E1", "ocr", "done", n_lines=len(ocr_lines), cached=True, preview=[ln["text"] for ln in ocr_lines[:12]])
    if vlm is None:
        yield _ev("E1", "llm_text", "error", detail="No language model loaded (EXTRACTOR_BACKEND=qwen needed).")
        yield {"pipeline": "E1", "result": {"answer": None, "status": "error", "error": "no model", "ocr_lines": ocr_lines}}
        return
    yield _ev("E1", "llm_text", "running")
    r = qa_nl.ocr_answer(vlm, ocr_lines, image.size, question, choices)
    yield _ev("E1", "llm_text", "done", latency_s=r["latency_s"])
    r.update(status="image_text_only", ocr_lines=ocr_lines, total_s=round(time.perf_counter() - t0, 2))
    yield _ev("E1", "answer", "done")
    yield {"pipeline": "E1", "result": r}


def run_e2(vlm, image, question: str, choices: list[str] | None = None) -> Iterator[dict]:
    t0 = time.perf_counter()
    yield _ev("E2", "image", "done")
    if vlm is None:
        yield _ev("E2", "vlm", "error", detail="No vision-language model loaded (EXTRACTOR_BACKEND=qwen needed).")
        yield {"pipeline": "E2", "result": {"answer": None, "status": "error", "error": "no model"}}
        return
    yield _ev("E2", "vlm", "running")
    r = qa_nl.direct_answer(vlm, image, question, choices)
    yield _ev("E2", "vlm", "done", latency_s=r["latency_s"])
    r.update(status="no_evidence", total_s=round(time.perf_counter() - t0, 2))
    yield _ev("E2", "answer", "done")
    yield {"pipeline": "E2", "result": r}


def combine_e3(s: dict, f: dict | None, graph: DiagramGraph, abstain_on_query_abstain: bool = True) -> dict:
    """Final E3 decision from the graph-query result ``s`` and (optionally) the model fallback ``f``.

    * query answered                       -> the graph query's answer (deterministic)
    * query ran but found no supporting arrow -> abstain (the proposal's behaviour), unless
      ``abstain_on_query_abstain`` is False, in which case the fallback is used
    * query not applicable (non-structural question, unknown entity, unparsed, no option matched)
                                           -> the model's answer from image + graph, with its
                                              arrow citations checked (``f``)
    Shared by the app and the evaluation scripts so both use exactly the same rule.
    """
    if s["status"] == "answered":
        return {"answer": s["answer"], "choice_index": s["choice_index"], "status": "graph_answer", "source": "graph_query",
                "evidence": [{"id": None, **e} for e in s["evidence"]], "answer_nodes": s["answer_nodes"],
                "method": "Deterministic NetworkX query over the extracted graph (E3)"}
    if needs_fallback(s, abstain_on_query_abstain) and f is not None:
        status = f["status"]
        if status == "unsupported" and not graph.edges:
            status = "image_only"  # nothing to cite: no arrows were extracted
        return {"answer": f["answer"], "choice_index": f["choice_index"], "status": status, "source": "model_fallback",
                "evidence": f["evidence"], "answer_nodes": f["answer_nodes"], "method": f["method"]}
    return {"answer": s["answer"], "choice_index": None, "status": "abstained", "source": "graph_query",
            "evidence": [], "answer_nodes": [], "method": "Deterministic NetworkX query: no arrow supports an answer"}


def needs_fallback(s: dict, abstain_on_query_abstain: bool = True) -> bool:
    if s["status"] == "answered":
        return False
    return not (s["status"] == "abstained" and abstain_on_query_abstain)


def run_e3(vlm, image, graph: DiagramGraph, question: str, choices: list[str] | None = None,
           explain: bool = True, abstain_on_query_abstain: bool = True) -> Iterator[dict]:
    """Graph query first; the model's own (citation-checked) answer only if the query cannot apply."""
    t0 = time.perf_counter()
    yield _ev("E3", "image", "done")
    yield _ev("E3", "extract", "done", entities=len(graph.nodes), arrows=len(graph.edges), source=graph.source)
    yield _ev("E3", "parse", "running")
    s = structured_answer(question, graph, choices, parser=vlm)
    yield _ev("E3", "parse", "done", parser=s["parser"], query=s["query"], latency_s=s["parse_latency_s"])
    if s["status"] in ("answered", "abstained", "no_option_match"):
        yield _ev("E3", "query", "done", status=s["status"], answer=s["answer"], n_evidence=len(s["evidence"]))
    else:
        yield _ev("E3", "query", "skipped", status=s["status"], answer=s["answer"])

    explanation, f = None, None
    if s["status"] == "answered" and explain and s["evidence"] and vlm is not None:
        yield _ev("E3", "explain", "running")
        x = qa_nl.explain(vlm, question, s["answer"], s["evidence"])
        explanation = x["text"]
        yield _ev("E3", "explain", "done", latency_s=x["latency_s"])
    else:
        yield _ev("E3", "explain", "skipped")
    if needs_fallback(s, abstain_on_query_abstain) and vlm is not None:
        yield _ev("E3", "fallback", "running")
        f = qa_nl.grounded_answer(vlm, image, question, graph, choices)
        yield _ev("E3", "fallback", "done", status=f["status"], latency_s=f["latency_s"])
    else:
        yield _ev("E3", "fallback", "skipped")
    res = combine_e3(s, f, graph, abstain_on_query_abstain)
    res.update(query=s, fallback=f, explanation=explanation, total_s=round(time.perf_counter() - t0, 2))
    yield _ev("E3", "answer", "done")
    yield {"pipeline": "E3", "result": res}
