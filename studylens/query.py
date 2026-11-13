"""Question -> structured graph query -> deterministic NetworkX answer (the core of condition E3).

The proposal's design: "a deterministic graph query answers structural questions, and the
language model explains". This module turns a free-form question (any wording, optionally with
multiple-choice options) into ONE structured operation over the extracted graph and executes it
with NetworkX. The answer therefore comes from the graph, never from the language model.

Parsing, in order:
1. keyword rules (``graph_qa.rule_parse``), no model;
2. if the rules do not recognise the question, Qwen2.5-VL (text only, base weights) acts ONLY as a
   semantic parser: it must output {"op", "x", "y", "negate"} JSON. It never sees the arrows,
   so it cannot answer the question itself.

Everything after parsing is deterministic: entity names are matched to graph nodes (exact, then
fuzzy); the operation runs on the graph; multiple-choice options are matched to the result. If
any step fails (not a structural question, unknown entity, no option matches, or the graph has no
supporting arrow), the result says so and the caller may fall back to the model's own answer,
which is then clearly labelled as NOT coming from the graph query.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from . import graph_qa
from .graph_qa import ANSWERED, NEEDS_X, NEEDS_Y, OPS
from .schema import DiagramGraph

PARSE_PROMPT = """Convert a question about a diagram into ONE graph operation. Do not answer the question.

The diagram's entities: {entities}

Operations (an arrow A -> B means the diagram draws an arrow from A to B):
- successors(x): what x points to directly
- predecessors(x): what points to x directly
- descendants(x): everything reachable from x by following arrows
- ancestors(x): everything that leads to x
- edge(x, y): is there an arrow x -> y
- path(x, y): is there a chain of arrows from x to y
- count_in(x) / count_out(x): number of arrows into / out of x
- sources: entities with no incoming arrows (start points; in a food web: producers)
- sinks: entities with no outgoing arrows (end points; in a food web: top predators)
- none: the question is not about arrows or connections (e.g. colour, shape, a definition, what a label means)

Arrow conventions: in food chains and food webs arrows go from the food to the eater (direction of energy flow), so "what eats x" = successors(x), "what does x eat" = predecessors(x), "which organism would be affected if x disappeared" = successors(x). In cycles and processes arrows go from a stage to the next one, so "what comes after x" = successors(x) and "what comes before x" = predecessors(x).
Use entity names exactly as listed. Set "negate" to true if the question asks which option is NOT / does NOT.

Examples (entities from other diagrams):
"What does the frog eat?" -> {{"op": "predecessors", "x": "frog", "y": null, "negate": false}}
"Which organism eats the frog?" -> {{"op": "successors", "x": "frog", "y": null, "negate": false}}
"Which of these is NOT eaten by the hawk?" -> {{"op": "predecessors", "x": "hawk", "y": null, "negate": true}}
"Does energy flow from grass to the snake?" -> {{"op": "path", "x": "grass", "y": "snake", "negate": false}}
"What stage comes after the pupa?" -> {{"op": "successors", "x": "pupa", "y": null, "negate": false}}
"Which organisms get energy from the sun, directly or indirectly?" -> {{"op": "descendants", "x": "sun", "y": null, "negate": false}}
"What is the function of the leaf?" -> {{"op": "none", "x": "leaf", "y": null, "negate": false}}

Question: {question}
{options}Reply with ONLY JSON: {{"op": "...", "x": "<entity or null>", "y": "<entity or null>", "negate": false}}"""


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def build_parse_prompt(question: str, graph: DiagramGraph, choices: list[str] | None = None) -> str:
    ents = ", ".join(n.label for n in graph.nodes) or "(none)"
    opts = ""
    if choices:
        opts = "Options: " + "; ".join(choices) + "\n"
    return PARSE_PROMPT.format(entities=ents, question=question, options=opts)


def resolve_entity(name: Any, graph: DiagramGraph) -> str | None:
    """Map a name to a node id: exact (normalised), then unique containment, then fuzzy."""
    if name is None:
        return None
    key = _norm(name)
    if not key or key in ("null", "none"):
        return None
    labels = {_norm(n.label): n.id for n in graph.nodes}
    key = re.sub(r"^(the|a|an) ", "", key)
    for k in (key, re.sub(r"(es|s)$", "", key), key + "s"):  # "rabbits" -> "rabbit"
        if k in labels:
            return labels[k]
    contains = [nid for lab, nid in labels.items() if lab and (re.search(rf"\b{re.escape(lab)}\b", key) or re.search(rf"\b{re.escape(key)}\b", lab))]
    if len(set(contains)) == 1:
        return contains[0]
    close = difflib.get_close_matches(key, labels.keys(), n=1, cutoff=0.8)
    return labels[close[0]] if close else None


def parse_llm_query(raw: str, graph: DiagramGraph) -> dict | None:
    """Validate the parser's JSON; entity names are resolved deterministically."""
    from .qa_nl import parse_json_obj  # tolerant JSON parsing shared with the QA prompts

    txt = re.sub(r"```(?:json)?", "", raw or "")
    first = re.search(r"\{[^{}]*\}", txt)  # the model sometimes emits several objects: keep the first
    obj = None
    if first:
        try:
            obj = json.loads(first.group(0).replace("None", "null").replace("True", "true").replace("False", "false"))
        except json.JSONDecodeError:
            obj = parse_json_obj(first.group(0))
    if not isinstance(obj, dict):
        return None
    op = _norm(obj.get("op", "")).replace(" ", "_")
    out = {"op": op, "x_name": obj.get("x"), "y_name": obj.get("y"), "negate": bool(obj.get("negate", False))}
    if op not in OPS:
        out["op"] = "none"
        return out
    out["x"] = resolve_entity(obj.get("x"), graph)
    out["y"] = resolve_entity(obj.get("y"), graph) if op in NEEDS_Y else None
    return out


# ------------------------------------------------------------------ food-web wording rules
# Deterministic rules for the most common food-web phrasings (AI2D's largest category). Arrows in
# food webs point from the food to the eater, so "what eats X" asks for X's successors.
_FW_PRED = [r"\bENTITY (gets?|receives?|obtains?) (its |their )?(energy|food) from\b", r"\bdoes ENTITY (eat|consume|feed on|prey on|hunt)\b", r"\bENTITY (eats?|consumes?|feeds? on|preys? on|hunts?)\b",
            r"\b(eaten|consumed|hunted) by (the )?ENTITY\b", r"\b(prey|food|food sources?) (of|for) (the )?ENTITY\b"]
_FW_SUCC = [r"\b(eats?|consumes?|feeds? on|preys? on|hunts?) (the )?ENTITY\b", r"\b(gets?|receives?|obtains?) (its |their )?(energy|food) from (the )?ENTITY\b", r"\bENTITY (is|are) (eaten|consumed|hunted|preyed on)\b",
            r"\b(predators?|consumers?|eaters?) (of|for) (the )?ENTITY\b"]


def food_web_parse(question: str, graph: DiagramGraph) -> dict | None:
    """{"op", "x", "y"} for one-entity food-web questions ("what eats X", "what does X eat"), else None."""
    ids = [nid for _, nid in graph_qa.find_entities(question, graph)]
    if len(ids) != 1:
        return None
    qm = graph_qa._mask(graph_qa._norm(question), graph.label_of(ids[0]))
    if any(re.search(p, qm) for p in _FW_PRED):
        return {"op": "predecessors", "x": ids[0], "y": None}
    if any(re.search(p, qm) for p in _FW_SUCC):
        return {"op": "successors", "x": ids[0], "y": None}
    return None


# ------------------------------------------------------------------ multiple-choice matching
_NUM_WORDS = {"zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10}


def _option_node(opt: str, graph: DiagramGraph) -> str | None:
    """The graph node an option names (exact or unique containment), else None."""
    k = _norm(opt)
    labels = {_norm(n.label): n.id for n in graph.nodes}
    if k in labels:
        return labels[k]
    hits = {nid for lab, nid in labels.items() if lab and (re.search(rf"\b{re.escape(lab)}\b", k) or re.search(rf"\b{re.escape(k)}\b", lab))}
    return hits.pop() if len(hits) == 1 else None


def match_options(op: str, res: graph_qa.QAResult, choices: list[str], graph: DiagramGraph, negate: bool = False) -> int | None:
    """Pick the unique option consistent with the query result, or None (no deterministic choice)."""
    if res.status != ANSWERED:
        return None
    norms = [_norm(c) for c in choices]
    if op in ("edge", "path"):
        yes = res.answer.lower().startswith("yes")
        want = {"yes", "true"} if yes != negate else {"no", "false"}
        hits = [i for i, c in enumerate(norms) if c in want]
        return hits[0] if len(hits) == 1 else None
    if op in ("count_in", "count_out"):
        n = len(res.evidence)
        hits = [i for i, c in enumerate(norms) if (c.isdigit() and int(c) == n) or _NUM_WORDS.get(c) == n]
        return hits[0] if len(hits) == 1 else None
    answer = set(res.answer_nodes)
    nodes = [_option_node(c, graph) for c in choices]
    if negate:
        hits = [i for i, nid in enumerate(nodes) if nid is not None and nid not in answer]
    else:
        hits = [i for i, nid in enumerate(nodes) if nid is not None and nid in answer]
    return hits[0] if len(hits) == 1 else None


# ------------------------------------------------------------------ main entry point
def structured_answer(question: str, graph: DiagramGraph, choices: list[str] | None = None, parser=None) -> dict:
    """Answer from the graph with a deterministic query.

    ``parser`` is an object with ``generate_text(prompt, max_new_tokens)`` (the Qwen extractor);
    it is only used when the keyword rules do not recognise the question. Returns a dict with
    ``status``: "answered" (the graph query produced the answer), "abstained" (the query ran but
    the graph has no supporting arrow), or "not_structural" / "unparsed" / "unknown_entity" /
    "no_option_match" (the graph query cannot answer; the caller may fall back).
    """
    out: dict[str, Any] = {"parser": None, "query": None, "status": None, "answer": None, "choice_index": None,
                           "evidence": [], "answer_nodes": [], "parse_raw": None, "parse_latency_s": None}
    q = graph_qa.rule_parse(question, graph)
    negate = False
    if q is not None:
        out["parser"] = "rules"
    elif (q := food_web_parse(question, graph)) is not None:
        out["parser"] = "rules (food-web wording)"
    if q is not None:
        negate = bool(choices) and bool(re.search(r"\b(not|except)\b", question.lower()))
    elif parser is not None:
        raw = parser.generate_text(build_parse_prompt(question, graph, choices), max_new_tokens=60)
        out["parse_raw"] = raw
        out["parse_latency_s"] = getattr(parser, "last_latency_s", None)
        out["parser"] = "qwen (parser only)"
        q = parse_llm_query(raw, graph)
        if q is None:
            out["status"] = "unparsed"
            out["answer"] = "The question could not be turned into a graph query."
            return out
        negate = q.get("negate", False) or (bool(choices) and bool(re.search(r"\b(not|except)\b", question.lower())))
    else:
        out["status"] = "unparsed"
        out["answer"] = "The keyword rules did not recognise this question (no parser model loaded)."
        return out

    op = q["op"]
    out["query"] = {"op": op, "x": graph.label_of(q["x"]) if q.get("x") else q.get("x_name"),
                    "y": graph.label_of(q["y"]) if q.get("y") else q.get("y_name"), "negate": negate}
    if op == "none":
        out["status"] = "not_structural"
        out["answer"] = "This question is not about the diagram's arrows, so the graph query does not apply."
        return out
    if (op in NEEDS_X and q.get("x") is None) or (op in NEEDS_Y and q.get("y") is None):
        out["status"] = "unknown_entity"
        out["answer"] = "The question names an entity that is not in the extracted graph."
        return out
    res = graph_qa.execute(op, graph, q.get("x"), q.get("y"))
    out["evidence"] = res.evidence
    out["answer_nodes"] = res.answer_nodes
    out["answer"] = res.answer
    out["status"] = "answered" if res.status == ANSWERED else "abstained"
    if choices and res.status == ANSWERED:
        ci = match_options(op, res, choices, graph, negate)
        out["choice_index"] = ci
        if ci is None:
            out["status"] = "no_option_match"
        else:
            out["answer"] = choices[ci]
    return out
