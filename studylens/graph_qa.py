"""Deterministic, graph-grounded question answering (condition E3 in the proposal).

This module contains NO machine learning. It maps a structural question onto a
NetworkX query over the extracted ``DiagramGraph`` and returns the answer together
with the exact edges that support it. If the question cannot be mapped, an entity
cannot be found, or the graph contains no supporting edge, it ABSTAINS instead of
guessing.

Arrow semantics are generic: an edge A -> B means "A points to / feeds into B".
Domain-specific meanings (e.g. "eaten by" in a food web) are not interpreted.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

from .schema import DiagramGraph, Edge

ANSWERED = "answered"
ABSTAINED = "abstained"

_VERBS = r"(feed|feeds|fed|point|points|pointed|flow|flows|lead|leads|go|goes|connect|connects|connected|send|sends|sent|arrow|arrows|link|links|linked|output|outputs|input|inputs)"
_PRED_HINTS = re.compile(
    r"\b(before|precede|precedes|preceding|upstream|receive|receives|comes? from|come into|depends? on|inputs? (of|to|for)|sources? of|fed by|pointed to by)\b"
)
_SUCC_HINTS = re.compile(r"\b(after|follow|follows|following|downstream|outputs? of|results? of|targets? of)\b")
_TRANSITIVE = re.compile(
    r"\b(indirectly|eventually|ultimately|all (the )?(upstream|downstream|ancestors|descendants)|upstream|downstream|ancestors?|descendants?|anywhere|reachable|everything)\b"
)
_NO_IN = re.compile(r"\b(no (incoming|inbound) arrows?|no arrows? (coming|pointing|going) (in|into)|nothing points (to|into) (them|it))\b")
_NO_OUT = re.compile(r"\b(no (outgoing|outbound) arrows?|no arrows? (leaving|going out|coming out)|end points?)\b")
_ARROWS_FROM = re.compile(r"\b(arrows?|edges?|links?) (from|out of|leaving) ENTITY\b")
_ARROWS_INTO = re.compile(r"\b(arrows?|edges?|links?) (into|to|towards|entering|ending at) ENTITY\b")
_PATH = re.compile(r"\b(path|reach|reachable|route|eventually|ultimately|indirectly|chain|get from|follow(ing)? the arrows)\b")
_YESNO = re.compile(r"^\s*(does|do|is|are|can|could|will|did|was|has)\b")
_COUNT = re.compile(r"\bhow many\b")
_OUT_HINTS = re.compile(r"\b(leave|leaves|leaving|out of|outgoing|outputs?|from ENTITY|does ENTITY (point|feed|lead|send|go|flow))\b")
_START = re.compile(r"\b(start|starts|starting|begin|begins|beginning|source|sources|first|origin|entry)\b")
_END = re.compile(r"\b(end|ends|ending|final|last|sink|sinks|terminal|exit)\b")


@dataclass
class QAResult:
    status: str  # "answered" | "abstained"
    answer: str
    intent: str | None = None
    entities: list[str] = field(default_factory=list)  # node ids mentioned in the question
    evidence: list[dict[str, Any]] = field(default_factory=list)  # supporting edges
    answer_nodes: list[str] = field(default_factory=list)  # node ids that form the answer
    method: str = "deterministic graph query (NetworkX)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Entity linking
# ---------------------------------------------------------------------------
def _norm(s: str, lower: bool = True) -> str:
    s = s.lower() if lower else s
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]+", " ", s)).strip()


# Labels that are also common English words ("A" vs the article "a") must match with exact case.
_STOPWORDS = {"a", "i", "an", "the", "in", "on", "to", "is", "it", "of", "or", "and", "at", "by", "as", "be", "do", "if"}


def find_entities(question: str, graph: DiagramGraph) -> list[tuple[int, str]]:
    """Return (position, node_id) for each entity mentioned in the question, in order.

    Exact (normalised, word-boundary) label matches are preferred, longest labels first so
    that "Light reactions" wins over "Light". Labels that are also English stopwords (e.g. a
    node called "A") only match with exact case. If nothing matches exactly, a fuzzy match on
    single words is tried (handles small typos such as "Rabit").
    """
    q = " " + _norm(question) + " "
    q_cased = " " + _norm(question, lower=False) + " "  # same character positions as q
    taken = [False] * len(q)
    found: list[tuple[int, str]] = []
    for node in sorted(graph.nodes, key=lambda n: -len(_norm(n.label))):
        lab = _norm(node.label)
        if not lab:
            continue
        hay, needle = (q_cased, _norm(node.label, lower=False)) if lab in _STOPWORDS else (q, lab)
        for m in re.finditer(r"(?<=\s)" + re.escape(needle) + r"(?=\s)", hay):
            if any(taken[m.start() : m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            found.append((m.start(), node.id))
            break
    if not found:
        labels = {_norm(n.label): n.id for n in graph.nodes}
        for m in re.finditer(r"\S+", q):
            word = m.group()
            if len(word) < 4:
                continue
            close = difflib.get_close_matches(word, labels.keys(), n=1, cutoff=0.8)
            if close and labels[close[0]] not in [f[1] for f in found]:
                found.append((m.start(), labels[close[0]]))
    return sorted(found)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def answer_question(question: str, graph: DiagramGraph) -> QAResult:
    question = (question or "").strip()
    if not question:
        return QAResult(ABSTAINED, "Please type a question.")
    if not graph.nodes:
        return QAResult(ABSTAINED, "The diagram graph is empty, so there is nothing to answer from.")

    G = graph.to_networkx()
    q = _norm(question)
    mentions = find_entities(question, graph)
    ids = [m[1] for m in mentions]

    # --- sources / sinks (no entity needed) -------------------------------
    if not ids and (_NO_IN.search(q) or _NO_OUT.search(q)):
        return execute("sources" if _NO_IN.search(q) else "sinks", graph)
    if not ids and (_START.search(q) or _END.search(q)) and re.search(r"\b(which|what|where)\b", q):
        if _START.search(q):
            nodes = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) > 0]
            ev = [e for e in graph.edges if e.source in nodes]
            return _list_answer(graph, "sources", nodes, ev, [], "starting point(s) (no incoming arrows)")
        nodes = [n for n in G.nodes if G.out_degree(n) == 0 and G.in_degree(n) > 0]
        ev = [e for e in graph.edges if e.target in nodes]
        return _list_answer(graph, "sinks", nodes, ev, [], "end point(s) (no outgoing arrows)")

    if not ids:
        return QAResult(
            ABSTAINED,
            "I could not match any entity in your question to an entity in the extracted graph, "
            "so I can't answer from the diagram. Check the entity names in the graph panel.",
            intent="unknown",
        )

    # --- two-entity questions ---------------------------------------------
    if len(ids) >= 2:
        a, b = ids[0], ids[1]
        # "from B to A" / "B is fed by A" style phrasing flips the order
        if re.search(r"\bfrom\b", q) and re.search(r"\bto\b", q):
            fpos, tpos = q.find(" from "), q.find(" to ")
            if fpos > tpos >= 0:
                a, b = b, a
        if re.search(r"\b(fed|pointed to|reached|preceded|led to|sent to) by\b", q):
            a, b = b, a
        if _PATH.search(q):
            return _path_answer(graph, G, a, b)
        if _YESNO.search(q) or re.search(r"\b(arrow|edge|connection|link)\b", q):
            return _edge_answer(graph, G, a, b)
        return QAResult(
            ABSTAINED,
            "Your question mentions several entities but I could not tell which structural relation "
            "you are asking about. Try e.g. 'Does A point to B?' or 'Is there a path from A to B?'.",
            intent="unknown",
            entities=ids,
        )

    # --- one-entity questions ---------------------------------------------
    x = ids[0]
    if _COUNT.search(q):
        outgoing = bool(_OUT_HINTS.search(_mask(q, graph.label_of(x))))
        ev = [e for e in graph.edges if (e.source == x if outgoing else e.target == x)]
        word = "leave" if outgoing else "point into"
        return QAResult(
            ANSWERED,
            f"{len(ev)} arrow(s) {word} {graph.label_of(x)} in the extracted graph.",
            intent="count_out" if outgoing else "count_in",
            entities=[x],
            evidence=_ev(graph, ev),
            answer_nodes=[e.target if outgoing else e.source for e in ev],
        )

    direction = _direction(q, graph.label_of(x))
    if direction is None:
        return QAResult(
            ABSTAINED,
            f"I found '{graph.label_of(x)}' in the graph, but your question is not one of the structural "
            "question types I can answer from arrows (what feeds into / comes from / leads to an entity, "
            "is there an arrow or path between two entities, how many arrows, start/end points). "
            "The graph does not encode other facts about it.",
            intent="unsupported",
            entities=[x],
        )

    # "directly" alone means one step; "directly or indirectly" means all reachable entities
    transitive = bool(_TRANSITIVE.search(q)) and ("indirectly" in q or not re.search(r"\bdirectly\b", q))
    if direction == "pred":
        if transitive:
            nodes = sorted(nx.ancestors(G, x), key=graph_order(graph))
            ev = [e for e in graph.edges if e.source in nodes and (e.target in nodes or e.target == x)]
            return _list_answer(graph, "ancestors", nodes, ev, [x], f"everything upstream of {graph.label_of(x)}")
        ev = [e for e in graph.edges if e.target == x]
        return _list_answer(graph, "predecessors", [e.source for e in ev], ev, [x], f"directly feeds into {graph.label_of(x)}")
    if transitive:
        nodes = sorted(nx.descendants(G, x), key=graph_order(graph))
        ev = [e for e in graph.edges if (e.source in nodes or e.source == x) and e.target in nodes]
        return _list_answer(graph, "descendants", nodes, ev, [x], f"everything downstream of {graph.label_of(x)}")
    ev = [e for e in graph.edges if e.source == x]
    return _list_answer(graph, "successors", [e.target for e in ev], ev, [x], f"{graph.label_of(x)} directly feeds into")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def graph_order(graph: DiagramGraph):
    order = {n.id: i for i, n in enumerate(graph.nodes)}
    return lambda nid: order.get(nid, 0)


def _ev(graph: DiagramGraph, edges: list[Edge]) -> list[dict[str, Any]]:
    return [{"source": e.source, "target": e.target, "relation": e.relation, "text": graph.edge_text(e)} for e in edges]


def _mask(q: str, label: str) -> str:
    """Replace the entity's text with the token ENTITY so its words can't trigger keyword hints."""
    return (" " + q + " ").replace(" " + _norm(label) + " ", " ENTITY ", 1)


def _direction(q: str, label: str) -> str | None:
    """Decide whether a one-entity question asks for predecessors ("pred") or successors ("succ")."""
    qm = _mask(q, label)
    # food-web verbs ("X feeds on", "what eats X") reverse the generic reading of "feed";
    # leave them to the food-web rules in studylens.query
    if re.search(r"\b(feeds? on|fed on|eats?|eaten|preys?|consumes?|hunts?)\b", qm):
        return None
    if _ARROWS_FROM.search(qm):
        return "succ"
    if _ARROWS_INTO.search(qm):
        return "pred"
    if _SUCC_HINTS.search(qm):
        return "succ"
    # "what does X feed into / point to / lead to" -> successors
    if re.search(r"\bdoes ENTITY (directly |eventually |indirectly |ultimately )?" + _VERBS, qm):
        return "succ"
    if _PRED_HINTS.search(qm):
        return "pred"
    verb = re.search(_VERBS, qm)
    if verb is None:
        return None
    # verb before the entity ("what feeds into X") -> pred; after ("X leads to what") -> succ
    return "pred" if verb.start() < qm.find("ENTITY") else "succ"


def _list_answer(graph, intent, nodes, ev, ents, phrase) -> QAResult:
    if not nodes:
        target = graph.label_of(ents[0]) if ents else "the diagram"
        return QAResult(
            ABSTAINED,
            f"The extracted graph has no arrow that supports an answer here ({intent} of {target}: none found). "
            "I won't guess an edge that isn't in the graph. If the diagram does show one, the extraction "
            "probably missed it: correct the graph and ask again.",
            intent=intent,
            entities=ents,
        )
    labels = [graph.label_of(n) for n in nodes]
    if ents:
        if intent in ("predecessors", "ancestors"):
            ans = f"{_join(labels)} — {phrase}."
        else:
            ans = f"{phrase}: {_join(labels)}."
    else:
        ans = f"The {phrase}: {_join(labels)}."
    return QAResult(ANSWERED, ans, intent=intent, entities=ents, evidence=_ev(graph, ev), answer_nodes=list(nodes))


def _edge_answer(graph, G, a, b) -> QAResult:
    la, lb = graph.label_of(a), graph.label_of(b)
    if G.has_edge(a, b):
        e = next(e for e in graph.edges if e.key == (a, b))
        return QAResult(ANSWERED, f"Yes. The graph has a direct arrow {la} -> {lb}.", "edge_check", [a, b], _ev(graph, [e]), [b])
    if G.has_edge(b, a):
        e = next(e for e in graph.edges if e.key == (b, a))
        return QAResult(
            ANSWERED,
            f"No. There is no arrow {la} -> {lb}; the arrow between them goes the other way ({lb} -> {la}).",
            "edge_check",
            [a, b],
            _ev(graph, [e]),
            [],
        )
    return QAResult(
        ANSWERED,
        f"No. The extracted graph has no direct arrow between {la} and {lb} in either direction.",
        "edge_check",
        [a, b],
        [],
        [],
    )


def _path_answer(graph, G, a, b) -> QAResult:
    la, lb = graph.label_of(a), graph.label_of(b)
    try:
        path = nx.shortest_path(G, a, b)
    except nx.NetworkXNoPath:
        return QAResult(
            ANSWERED,
            f"No. The extracted graph has no directed path from {la} to {lb}.",
            "path",
            [a, b],
            [],
            [],
        )
    edges = [next(e for e in graph.edges if e.key == (u, v)) for u, v in zip(path, path[1:])]
    return QAResult(
        ANSWERED,
        "Yes. Path: " + " -> ".join(graph.label_of(n) for n in path) + ".",
        "path",
        [a, b],
        _ev(graph, edges),
        path,
    )


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# Structured queries: the same NetworkX operations, called with an explicit operation and
# entity ids. Used by studylens.query, which turns free-form questions (any wording, incl.
# multiple choice) into one of these operations, either with the rules above or with an LLM
# acting ONLY as a parser.
# ---------------------------------------------------------------------------
OPS = ("successors", "predecessors", "descendants", "ancestors", "edge", "path",
       "count_in", "count_out", "sources", "sinks")
NEEDS_X = {"successors", "predecessors", "descendants", "ancestors", "count_in", "count_out", "edge", "path"}
NEEDS_Y = {"edge", "path"}

# intents produced by answer_question() -> operation names
_INTENT_TO_OP = {"predecessors": "predecessors", "successors": "successors", "ancestors": "ancestors",
                 "descendants": "descendants", "edge_check": "edge", "path": "path", "count_in": "count_in",
                 "count_out": "count_out", "sources": "sources", "sinks": "sinks"}


def rule_parse(question: str, graph: DiagramGraph) -> dict | None:
    """Map a question to {"op", "x", "y"} (node ids) with the keyword rules, or None if the
    rules do not recognise it."""
    r = answer_question(question, graph)
    op = _INTENT_TO_OP.get(r.intent or "")
    if op is None:
        return None
    ents = list(r.entities)
    return {"op": op, "x": ents[0] if ents else None, "y": ents[1] if len(ents) > 1 else None}


def execute(op: str, graph: DiagramGraph, x: str | None = None, y: str | None = None) -> QAResult:
    """Run one structural operation on the graph. Deterministic; abstains when the graph has no
    supporting arrow."""
    if op not in OPS:
        return QAResult(ABSTAINED, f"Unknown graph operation {op!r}.", intent=op)
    if not graph.nodes:
        return QAResult(ABSTAINED, "The diagram graph is empty, so there is nothing to answer from.", intent=op)
    if (op in NEEDS_X and x is None) or (op in NEEDS_Y and y is None):
        return QAResult(ABSTAINED, f"The operation {op} needs an entity that was not found in the graph.", intent=op)
    G = graph.to_networkx()
    lx = graph.label_of(x) if x else ""
    if op == "sources":
        nodes = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        return _list_answer(graph, "sources", nodes, [e for e in graph.edges if e.source in nodes], [],
                            "starting point(s) (no incoming arrows)")
    if op == "sinks":
        nodes = [n for n in G.nodes if G.out_degree(n) == 0 and G.in_degree(n) > 0]
        return _list_answer(graph, "sinks", nodes, [e for e in graph.edges if e.target in nodes], [],
                            "end point(s) (no outgoing arrows)")
    if op == "edge":
        return _edge_answer(graph, G, x, y)
    if op == "path":
        return _path_answer(graph, G, x, y)
    if op in ("count_in", "count_out"):
        out = op == "count_out"
        ev = [e for e in graph.edges if (e.source == x if out else e.target == x)]
        return QAResult(ANSWERED, f"{len(ev)} arrow(s) {'leave' if out else 'point into'} {lx} in the extracted graph.",
                        intent=op, entities=[x], evidence=_ev(graph, ev),
                        answer_nodes=[e.target if out else e.source for e in ev])
    if op == "predecessors":
        ev = [e for e in graph.edges if e.target == x]
        return _list_answer(graph, op, [e.source for e in ev], ev, [x], f"directly feeds into {lx}")
    if op == "successors":
        ev = [e for e in graph.edges if e.source == x]
        return _list_answer(graph, op, [e.target for e in ev], ev, [x], f"{lx} directly feeds into")
    if op == "ancestors":
        nodes = sorted(nx.ancestors(G, x), key=graph_order(graph))
        ev = [e for e in graph.edges if e.source in nodes and (e.target in nodes or e.target == x)]
        return _list_answer(graph, op, nodes, ev, [x], f"everything upstream of {lx}")
    nodes = sorted(nx.descendants(G, x), key=graph_order(graph))
    ev = [e for e in graph.edges if (e.source in nodes or e.source == x) and e.target in nodes]
    return _list_answer(graph, "descendants", nodes, ev, [x], f"everything downstream of {lx}")
