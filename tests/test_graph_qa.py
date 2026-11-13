import pytest

from studylens.graph_qa import ABSTAINED, ANSWERED, answer_question
from studylens.schema import DiagramGraph


@pytest.fixture
def g():
    # A -> B -> C -> D, plus E -> C ; "Light reactions" tests multi-word / overlapping labels
    return DiagramGraph.from_dict({
        "nodes": [{"id": i, "label": l} for i, l in
                  [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "Enzyme"), ("l", "Light reactions"), ("m", "Light")]],
        "edges": [{"source": s, "target": t} for s, t in
                  [("a", "b"), ("b", "c"), ("c", "d"), ("e", "c"), ("m", "l")]],
    })


@pytest.mark.parametrize("q", [
    "Which component directly feeds into C?",
    "What points to C?",
    "What are the inputs to C?",
    "What comes before C?",
    "What is C fed by?",
    "Where does C come from?",
])
def test_predecessors_paraphrases(g, q):
    r = answer_question(q, g)
    assert r.status == ANSWERED, r.answer
    assert set(r.answer_nodes) == {"b", "e"}
    assert {(e["source"], e["target"]) for e in r.evidence} == {("b", "c"), ("e", "c")}


@pytest.mark.parametrize("q", ["What does B point to?", "What does B feed into?", "What comes after B?", "What are the outputs of B?"])
def test_successors_paraphrases(g, q):
    r = answer_question(q, g)
    assert r.status == ANSWERED and r.answer_nodes == ["c"], r.answer


def test_upstream_is_transitive(g):
    r = answer_question("What is upstream of D?", g)
    assert set(r.answer_nodes) == {"a", "b", "c", "e"}


def test_edge_check_yes_no_and_reverse(g):
    assert answer_question("Does B point to C?", g).answer.startswith("Yes")
    r = answer_question("Does C point to B?", g)
    assert r.answer.startswith("No") and r.evidence[0]["source"] == "b"  # cites the opposite arrow
    assert answer_question("Is there an arrow between A and D?", g).answer.startswith("No")


def test_path(g):
    r = answer_question("Is there a path from A to D?", g)
    assert r.answer.startswith("Yes") and len(r.evidence) == 3
    assert answer_question("Is there a path from D to A?", g).answer.startswith("No")


def test_count(g):
    assert answer_question("How many arrows point into C?", g).answer.startswith("2")
    assert answer_question("How many arrows leave C?", g).answer.startswith("1")


def test_longest_label_wins(g):
    r = answer_question("What feeds into Light reactions?", g)
    assert r.answer_nodes == ["m"]


def test_fuzzy_entity_match(g):
    r = answer_question("What does Enzime point to?", g)
    assert r.status == ANSWERED and r.answer_nodes == ["c"]


@pytest.mark.parametrize("q", [
    "Which component directly feeds into A?",  # A has no incoming arrow -> must not invent one
    "What colour is B?",                       # not structural
    "Which component feeds into Mitochondria?",  # entity not in graph
    "",
])
def test_abstains(g, q):
    r = answer_question(q, g)
    assert r.status == ABSTAINED
    assert r.evidence == []


def test_sources_and_sinks(g):
    assert set(answer_question("Where does the flow start?", g).answer_nodes) == {"a", "e", "m"}
    assert set(answer_question("Which entities are end points?", g).answer_nodes) == {"d", "l"}


def test_invalid_edges_dropped():
    g = DiagramGraph.from_dict({"nodes": [{"id": "x", "label": "X"}],
                                "edges": [{"source": "x", "target": "zz"}, {"source": "x", "target": "x"}]})
    assert g.edges == [] and len(g.warnings) == 2
