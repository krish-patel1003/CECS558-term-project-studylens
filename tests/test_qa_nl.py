"""Natural-language QA: prompt building, JSON parsing, option matching and the deterministic
citation verifier. The VLM is replaced by a stub, so no model is needed."""

import pytest

from studylens import qa_nl
from studylens.schema import DiagramGraph


@pytest.fixture
def g():
    return DiagramGraph.from_dict({
        "nodes": [{"id": "n1", "label": "Grass"}, {"id": "n2", "label": "Rabbit"}, {"id": "n3", "label": "Fox"}],
        "edges": [{"source": "n1", "target": "n2"}, {"source": "n2", "target": "n3"}],
    })


class StubVLM:
    last_latency_s = 0.01

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, image, prompt, max_new_tokens=96):
        self.prompts.append(prompt)
        return self.reply


def test_prompt_numbers_arrows_and_lists_options(g):
    prompt, ids = qa_nl.build_grounded_prompt("What eats the rabbit?", g, ["Grass", "Fox"])
    assert "e1: Grass -> Rabbit" in prompt and "e2: Rabbit -> Fox" in prompt
    assert "a) Grass" in prompt and "b) Fox" in prompt and '"choice"' in prompt
    assert set(ids) == {"e1", "e2"}


def test_grounded_answer_verified(g):
    r = qa_nl.grounded_answer(StubVLM('{"answer": "Fox", "evidence": ["e2"], "graph_supported": true}'), None, "What eats the rabbit?", g)
    assert r["status"] == "grounded" and r["answer"] == "Fox"
    assert r["evidence"][0]["text"] == "Rabbit -> Fox" and r["answer_in_evidence"]


def test_hallucinated_citation_is_removed_and_not_grounded(g):
    r = qa_nl.grounded_answer(StubVLM('{"answer": "Hawk", "evidence": ["e7"], "graph_supported": true}'), None, "What eats the fox?", g)
    assert r["status"] == "unsupported"
    assert r["invalid_citations"] == ["e7"] and r["evidence"] == []


def test_model_says_not_supported(g):
    r = qa_nl.grounded_answer(StubVLM('```json\n{"answer": "green", "evidence": [], "graph_supported": false}\n```'), None, "What colour is grass?", g)
    assert r["status"] == "unsupported" and r["answer"] == "green"


def test_cannot_answer_is_abstention(g):
    r = qa_nl.grounded_answer(StubVLM('{"answer": "cannot answer", "evidence": [], "graph_supported": false}'), None, "Why?", g)
    assert r["status"] == "abstained"


def test_unparseable_output_never_grounded(g):
    r = qa_nl.grounded_answer(StubVLM("The fox, because of e2"), None, "What eats the rabbit?", g)
    assert r["parse_ok"] is False and r["status"] == "unsupported"


@pytest.mark.parametrize("reply,idx", [("b", 1), ("(b)", 1), ("b) Fox", 1), ("Fox", 1), ("The answer is Fox.", 1), ("Foxx", 1), ("e", None), ("Wolf", None)])
def test_match_choice(reply, idx):
    assert qa_nl.match_choice(reply, ["Grass", "Fox", "Sun", "Rabbit"]) == idx


def test_multiple_choice_grounded(g):
    r = qa_nl.grounded_answer(StubVLM('{"choice": "b", "evidence": ["e2"], "graph_supported": true}'), None, "What eats the rabbit?", g, ["Grass", "Fox"])
    assert r["choice_index"] == 1 and r["answer"] == "Fox" and r["status"] == "grounded"


def test_direct_answer_maps_letter(g):
    r = qa_nl.direct_answer(StubVLM("b"), None, "What eats the rabbit?", ["Grass", "Fox"])
    assert r["choice_index"] == 1 and "no graph" in r["method"]


def test_empty_graph_prompt(g):
    prompt, ids = qa_nl.build_grounded_prompt("q", DiagramGraph(), None)
    assert "(no arrows were extracted)" in prompt and ids == {}


@pytest.mark.parametrize("raw", [
    '```json\n{\n  "answer": "Fox",\n  "evidence": [e2],\n  "graph_supported": true\n}\n```',
    '{"answer": "Fox", "evidence": [e2,], "graph_supported": true}',
    '{"answer": "Fox", "evidence": [e2] "graph_supported": true}',  # missing comma -> field extraction
])
def test_unquoted_and_broken_json_is_parsed(g, raw):
    r = qa_nl.grounded_answer(StubVLM(raw), None, "What eats the rabbit?", g)
    assert r["parse_ok"] and r["answer"] == "Fox" and r["status"] == "grounded"


def test_citation_not_involving_answer_is_rejected(g):
    r = qa_nl.grounded_answer(StubVLM('{"answer": "blue", "evidence": ["e2"], "graph_supported": true}'), None, "What colour is the fox?", g)
    assert r["status"] == "unsupported" and r["grounding_note"]


def test_yes_no_answer_can_be_grounded(g):
    r = qa_nl.grounded_answer(StubVLM('{"answer": "Yes, the rabbit eats grass", "evidence": ["e1"], "graph_supported": true}'), None, "Is grass eaten?", g)
    assert r["status"] == "grounded"
