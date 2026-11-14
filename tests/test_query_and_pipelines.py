"""Graph-query engine (E3), the three pipelines with stub models, the streaming endpoint, and scoring.
No model weights or OCR models are needed."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studylens import app as app_mod
from studylens import pipelines
from studylens.evalkit import first_number, score_e3, score_text
from studylens.graph_qa import execute, rule_parse
from studylens.ocr import OCRClient, OCRUnavailable
from studylens.query import food_web_parse, match_options, parse_llm_query, resolve_entity, structured_answer
from studylens.schema import DiagramGraph


def food_web() -> DiagramGraph:
    return DiagramGraph.from_dict({
        "nodes": [{"id": f"n{i}", "label": l} for i, l in enumerate(["Grass", "Rabbit", "Deer", "Fox", "Wolf"], 1)],
        "edges": [{"source": s, "target": t} for s, t in [("n1", "n2"), ("n1", "n3"), ("n2", "n4"), ("n3", "n5")]],
    })


class StubVLM:
    """Records prompts; returns canned text for the parser, QA and explanation calls."""

    is_ai = True

    def __init__(self, parse_reply='{"op": "none", "x": null, "y": null, "negate": false}'):
        self.parse_reply = parse_reply
        self.last_latency_s = 0.01
        self.calls = []

    def generate_text(self, prompt, max_new_tokens=96):
        self.calls.append(("text", prompt))
        if "Convert a question" in prompt:
            return self.parse_reply
        if "Explain this answer" in prompt:
            return "Because the arrows show it."
        return "a"  # E1 answer

    def generate(self, image, prompt, max_new_tokens=96):
        self.calls.append(("image", prompt))
        if "Reply with ONLY a JSON object" in prompt:
            return '{"answer": "Fox", "evidence": ["e3"], "graph_supported": true}'
        return "Fox."


# ------------------------------------------------------------------ rules and execution
@pytest.mark.parametrize("q, op", [
    ("Which entities lead to Fox, directly or indirectly?", "ancestors"),
    ("Name everything reachable from Grass by following the arrows.", "descendants"),
    ("Where do the arrows from Grass go?", "successors"),
    ("Where do the arrows into Fox come from?", "predecessors"),
    ("Can you get from Grass to Wolf by following the arrows?", "path"),
    ("Which entities have no incoming arrows?", "sources"),
    ("Which boxes are end points, with no arrows leaving them?", "sinks"),
    ("What does Rabbit point to directly?", "successors"),
])
def test_rule_parse_operations(q, op):
    assert rule_parse(q, food_web())["op"] == op


def test_food_web_wording_is_left_to_food_web_rules():
    g = food_web()
    assert rule_parse("What does the Fox feed on?", g) is None  # "feed on" reverses the generic reading
    assert food_web_parse("What does the Fox feed on?", g) == {"op": "predecessors", "x": "n4", "y": None}
    assert food_web_parse("What eats the Rabbit?", g) == {"op": "successors", "x": "n2", "y": None}
    assert food_web_parse("Which organism gets energy from Grass?", g)["op"] == "successors"


def test_execute_transitive_and_abstain():
    g = food_web()
    r = execute("descendants", g, "n1")
    assert {g.label_of(n) for n in r.answer_nodes} == {"Rabbit", "Deer", "Fox", "Wolf"}
    assert execute("predecessors", g, "n1").status == "abstained"  # nothing points to Grass
    assert execute("successors", g, None).status == "abstained"  # entity missing


# ------------------------------------------------------------------ LLM parser output (validated deterministically)
def test_parse_llm_query_keeps_first_object_and_resolves_entities():
    g = food_web()
    raw = '{"op": "predecessors", "x": "fox", "y": "Rabbit", "negate": false} {"op": "predecessors", "x": "fox"}'
    q = parse_llm_query(raw, g)
    assert q["op"] == "predecessors" and q["x"] == "n4" and q["y"] is None  # y ignored for a one-entity op
    assert parse_llm_query('{"op": "teleport", "x": "Fox"}', g)["op"] == "none"
    assert parse_llm_query("no json here", g) is None


def test_resolve_entity_fuzzy_and_unknown():
    g = food_web()
    assert resolve_entity("the rabbits", g) == "n2"
    assert resolve_entity("Rabit", g) == "n2"
    assert resolve_entity("Eagle", g) is None


# ------------------------------------------------------------------ multiple choice
def test_match_options_lists_negation_yes_no_and_counts():
    g = food_web()
    res = execute("predecessors", g, "n4")  # Rabbit -> Fox
    assert match_options("predecessors", res, ["Deer", "Rabbit", "Wolf"], g) == 1
    assert match_options("predecessors", res, ["Rabbit", "Grass"], g, negate=True) == 1
    assert match_options("predecessors", res, ["sun", "rain"], g) is None  # no option names a graph node
    assert match_options("edge", execute("edge", g, "n1", "n2"), ["no", "yes"], g) == 1
    assert match_options("count_out", execute("count_out", g, "n1"), ["1", "2", "3"], g) == 1


def test_structured_answer_with_parser_and_options():
    g = food_web()
    vlm = StubVLM('{"op": "successors", "x": "Rabbit", "y": null, "negate": false}')
    s = structured_answer("Which animal hunts down the rabbit in this picture?", g, ["Wolf", "Fox"], parser=vlm)
    assert s["parser"] == "qwen (parser only)" and s["status"] == "answered" and s["choice_index"] == 1
    assert [c for c in vlm.calls if c[0] == "text"] and "Grass" in vlm.calls[0][1]  # parser sees entity names
    assert "Rabbit -> Fox" not in vlm.calls[0][1]  # ...but never the arrows
    s = structured_answer("What colour is the fox?", g, None, parser=StubVLM())
    assert s["status"] == "not_structural"


# ------------------------------------------------------------------ E3 decision rule
def test_combine_e3_policies():
    g = food_web()
    s_abst = {"status": "abstained", "answer": "no arrow", "evidence": [], "answer_nodes": [], "choice_index": None}
    f = {"status": "grounded", "answer": "Fox", "choice_index": None, "evidence": [{"id": "e3", "source": "n2", "target": "n4", "text": "Rabbit -> Fox"}],
         "answer_nodes": ["n2", "n4"], "method": "m"}
    assert pipelines.combine_e3(s_abst, f, g)["status"] == "abstained"
    assert pipelines.combine_e3(s_abst, f, g, abstain_on_query_abstain=False)["source"] == "model_fallback"
    s_ns = {**s_abst, "status": "not_structural"}
    empty = DiagramGraph.from_dict({"nodes": [{"id": "n1", "label": "A"}], "edges": []})
    assert pipelines.combine_e3(s_ns, {**f, "status": "unsupported"}, empty)["status"] == "image_only"


def test_run_pipelines_with_stub_models():
    g, img = food_web(), Image.new("RGB", (100, 80), "white")
    vlm = StubVLM()

    class StubOCR:
        last_latency_s = 0.0

        def read(self, image):
            return [{"text": "Fox", "score": 1.0, "box": [1, 2, 30, 12]}]

    ev = list(pipelines.run_e3(vlm, img, g, "What eats the Rabbit?"))
    stages = [(e["stage"], e["state"]) for e in ev if "stage" in e]
    assert ("query", "done") in stages and ("explain", "done") in stages and ("fallback", "skipped") in stages
    res = ev[-1]["result"]
    assert res["status"] == "graph_answer" and res["answer_nodes"] == ["n4"] and res["explanation"]
    res2 = pipelines.collect(pipelines.run_e3(vlm, img, g, "What colour is the fox?"))
    assert res2["source"] == "model_direct" and res2["status"] == "image_only"
    res3 = pipelines.collect(pipelines.run_e3(vlm, img, g, "What colour is the fox?", policy="v1"))
    assert res3["source"] == "model_fallback" and res3["status"] == "grounded"
    r1 = pipelines.collect(pipelines.run_e1(vlm, StubOCR(), img, "What eats grass?", ["Fox", "Rabbit"]))
    assert r1["choice_index"] == 0 and r1["ocr_lines"][0]["text"] == "Fox"
    assert pipelines.collect(pipelines.run_e2(None, img, "q"))["status"] == "error"


# ------------------------------------------------------------------ streaming endpoint (no model: rules only)
def _png(size=(120, 90)):
    b = io.BytesIO()
    Image.new("RGB", size, "white").save(b, format="PNG")
    return b.getvalue()


def test_ask_stream_runs_three_pipelines(monkeypatch):
    class StubOCR:
        last_latency_s = 0.0

        def available(self):
            return True

        def read(self, image):
            return [{"text": "Grass", "score": 1.0, "box": [0, 0, 10, 10]}]

    monkeypatch.setattr(app_mod, "get_ocr", lambda: StubOCR())
    c = TestClient(app_mod.app)
    r = c.post("/api/ask_stream", files={"file": ("d.png", _png(), "image/png")},
               data={"question": "What eats the Rabbit?", "graph": json.dumps(food_web().to_dict())})
    assert r.status_code == 200
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    assert lines[0]["order"] == ["E3", "E2", "E1"] and lines[-1] == {"done": True}
    results = {l["pipeline"]: l["result"] for l in lines if "result" in l}
    assert results["E3"]["status"] == "graph_answer" and results["E3"]["answer_nodes"] == ["n4"]
    assert results["E2"]["status"] == "error"  # no model loaded in tests
    assert results["E1"]["ocr_lines"][0]["text"] == "Grass"


def test_ask_stream_rejects_bad_input():
    c = TestClient(app_mod.app)
    r = c.post("/api/ask_stream", files={"file": ("d.png", b"not an image", "image/png")},
               data={"question": "q", "graph": json.dumps(food_web().to_dict())})
    assert r.status_code == 400


def test_ocr_client_unavailable():
    with pytest.raises(OCRUnavailable):
        OCRClient(python="/nonexistent/python").read(Image.new("RGB", (10, 10)))


# ------------------------------------------------------------------ scoring
def test_scoring_helpers():
    labels = ["Grass", "Rabbit", "Fox"]
    assert score_text("The fox eats the rabbit.", {"type": "set", "value": ["Fox"]}, ["Rabbit"], labels)
    assert score_text("Not shown in the diagram.", {"type": "abstain"}, [], labels)
    assert score_text("Two arrows.", {"type": "count", "value": 2}, [], labels)
    assert first_number("There are 3 arrows") == 3
    g = food_web()
    res = pipelines.combine_e3({"status": "answered", "answer": "x", "choice_index": None,
                                "evidence": [], "answer_nodes": ["n4"]}, None, g)
    assert score_e3(res, {"type": "set", "value": ["Fox"]}, ["Rabbit"], labels, g)
    res = pipelines.combine_e3({"status": "abstained", "answer": "x", "choice_index": None, "evidence": [], "answer_nodes": []}, None, g)
    assert score_e3(res, {"type": "abstain"}, [], labels, g)
