"""Diagram Scene Graph pieces, reference linking, grounding gold/scoring and the Explain pipeline.
No model weights or OCR models are needed (stubs and synthetic images)."""

from PIL import Image, ImageDraw

from studylens import pipelines
from studylens.datasets.ai2d import gold_locations, location_hit
from studylens.dsg import group_parts, number_texts, parse_arrows
from studylens.explain import build_references, link_mentions
from studylens.regions import binarise, enclosing_box
from studylens.schema import DiagramGraph


def boxed_image():
    img = Image.new("RGB", (400, 200), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 60, 140, 120], outline="black", width=3)  # box around "Client"
    d.rectangle([250, 50, 380, 130], outline="black", width=3)  # box around two lines "Load" / "Balancer"
    return img


def test_enclosing_box_finds_shape_and_rejects_free_text():
    ink = binarise(boxed_image())
    b = enclosing_box(ink, [50, 82, 110, 98])
    assert b is not None and abs(b[0] - 20) <= 4 and abs(b[2] - 140) <= 4 and abs(b[1] - 60) <= 4
    assert enclosing_box(ink, [180, 5, 240, 20]) is None  # text in open space


def test_group_parts_merges_lines_in_one_shape_and_filters_noise():
    img = boxed_image()
    lines = [{"text": "Client", "score": 0.99, "box": [50, 82, 110, 98]},
             {"text": "Load", "score": 0.99, "box": [290, 65, 340, 85]},
             {"text": "Balancer", "score": 0.99, "box": [280, 92, 350, 112]},
             {"text": "V", "score": 0.6, "box": [190, 140, 200, 150]},     # arrowhead misread as text
             {"text": "-", "score": 0.9, "box": [200, 160, 205, 165]},
             {"text": "Web app", "score": 0.99, "box": [150, 2, 250, 16]}]  # title (top, outside shapes)
    texts = number_texts(lines)
    assert {t["text"] for t in texts} == {"Client", "Load", "Balancer", "Web app"}
    parts, title = group_parts(texts, binarise(img), img.size)
    labels = sorted(p["label"] for p in parts)
    assert labels == ["Client", "Load Balancer"]
    assert next(t["text"] for t in texts if t["id"] == title) == "Web app"
    assert all(p["shape"] is not None for p in parts)


def test_parse_arrows_by_name_fuzzy_and_unknown():
    parts = [{"label": "Client"}, {"label": "Load Balancer"}, {"label": "Server"}]
    raw = ('{"arrows": [{"from": "Client", "to": "Load balancer"}, {"from": "Load Balancr", "to": "Server"}, '
           '{"from": "Client", "to": "Database"}, {"from": "Server", "to"')
    arrows, warnings = parse_arrows(raw, parts)
    assert arrows == [(0, 1), (1, 2)]
    assert any("Database" in w for w in warnings) and any("cut off" in w for w in warnings)


def chain_graph():
    return DiagramGraph.from_dict({
        "nodes": [{"id": "n1", "label": "Algae", "bbox": [0, 0, 10, 10]}, {"id": "n2", "label": "Water flea", "bbox": [20, 0, 30, 10]},
                  {"id": "n3", "label": "Small fish", "bbox": [40, 0, 50, 10]}, {"id": "n4", "label": "A", "bbox": [60, 0, 70, 10]}],
        "edges": [{"source": "n1", "target": "n2"}, {"source": "n2", "target": "n3"}]})


def test_link_mentions_case_plural_and_letter_labels():
    g = chain_graph()
    found = [nid for _, _, nid in link_mentions("Water fleas eat ALGAE; a small fish eats them. Label A marks it.", g)]
    assert found == ["n2", "n1", "n3", "n4"]  # "a" (article) is not the label "A"


def test_build_references_parts_and_flows():
    g = chain_graph()
    text, refs, stats = build_references("The algae are eaten by the water flea, which the small fish eats.", g)
    kinds = [(r["kind"], r["label"]) for r in refs]
    assert ("part", "Algae") in kinds and ("flow", "Algae → Water flea → Small fish") in kinds
    assert text.count("[[") == stats["n_refs"]
    s = {"status": "answered", "answer": "x", "evidence": [{"source": "n1", "target": "n2", "text": "Algae -> Water flea"}]}
    _, refs2, stats2 = build_references("No names here.", g, s)
    assert stats2["facts_ref"] == 1 and refs2[0]["from_query"]


def test_gold_locations_and_hit():
    ann = {"text": {"T0": {"rectangle": [[0, 0], [20, 10]], "value": "Aorta"}, "T1": {"rectangle": [[100, 100], [130, 110]], "value": "Heart"}},
           "arrows": {"A0": {"polygon": [[22, 5], [60, 40], [62, 42]]}},
           "blobs": {"B0": {"polygon": [[50, 30], [90, 30], [90, 80], [50, 80]]}},
           "relationships": {"r1": {"category": "intraObjectLinkage", "origin": "T0", "destination": "B0", "connector": "A0"},
                             "r2": {"category": "intraObjectLabel", "origin": "T1", "destination": "B0"}}}
    g = gold_locations(ann)
    assert g["aorta"][0]["type"] == "point" and g["aorta"][0]["xy"] == (62.0, 42.0)
    assert g["heart"][0]["box"] == [50, 30, 90, 80]
    assert location_hit([55, 35, 65, 45], g["aorta"], (200, 200))
    assert not location_hit([150, 150, 160, 160], g["aorta"], (200, 200))
    assert location_hit([60, 40, 80, 70], g["heart"], (200, 200))


def test_run_explain_with_stub_model():
    class Stub:
        last_latency_s = 0.01

        def generate_text(self, prompt, max_new_tokens=96):
            return '{"op": "none", "x": null, "y": null, "negate": false}'

        def generate(self, image, prompt, max_new_tokens=96):
            assert "Water flea" in prompt and "Algae -> Water flea" in prompt
            return "Energy goes from the algae to the water flea and then to the small fish."

    ev = list(pipelines.run_explain(Stub(), Image.new("RGB", (80, 20)), chain_graph(), "Explain the energy flow"))
    stages = [(e["stage"], e["state"]) for e in ev if "stage" in e]
    assert ("explain", "done") in stages and ("check", "done") in stages
    res = ev[-1]["result"]
    assert res["status"] == "explained" and any(r["kind"] == "flow" for r in res["references"])
    assert pipelines.collect(pipelines.run_explain(None, Image.new("RGB", (8, 8)), chain_graph(), "q"))["status"] == "error"
