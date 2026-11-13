"""AI2D gold-graph construction (on a hand-written annotation, no dataset needed), evaluation
statistics, and the evaluation API."""

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from studylens import app as app_mod
from studylens.datasets.ai2d import build_graph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate_ai2d  # noqa: E402

ANN = {
    "text": {
        "T0": {"rectangle": [[0, 0], [40, 10]], "value": "Grass "},
        "T1": {"rectangle": [[100, 0], [140, 10]], "value": "Rabbit"},
        "T2": {"rectangle": [[200, 0], [240, 10]], "value": "Fox"},
        "T3": {"rectangle": [[300, 300], [340, 310]], "value": "Terrestrial"},  # far-away header
        "T4": {"rectangle": [[0, 50], [40, 60]], "value": "grass"},  # duplicate label (case)
    },
    "blobs": {"B0": {"polygon": [[90, 10], [150, 10], [150, 40], [90, 40]]},
              "B1": {"polygon": [[190, 10], [250, 10], [250, 40], [190, 40]]},
              "B2": {"polygon": [[500, 500], [510, 500], [510, 510]]}},
    "relationships": {
        "a": {"category": "intraObjectLabel", "origin": "T1", "destination": "B0"},
        "b": {"category": "intraObjectLabel", "origin": "T3", "destination": "B0"},  # noisy second label
        "c": {"category": "intraObjectLabel", "origin": "T2", "destination": "B1"},
        "d": {"category": "interObjectLinkage", "origin": "T0", "destination": "B0", "hasDirectionality": True},
        "e": {"category": "interObjectLinkage", "origin": "B0", "destination": "B1", "hasDirectionality": True},
        "f": {"category": "interObjectLinkage", "origin": "B1", "destination": "B2", "hasDirectionality": True},  # unlabelled blob
        "g": {"category": "interObjectLinkage", "origin": "T0", "destination": "T2", "hasDirectionality": False},  # undirected
    },
}


def test_build_graph_maps_blobs_and_directions():
    g, dropped = build_graph(ANN)
    assert [n.label for n in g.nodes] == ["Grass", "Rabbit", "Fox", "Terrestrial"]  # duplicate merged, stripped
    assert {(g.label_of(e.source), g.label_of(e.target)) for e in g.edges} == {("Grass", "Rabbit"), ("Rabbit", "Fox")}
    assert dropped == 1  # edge to the unlabelled blob


def test_mcnemar_and_bootstrap():
    t = evaluate_ai2d.mcnemar_exact([1, 1, 0, 0, 1], [1, 0, 1, 1, 1])
    assert t["only_first_correct"] == 1 and t["only_second_correct"] == 2 and 0 < t["p_value"] <= 1
    lo, hi = evaluate_ai2d.bootstrap_ci([1] * 50 + [0] * 50)
    assert lo < 0.5 < hi


def test_eval_api(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "RESULTS", tmp_path)
    (tmp_path / "ai2d_eval.json").write_text(json.dumps({"dataset": "ai2d", "title": "t", "summary": {"n_questions": 3}, "records": []}))
    c = TestClient(app_mod.app)
    runs = c.get("/api/eval").json()["runs"]
    assert runs[0]["run"] == "ai2d_eval" and runs[0]["dataset"] == "ai2d"
    assert c.get("/api/eval/ai2d_eval").json()["summary"]["n_questions"] == 3
    assert c.get("/api/eval/..%2Fsecret").status_code == 404
    assert c.get("/api/ai2d/image/..%2F..%2FREADME.md").status_code == 404
