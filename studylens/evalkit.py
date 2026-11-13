"""Shared evaluation helpers: resumable cache, statistics, extraction metrics and answer scoring.

Used by scripts/evaluate.py (synthetic pilot) and scripts/evaluate_ai2d.py (AI2D test split).
"""

from __future__ import annotations

import json
import random
import re
from math import comb
from pathlib import Path

from .schema import DiagramGraph


def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


# ------------------------------------------------------------------ cache
class Cache:
    """Append-only JSONL cache so long runs can be stopped and resumed."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.d: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.d[rec["key"]] = rec
        self.f = open(path, "a")

    def get(self, key):
        return self.d.get(key)

    def put(self, key, value: dict):
        rec = {"key": key, **value}
        self.d[key] = rec
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


# ------------------------------------------------------------------ statistics
def bootstrap_ci(vals: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    accs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (round(accs[int(0.025 * n_boot)], 3), round(accs[int(0.975 * n_boot) - 1], 3))


def mcnemar_exact(a: list[int], b: list[int]) -> dict:
    """Exact two-sided McNemar (binomial) test on paired 0/1 outcomes."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # only b correct
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # only a correct
    n = b01 + b10
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_first_correct": b10, "only_second_correct": b01, "p_value": round(p, 4)}


def accuracy(vals: list[int]) -> dict:
    return {"correct": sum(vals), "n": len(vals), "accuracy": round(sum(vals) / len(vals), 3) if vals else None,
            "ci95": bootstrap_ci(vals)}


# ------------------------------------------------------------------ extraction metrics
def prf(pred: set, gold: set) -> dict:
    return {"tp": len(pred & gold), "pred": len(pred), "gold": len(gold)}


def finish(c: dict) -> dict:
    p = c["tp"] / c["pred"] if c["pred"] else 0.0
    r = c["tp"] / c["gold"] if c["gold"] else 0.0
    return {**c, "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def extraction_scores(pred: DiagramGraph, gold: DiagramGraph) -> dict:
    pn = {norm(n.label) for n in pred.nodes}
    gn = {norm(n.label) for n in gold.nodes}
    pe = {(norm(pred.label_of(e.source)), norm(pred.label_of(e.target))) for e in pred.edges}
    ge = {(norm(gold.label_of(e.source)), norm(gold.label_of(e.target))) for e in gold.edges}
    return {"entities": prf(pn, gn), "edges": prf(pe, ge), "reversed": len({(s, t) for s, t in pe - ge if (t, s) in ge}),
            "missing": sorted(f"{s}->{t}" for s, t in ge - pe), "extra": sorted(f"{s}->{t}" for s, t in pe - ge)}


# ------------------------------------------------------------------ free-text scoring (synthetic pilot)
ABSTAIN_PAT = re.compile(r"not shown|cannot|can t|does not show|no component|nothing|^none|not (directly )?fed|no arrow|"
                         r"not enough information|does not (provide|contain|give|indicate)")
_NUMS = {"zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def mentioned(text: str, labels: list[str]) -> set[str]:
    t = " " + norm(text) + " "
    found = set()
    for lab in sorted(labels, key=len, reverse=True):
        nl = norm(lab)
        if nl and f" {nl} " in t:
            found.add(nl)
            t = t.replace(f" {nl} ", " ")
    return found


def first_number(text: str) -> int | None:
    for w in norm(text).split():
        if w.isdigit():
            return int(w)
        if w in _NUMS:
            return _NUMS[w]
    return None


def score_text(answer: str, gold: dict, q_entities: list[str], labels: list[str], abstained: bool = False) -> bool:
    """Score a free-text answer (E1, E2, or E3's model fallback) against a synthetic gold answer."""
    a = norm(answer or "")
    if gold["type"] == "abstain":
        return abstained or bool(ABSTAIN_PAT.search(a))
    if abstained:
        return False
    if gold["type"] == "yesno":
        if a.startswith("yes"):
            return gold["value"] is True
        if a.startswith("no"):
            return gold["value"] is False
        return False
    if gold["type"] == "count":
        return first_number(answer) == gold["value"]
    if "not shown" in a:
        return False
    want = {norm(v) for v in gold["value"]}
    got = mentioned(answer, labels) - ({norm(e) for e in q_entities} - want)
    return got == want


def score_e3(res: dict, gold: dict, q_entities: list[str], labels: list[str], graph: DiagramGraph) -> bool:
    """Score a final E3 result: graph-query answers by their answer nodes, fallbacks as free text."""
    if res["source"] == "graph_query":
        if gold["type"] == "abstain":
            return res["status"] == "abstained"
        if res["status"] != "graph_answer":
            return False
        if gold["type"] == "yesno":
            return res["answer"].lower().startswith("yes" if gold["value"] else "no")
        if gold["type"] == "count":
            return first_number(res["answer"]) == gold["value"]
        got = {norm(graph.label_of(n)) for n in res["answer_nodes"]}
        return got == {norm(v) for v in gold["value"]}
    return score_text(res["answer"], gold, q_entities, labels, abstained=res["status"] == "abstained")
