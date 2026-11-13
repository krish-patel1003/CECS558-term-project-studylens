"""Evaluation on REAL data: AI2D official test split, human-written multiple-choice questions.

Conditions (identical questions for all):
  E1      PaddleOCR text + positions + question + options -> Qwen (text only)   [no image, no graph]
  E2      image + question + options -> Qwen2.5-VL-3B (base)                    [no graph]
  E3-vlm  image + FINE-TUNED graph -> Qwen answers and cites arrow ids           [no graph query]
  E3-zs   question -> graph query (rules, else Qwen as parser only) -> NetworkX over the ZERO-SHOT
          extracted graph -> option; Qwen's cited answer only when the query cannot apply;
          abstains when the query finds no supporting arrow
  E3-ft   the same over the FINE-TUNED (LoRA) extracted graph
  E3*     the same over the GOLD graph (AI2D annotations)                       [upper bound]
  "+fb" variants fall back to the model instead of abstaining; "-vlm" variants skip the query.
Extraction quality of the zero-shot and fine-tuned extractors is scored against the gold graph.

    python scripts/evaluate_ai2d.py --adapter checkpoints/ai2d_lora_r16 --n-diagrams 100
    python scripts/evaluate_ai2d.py --summarize-only      # recompute statistics from the cache

Licence handling: AI2D text must not be redistributed, so raw model outputs (which transcribe
diagram text) are cached in external/ai2d_runs/ (git-ignored). The committed file
results/ai2d_eval.json holds only ids, choice indices, statuses and metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Pin the evaluated configuration to the fine-tuning conditions (448 px) and to the settings the
# cached runs used, independent of the app's defaults. Set the variables explicitly to override.
os.environ.setdefault("QWEN_MAX_IMAGE_SIDE", "448")
os.environ.setdefault("QWEN_MAX_NEW_TOKENS", "1024")

from PIL import Image  # noqa: E402

from studylens import qa_nl  # noqa: E402
from studylens.datasets import ai2d  # noqa: E402
from studylens.datasets.ai2d import norm  # noqa: E402
from studylens.evalkit import Cache, bootstrap_ci, extraction_scores, finish, mcnemar_exact, prf  # noqa: E402,F401
from studylens.pipelines import combine_e3  # noqa: E402
from studylens.query import structured_answer  # noqa: E402
from studylens.schema import DiagramGraph  # noqa: E402

CACHE_DIR = ROOT / "external" / "ai2d_runs"
RESULTS = ROOT / "results"
VLM_CONDS = ["E2", "E3-zs", "E3-ft", "E3*"]  # cache keys "qa|<cond>|qid": direct and graph+citation answers
GRAPH_OF = {"E3-zs": "zs", "E3-ft": "ft", "E3*": "gold"}
CONDS = ["E1", "E2", "E3-vlm", "E3-zs", "E3-ft", "E3*"]


def is_relation_question(q, d) -> bool:
    """Relationship question (the proposal's research question): the diagram has gold arrows and the
    question or its correct answer names an entity at the end of a gold arrow. A deterministic proxy."""
    if not d.graph.edges:
        return False
    ends = {norm(d.graph.label_of(n)) for e in d.graph.edges for n in (e.source, e.target)}
    text = " " + norm(q.text + " " + q.choices[q.answer]) + " "
    return any(e and f" {e} " in text for e in ends)


# ------------------------------------------------------------------------------ sampling
def sample(n_diagrams: int, frac_with_edges: float, q_per_diagram: int, seed: int):
    test = [d for d in ai2d.load("test") if d.questions]
    rng = random.Random(seed)
    with_e = sorted([d for d in test if d.graph.edges], key=lambda d: d.image_id)
    no_e = sorted([d for d in test if not d.graph.edges], key=lambda d: d.image_id)
    rng.shuffle(with_e)
    rng.shuffle(no_e)
    k = min(len(with_e), round(n_diagrams * frac_with_edges))
    chosen = with_e[:k] + no_e[: n_diagrams - k]
    items = []
    for d in chosen:
        qs = list(d.questions)
        rng.shuffle(qs)
        for q in qs[:q_per_diagram]:
            items.append((d, q))
    return chosen, items


# ------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "ai2d_lora_r16"))
    ap.add_argument("--n-diagrams", type=int, default=100)
    ap.add_argument("--frac-with-edges", type=float, default=0.6)
    ap.add_argument("--q-per-diagram", type=int, default=8)
    ap.add_argument("--seed", type=int, default=458)
    ap.add_argument("--run", default="ai2d_eval")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    diagrams, items = sample(args.n_diagrams, args.frac_with_edges, args.q_per_diagram, args.seed)
    print(f"sample: {len(diagrams)} test diagrams ({sum(bool(d.graph.edges) for d in diagrams)} with gold arrows), "
          f"{len(items)} questions", flush=True)
    cache = Cache(CACHE_DIR / f"{args.run}_cache.jsonl")
    vlm = None
    if not args.summarize_only:
        from studylens.extractors.qwen_vl import QwenVLExtractor

        from studylens.ocr import get_ocr

        vlm = QwenVLExtractor(adapter_path=args.adapter if Path(args.adapter, "adapter_config.json").exists() else "")
        vlm._load()
        ocr = get_ocr()
        print("adapter loaded:", vlm.has_adapter, flush=True)

    t0 = time.time()
    graphs: dict[tuple[str, str], DiagramGraph] = {}
    # ---- 1. extraction (zero-shot and fine-tuned) per diagram
    for i, d in enumerate(diagrams):
        img = Image.open(d.image_path).convert("RGB")
        for kind in ("zs", "ft"):
            key = f"extract|{kind}|{d.image_id}"
            rec = cache.get(key)
            if rec is None and vlm is not None and (kind == "zs" or vlm.has_adapter):
                g = vlm.extract(img, use_adapter=(kind == "ft"))
                rec = {"graph": g.to_dict(), "raw": vlm.last_raw_output, "latency_s": round(vlm.last_latency_s, 2)}
                cache.put(key, rec)
            if rec is not None:
                graphs[(kind, d.image_id)] = DiagramGraph.from_dict(rec["graph"])
        graphs[("gold", d.image_id)] = d.graph
        if vlm is not None and cache.get(f"ocr|{d.image_id}") is None:
            cache.put(f"ocr|{d.image_id}", {"lines": ocr.read(img), "latency_s": ocr.last_latency_s})
        if vlm is not None:
            print(f"[extract {i + 1}/{len(diagrams)}] {d.image_id} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. QA
    by_img = {d.image_id: d for d in diagrams}
    for j, (d, q) in enumerate(items):
        img = None
        if vlm is not None and cache.get(f"E1|{q.qid}") is None:
            img = img or Image.open(d.image_path).convert("RGB")
            lines = cache.get(f"ocr|{d.image_id}")["lines"]
            cache.put(f"E1|{q.qid}", qa_nl.ocr_answer(vlm, lines, img.size, q.text, q.choices))
        for gk in ("zs", "ft", "gold"):
            g = graphs.get((gk, d.image_id))
            if vlm is not None and g is not None and cache.get(f"Q|{gk}|{q.qid}") is None:
                cache.put(f"Q|{gk}|{q.qid}", structured_answer(q.text, g, q.choices, parser=vlm))
        for cond in VLM_CONDS:
            key = f"qa|{cond}|{q.qid}"
            if cache.get(key) is not None or vlm is None:
                continue
            g = {"E3-zs": graphs.get(("zs", d.image_id)), "E3-ft": graphs.get(("ft", d.image_id)), "E3*": d.graph}.get(cond)
            if cond != "E2" and g is None:
                continue
            img = img or Image.open(d.image_path).convert("RGB")
            if cond == "E2":
                r = qa_nl.direct_answer(vlm, img, q.text, q.choices)
            else:
                r = qa_nl.grounded_answer(vlm, img, q.text, g, q.choices)
            cache.put(key, r)
        if vlm is not None and (j + 1) % 10 == 0:
            print(f"[qa {j + 1}/{len(items)}] ({time.time() - t0:.0f}s)", flush=True)

    # ---- 3. summarise (ids + numbers only)
    records = []
    for d, q in items:
        rec = {"qid": q.qid, "image_id": d.image_id, "category": d.category, "gold_edges": len(d.graph.edges),
               "answer_index": q.answer, "relation_question": is_relation_question(q, d)}

        def put(name, r, extra=None):
            rec[name] = {"choice": r.get("choice_index"), "correct": int(r.get("choice_index") == q.answer),
                         "status": r.get("status"), "n_evidence": len(r.get("evidence", [])),
                         "invalid_citations": len(r.get("invalid_citations", [])),
                         "latency_s": round(r.get("latency_s") or 0, 2), **(extra or {})}

        for cond, name in (("E1", "E1"), ("qa|E2", "E2"), ("qa|E3-ft", "E3-vlm"), ("qa|E3-zs", "E3-zs-vlm"), ("qa|E3*", "E3*-vlm")):
            r = cache.get(f"{cond}|{q.qid}")
            if r is not None:
                put(name, r)
        for cond, gk in GRAPH_OF.items():
            s_ = cache.get(f"Q|{gk}|{q.qid}")
            f_ = cache.get(f"qa|{cond}|{q.qid}")
            g = d.graph if gk == "gold" else (DiagramGraph.from_dict(cache.get(f"extract|{gk}|{d.image_id}")["graph"])
                                              if cache.get(f"extract|{gk}|{d.image_id}") else None)
            if s_ is None or f_ is None or g is None:
                continue
            for policy, tag in ((True, ""), (False, "+fb")):
                res = combine_e3(s_, f_, g, abstain_on_query_abstain=policy)
                put(cond + tag, {**res, "latency_s": None}, {"source": res["source"], "query_status": s_["status"],
                                                             "parser": s_["parser"], "op": (s_["query"] or {}).get("op")})
        records.append(rec)
    conds = [c for c in CONDS if all(c in r for r in records)] if records else []
    variants = [c for c in ("E3-zs+fb", "E3-ft+fb", "E3*+fb", "E3-zs-vlm", "E3*-vlm") if records and all(c in r for r in records)]
    summary = {"n_diagrams": len(diagrams), "n_questions": len(records),
               "n_with_gold_edges": sum(r["gold_edges"] > 0 for r in records), "conditions": conds,
               "chance_accuracy": round(sum(1 / len(q.choices) for _, q in items) / max(1, len(items)), 3)}
    acc = {}
    for c in conds + variants:
        v = [r[c]["correct"] for r in records]
        acc[c] = {"correct": sum(v), "n": len(v), "accuracy": round(sum(v) / len(v), 3), "ci95": bootstrap_ci(v)}
    summary["accuracy"] = acc
    summary["subsets"] = {}
    for name, pred in [("relation_questions", lambda r: r["relation_question"]), ("other_questions", lambda r: not r["relation_question"]),
                       ("with_gold_arrows", lambda r: r["gold_edges"] > 0), ("without_gold_arrows", lambda r: r["gold_edges"] == 0)]:
        sub = [r for r in records if pred(r)]
        summary["subsets"][name] = {c: {"correct": sum(r[c]["correct"] for r in sub), "n": len(sub),
                                        "accuracy": round(sum(r[c]["correct"] for r in sub) / len(sub), 3) if sub else None,
                                        "ci95": bootstrap_ci([r[c]["correct"] for r in sub])} for c in conds}
    bycat = defaultdict(list)
    for r in records:
        bycat[r["category"]].append(r)
    summary["by_category"] = {k: {"n": len(v), **{c: round(sum(r[c]["correct"] for r in v) / len(v), 3) for c in conds}}
                              for k, v in sorted(bycat.items(), key=lambda kv: -len(kv[1]))}
    summary["paired_tests_vs_E2"] = {c: mcnemar_exact([r["E2"]["correct"] for r in records], [r[c]["correct"] for r in records])
                                     for c in conds + variants if c != "E2" and "E2" in conds}
    rel = [r for r in records if r["relation_question"]]
    summary["paired_tests_vs_E2_relation_questions"] = {
        c: mcnemar_exact([r["E2"]["correct"] for r in rel], [r[c]["correct"] for r in rel]) for c in conds if c != "E2" and "E2" in conds}
    summary["graph_query"] = {}
    for c in ("E3-zs", "E3-ft", "E3*"):
        if c not in conds:
            continue
        st, parsers = defaultdict(int), defaultdict(int)
        for r in records:
            st[r[c]["query_status"]] += 1
            parsers[r[c]["parser"] or "none"] += 1
        ans = [r for r in records if r[c]["query_status"] == "answered"]
        summary["graph_query"][c] = {
            "query_status_counts": dict(st), "parser_counts": dict(parsers), "answered_by_query": len(ans),
            "accuracy_when_query_answered": round(sum(r[c]["correct"] for r in ans) / len(ans), 3) if ans else None,
            "E2_accuracy_on_same_questions": round(sum(r["E2"]["correct"] for r in ans) / len(ans), 3) if ans and "E2" in conds else None,
            "paired_test_vs_E2_on_same_questions": mcnemar_exact([r["E2"]["correct"] for r in ans], [r[c]["correct"] for r in ans]) if ans else None}
    summary["grounding"] = {}
    for c in [c for c in conds if c not in ("E1", "E2")]:
        st = defaultdict(int)
        for r in records:
            st[r[c]["status"]] += 1
        grounded = [r for r in records if r[c]["status"] in ("grounded", "graph_answer")]
        summary["grounding"][c] = {
            "status_counts": dict(st),
            "accuracy_when_grounded": round(sum(r[c]["correct"] for r in grounded) / len(grounded), 3) if grounded else None,
            "accuracy_when_not_grounded": round(sum(r[c]["correct"] for r in records if r[c]["status"] not in ("grounded", "graph_answer")) / max(1, len(records) - len(grounded)), 3),
            "answers_with_invalid_citations": sum(r[c]["invalid_citations"] > 0 for r in records),
        }
    # extraction
    ex = {}
    per_diag = []
    for kind in ("zs", "ft"):
        tot = {"entities": {"tp": 0, "pred": 0, "gold": 0}, "edges": {"tp": 0, "pred": 0, "gold": 0}, "reversed": 0}
        lat = []
        n = 0
        for d in diagrams:
            rec = cache.get(f"extract|{kind}|{d.image_id}")
            if rec is None:
                continue
            n += 1
            s = extraction_scores(DiagramGraph.from_dict(rec["graph"]), d.graph)
            for k in ("entities", "edges"):
                for f in ("tp", "pred", "gold"):
                    tot[k][f] += s[k][f]
            tot["reversed"] += s["reversed"]
            lat.append(rec["latency_s"])
            per_diag.append({"image_id": d.image_id, "kind": kind, "category": d.category,
                             "entity_f1": finish(s["entities"])["f1"], "edge_f1": finish(s["edges"])["f1"] if s["edges"]["gold"] or s["edges"]["pred"] else None,
                             "gold_edges": s["edges"]["gold"], "pred_edges": s["edges"]["pred"], "reversed": s["reversed"],
                             "latency_s": rec["latency_s"]})
        if n:
            ex[kind] = {"n_diagrams": n, "entities": finish(tot["entities"]), "edges": finish(tot["edges"]),
                        "reversed_edges": tot["reversed"], "mean_latency_s": round(sum(lat) / len(lat), 1)}
    summary["extraction"] = ex
    lat = {c: round(sum(r[c]["latency_s"] or 0 for r in records) / len(records), 1) for c in conds if c in ("E1", "E2", "E3-vlm")} if records else {}
    summary["mean_answer_latency_s"] = lat

    train_log = []
    log_path = Path(args.adapter) / "train_log.jsonl"
    if log_path.exists():
        train_log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    cfg_path = Path(args.adapter) / "config.json"
    out = {
        "title": "AI2D test split: E1 (OCR) vs E2 (image) vs E3 (graph query; zero-shot, fine-tuned and gold graphs)",
        "dataset": "ai2d", "created": str(date.today()), "seed": args.seed,
        "model": "Qwen/Qwen2.5-VL-3B-Instruct", "adapter": Path(args.adapter).name if vlm is None or vlm.has_adapter else None,
        "training_config": json.loads(cfg_path.read_text()) if cfg_path.exists() else None,
        "training_log": train_log,
        "sampling": vars(args), "summary": summary, "records": records, "extraction_per_diagram": per_diag,
        "note": "Question text, options and model outputs are not included (AI2D licence); the app joins them from a local AI2D copy.",
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.run}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
