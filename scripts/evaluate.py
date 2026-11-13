"""Synthetic pilot (the proposal's ~200-question pilot): E1 vs E2 vs E3 on knowledge-controlled diagrams.

The pilot set (scripts/make_synthetic.py, 27 diagrams, 304 questions) is created by this project,
so the gold answers are exact. It contains knowledge-free graphs with nonsense labels and
"counterfactual" diagrams whose arrows contradict textbook knowledge (reversed food chains, life
cycle and water cycle), which test whether an answer follows the drawn arrows.

Conditions (identical questions):
  E1      PaddleOCR text + positions -> Qwen2.5-VL-3B text-only           (no image, no graph)
  E2      image + question -> Qwen2.5-VL-3B                                (no graph)
  E3-vlm  image + fine-tuned extracted graph -> Qwen answers and cites arrows (no graph query)
  E3-zs   graph query over the ZERO-SHOT extracted graph (model fallback only when the query
          cannot apply; abstains when the query finds no supporting arrow)
  E3-ft   the same over the FINE-TUNED (LoRA) extracted graph
  E3*     the same over the GOLD graph (upper bound: perfect extraction)

    python scripts/evaluate.py --adapter checkpoints/ai2d_lora_r16     # full run (resumable)
    python scripts/evaluate.py --summarize-only                         # recompute from the cache
    python scripts/evaluate.py --no-model                               # E3* with rules only, seconds

Writes results/pilot_eval.json (synthetic data, so questions and raw outputs are included).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# same configuration as the AI2D evaluation and the fine-tuning (448 px)
os.environ.setdefault("QWEN_MAX_IMAGE_SIDE", "448")
os.environ.setdefault("QWEN_MAX_NEW_TOKENS", "1024")

from PIL import Image  # noqa: E402

from studylens import qa_nl  # noqa: E402
from studylens.evalkit import (Cache, accuracy, extraction_scores, finish, mcnemar_exact, score_e3,  # noqa: E402
                               score_text)
from studylens.pipelines import combine_e3  # noqa: E402
from studylens.query import structured_answer  # noqa: E402
from studylens.schema import DiagramGraph  # noqa: E402

DATA = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"
CACHE = ROOT / "external" / "pilot_runs" / "pilot_cache.jsonl"
GRAPHS = {"E3-zs": "zs", "E3-ft": "ft", "E3*": "gold"}
CONDS = ["E1", "E2", "E3-vlm", "E3-zs", "E3-ft", "E3*"]


def kind_group(kind: str) -> str:
    return "abstain" if kind.startswith("unsupported") else "structural"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "ai2d_lora_r16"))
    ap.add_argument("--limit", type=int, default=None, help="first N diagrams only")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="E3* with the keyword rules only (no model)")
    args = ap.parse_args(argv)

    manifest = json.loads((DATA / "manifest.json").read_text())
    diagrams = manifest["diagrams"][: args.limit]
    cache = Cache(CACHE)
    vlm = ocr = None
    if not (args.summarize_only or args.no_model):
        from studylens.extractors.qwen_vl import QwenVLExtractor
        from studylens.ocr import get_ocr

        vlm = QwenVLExtractor(adapter_path=args.adapter if Path(args.adapter, "adapter_config.json").exists() else "")
        vlm._load()
        ocr = get_ocr()
        print("adapter loaded:", vlm.has_adapter, flush=True)

    t0 = time.time()
    for di, d in enumerate(diagrams):
        raw = (DATA / d["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != d["sha256"]:
            raise SystemExit(f"{d['file']} does not match manifest.json; regenerate with scripts/make_synthetic.py")
        if vlm is None and not args.no_model:
            continue
        img = Image.open(DATA / d["file"]).convert("RGB")
        graphs = {"gold": DiagramGraph.from_dict(d["graph"])}
        if vlm is not None:
            for kind in ("zs", "ft"):
                key = f"extract|{kind}|{d['file']}"
                rec = cache.get(key)
                if rec is None and (kind == "zs" or vlm.has_adapter):
                    g = vlm.extract(img, use_adapter=(kind == "ft"))
                    cache.put(key, {"graph": g.to_dict(), "raw": vlm.last_raw_output, "latency_s": round(vlm.last_latency_s, 2)})
                    rec = cache.get(key)
                if rec is not None:
                    graphs[kind] = DiagramGraph.from_dict(rec["graph"])
            if cache.get(f"ocr|{d['file']}") is None:
                cache.put(f"ocr|{d['file']}", {"lines": ocr.read(img), "latency_s": ocr.last_latency_s})
        ocr_lines = (cache.get(f"ocr|{d['file']}") or {}).get("lines")
        for qi, q in enumerate(d["questions"]):
            qid = f"{d['file']}#{qi}"
            if vlm is not None:
                if cache.get(f"E1|{qid}") is None:
                    cache.put(f"E1|{qid}", qa_nl.ocr_answer(vlm, ocr_lines, img.size, q["question"]))
                if cache.get(f"E2|{qid}") is None:
                    cache.put(f"E2|{qid}", qa_nl.direct_answer(vlm, img, q["question"]))
            for gk, g in graphs.items():
                if cache.get(f"Q|{gk}|{qid}") is None:
                    cache.put(f"Q|{gk}|{qid}", structured_answer(q["question"], g, None, parser=vlm))
                s = cache.get(f"Q|{gk}|{qid}")
                # the model fallback is computed whenever the query did not answer, so both E3 policies
                # (abstain vs fall back when the query finds no arrow) can be scored
                need_f = s["status"] != "answered" or gk == "ft"  # ft: also E3-vlm for every question
                if vlm is not None and need_f and cache.get(f"F|{gk}|{qid}") is None:
                    cache.put(f"F|{gk}|{qid}", qa_nl.grounded_answer(vlm, img, q["question"], g))
        print(f"[{di + 1}/{len(diagrams)}] {d['file']} ({time.time() - t0:.0f}s)", flush=True)

    out = summarize(diagrams, cache, args)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pilot_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    if ocr is not None:
        ocr.close()
    return 0


def summarize(diagrams, cache: Cache, args) -> dict:
    records = []
    ext = {k: {"entities": {"tp": 0, "pred": 0, "gold": 0}, "edges": {"tp": 0, "pred": 0, "gold": 0}, "reversed": 0, "lat": [], "n": 0}
           for k in ("zs", "ft")}
    per_diag = []
    for d in diagrams:
        gold_g = DiagramGraph.from_dict(d["graph"])
        graphs = {"gold": gold_g}
        for kind in ("zs", "ft"):
            rec = cache.get(f"extract|{kind}|{d['file']}")
            if rec is None:
                continue
            g = DiagramGraph.from_dict(rec["graph"])
            graphs[kind] = g
            s = extraction_scores(g, gold_g)
            for k in ("entities", "edges"):
                for f in ("tp", "pred", "gold"):
                    ext[kind][k][f] += s[k][f]
            ext[kind]["reversed"] += s["reversed"]
            ext[kind]["lat"].append(rec["latency_s"])
            ext[kind]["n"] += 1
            per_diag.append({"file": d["file"], "title": d["title"], "kind": kind, "entity_f1": finish(s["entities"])["f1"],
                             "edge_f1": finish(s["edges"])["f1"], "reversed": s["reversed"], "missing": s["missing"],
                             "extra": s["extra"], "latency_s": rec["latency_s"]})
        labels = [n.label for n in gold_g.nodes]
        for qi, q in enumerate(d["questions"]):
            qid = f"{d['file']}#{qi}"
            r = {"qid": qid, "file": d["file"], "title": d["title"], "question": q["question"], "kind": q["kind"],
                 "group": kind_group(q["kind"]), "counterfactual": d["title"].startswith("Counterfactual"),
                 "knowledge_free": d["title"].startswith("Nonsense"), "gold": q["gold"]}
            for c in ("E1", "E2"):
                a = cache.get(f"{c}|{qid}")
                if a:
                    r[c] = {"answer": a["answer"], "correct": int(score_text(a["answer"], q["gold"], q["entities"], labels)),
                            "latency_s": round(a.get("latency_s") or 0, 2)}
            f_ft = cache.get(f"F|ft|{qid}")
            if f_ft:
                r["E3-vlm"] = {"answer": f_ft["answer"], "status": f_ft["status"], "evidence": [e["text"] for e in f_ft["evidence"]],
                               "correct": int(score_text(f_ft["answer"], q["gold"], q["entities"], labels, f_ft["status"] == "abstained"))}
            for c, gk in GRAPHS.items():
                s = cache.get(f"Q|{gk}|{qid}")
                if s is None or gk not in graphs:
                    continue
                f = cache.get(f"F|{gk}|{qid}")
                if s["status"] != "answered" and f is None and not args.no_model:
                    continue  # run not finished for this question
                for policy, tag in ((True, ""), (False, "+fb")):
                    res = combine_e3(s, f, graphs[gk], abstain_on_query_abstain=policy)
                    r[c + tag] = {"answer": res["answer"], "status": res["status"], "source": res["source"],
                                  "parser": s["parser"], "query": s["query"], "query_status": s["status"],
                                  "evidence": [e["text"] for e in res["evidence"]],
                                  "correct": int(score_e3(res, q["gold"], q["entities"], labels, graphs[gk]))}
            records.append(r)

    conds = [c for c in CONDS if records and all(c in r for r in records)]
    variants = [c + "+fb" for c in ("E3-zs", "E3-ft", "E3*") if records and all(c + "+fb" in r for r in records)]
    summary = {"n_diagrams": len(diagrams), "n_questions": len(records), "conditions": conds,
               "accuracy": {c: accuracy([r[c]["correct"] for r in records]) for c in conds + variants}}
    summary["subsets"] = {}
    for name, pred in [("structural", lambda r: r["group"] == "structural"), ("should_abstain", lambda r: r["group"] == "abstain"),
                       ("counterfactual_diagrams", lambda r: r["counterfactual"]), ("knowledge_free_diagrams", lambda r: r["knowledge_free"])]:
        sub = [r for r in records if pred(r)]
        summary["subsets"][name] = {c: accuracy([r[c]["correct"] for r in sub]) for c in conds}
    bykind = defaultdict(list)
    for r in records:
        bykind[r["kind"]].append(r)
    summary["by_kind"] = {k: {"n": len(v), **{c: round(sum(r[c]["correct"] for r in v) / len(v), 3) for c in conds}}
                          for k, v in sorted(bykind.items())}
    summary["paired_tests"] = {}
    for a, b in [("E2", "E1"), ("E2", "E3-vlm"), ("E2", "E3-zs"), ("E2", "E3-ft"), ("E3-zs", "E3-ft"), ("E3-ft", "E3*")]:
        if a in conds and b in conds:
            summary["paired_tests"][f"{a} vs {b}"] = mcnemar_exact([r[a]["correct"] for r in records], [r[b]["correct"] for r in records])
    summary["graph_query"] = {}
    for c in ("E3-zs", "E3-ft", "E3*"):
        if c not in conds:
            continue
        st, parsers = defaultdict(int), defaultdict(int)
        for r in records:
            st[r[c]["query_status"]] += 1
            parsers[r[c]["parser"] or "none"] += 1
        ans = [r for r in records if r[c]["query_status"] == "answered"]
        summary["graph_query"][c] = {"query_status_counts": dict(st), "parser_counts": dict(parsers),
                                     "answered_by_query": len(ans),
                                     "accuracy_when_query_answered": round(sum(r[c]["correct"] for r in ans) / len(ans), 3) if ans else None,
                                     "E2_accuracy_on_same_questions": round(sum(r["E2"]["correct"] for r in ans) / len(ans), 3) if ans and "E2" in conds else None}
    summary["extraction"] = {k: {"n_diagrams": v["n"], "entities": finish(v["entities"]), "edges": finish(v["edges"]),
                                 "reversed_edges": v["reversed"], "mean_latency_s": round(sum(v["lat"]) / len(v["lat"]), 1)}
                             for k, v in ext.items() if v["n"]}
    summary["mean_answer_latency_s"] = {c: round(sum(r[c]["latency_s"] for r in records) / len(records), 2)
                                        for c in ("E1", "E2") if c in conds}
    return {"title": f"Synthetic pilot ({len(records)} questions): E1 vs E2 vs E3", "dataset": "synthetic",
            "created": str(date.today()), "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "adapter": Path(args.adapter).name, "ocr": "PaddleOCR 3.7 (PP-OCRv5 mobile det + en rec)",
            "summary": summary, "records": records, "extraction_per_diagram": per_diag}


if __name__ == "__main__":
    sys.exit(main())
