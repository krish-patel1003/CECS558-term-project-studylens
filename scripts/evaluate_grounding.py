"""Grounded-explanation evaluation on AI2D (test split): do the parts an answer points at match the
diagram, and do multiple-choice answers improve?

Systems (same base model, Qwen2.5-VL-3B, same images):
  OURS  StudyLens: Diagram Scene Graph (PaddleOCR parts + model arrows + shape/grounded regions)
        -> explanation that cites part / arrow ids -> deterministic reference check
  B2    plain VLM with native grounding: the model explains the diagram and gives a bounding box for
        every part it mentions (no scene graph, no checks): the strongest same-model baseline
For multiple-choice questions on the same diagrams:
  E2    image + question -> answer (plain VLM)
  E3    graph query over OUR scene graph, model fallback with checked citations

Metrics
  * pointing accuracy: share of referenced parts (whose name matches an annotated AI2D label) whose
    highlight lands on the gold location (leader-line tip, or the labelled object's box);
  * coverage: share of annotated labels that are referenced AND correctly highlighted;
  * flow-reference precision (OURS): cited arrows that are gold AI2D arrows;
  * invented references: ids that do not exist (OURS, removed) / named parts that match no text in
    the image (B2);
  * scene-graph region accuracy by region source (shape / grounded / label);
  * multiple-choice accuracy E2 vs E3 with McNemar.

    python scripts/evaluate_grounding.py --n-parts 60 --n-flow 40        # ~6-8 h on an M4 (resumable)
    python scripts/evaluate_grounding.py --summarize-only

Raw outputs (which quote AI2D text) stay in external/grounding_runs/; results/grounding_eval.json
holds ids and numbers only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QWEN_MAX_IMAGE_SIDE", "1024")  # the app's setting (OCR boxes need detail)

from PIL import Image  # noqa: E402

from studylens import explain as explain_mod, qa_nl  # noqa: E402
from studylens.datasets import ai2d  # noqa: E402
from studylens.evalkit import Cache, accuracy, mcnemar_exact, norm  # noqa: E402
from studylens.pipelines import collect, combine_e3, route_e3, run_explain  # noqa: E402
from studylens.query import structured_answer  # noqa: E402
from studylens.schema import DiagramGraph  # noqa: E402

CACHE = ROOT / "external" / "grounding_runs" / "grounding_cache.jsonl"
RESULTS = ROOT / "results"
WALK_Q = "Explain this diagram to a student: go through its labelled parts one by one and, if there are arrows, describe the flow they show."
B2_PROMPT = explain_mod.BOX_PROMPT  # the model explains and boxes parts itself; StudyLens reuses it for boxes
FLOW_CATS = {"foodChainsWebs", "lifeCycles", "waterCNPCycle", "rockCycle", "photosynthesisRespiration"}


def sample(n_parts: int, n_flow: int, seed: int, q_per: int, exclude: set[str] | None = None):
    rng = random.Random(seed)
    test = [d for d in ai2d.load("test") if d.image_id not in (exclude or set())]
    anns = {}
    parts_pool, flow_pool = [], []
    for d in test:
        ann = json.loads((ai2d.AI2D_DIR / "annotations" / f"{d.image_id}.json").read_text())
        gl = ai2d.gold_locations(ann)
        if d.category == "partsOfA" and len(gl) >= 3:
            parts_pool.append(d)
        elif d.category in FLOW_CATS and len(d.graph.edges) >= 2:
            flow_pool.append(d)
        anns[d.image_id] = gl
    parts_pool.sort(key=lambda d: d.image_id)
    flow_pool.sort(key=lambda d: d.image_id)
    rng.shuffle(parts_pool)
    rng.shuffle(flow_pool)
    chosen = parts_pool[:n_parts] + flow_pool[:n_flow]
    items = []
    for d in chosen:
        qs = sorted(d.questions, key=lambda q: q.qid)
        random.Random(f"{seed}|{d.image_id}").shuffle(qs)  # per diagram: same questions at any sample size
        items += [(d, q) for q in qs[:q_per]]
    return chosen, items, anns


_B2_PART = re.compile(r'"name"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)')


def b2_parts(raw: str, scale) -> list[dict]:
    sx, sy = scale
    return [{"label": m.group(1), "bbox": [float(m.group(2)) / sx, float(m.group(3)) / sy, float(m.group(4)) / sx, float(m.group(5)) / sy]}
            for m in _B2_PART.finditer(raw or "")]


def match_gold(label: str, gold: dict) -> str | None:
    import difflib

    k = norm(label)
    if k in gold:
        return k
    close = difflib.get_close_matches(k, list(gold), n=1, cutoff=0.85)
    return close[0] if close else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-parts", type=int, default=30)
    ap.add_argument("--n-flow", type=int, default=20)
    ap.add_argument("--q-per-diagram", type=int, default=3)
    ap.add_argument("--seed", type=int, default=458)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--tag", default="", help="system version: '' = Phase-2 system; 'v2' = learned grouping + "
                    "explain-and-box + routed E3; 'v3' = v2 + DSG adapter (set QWEN_ADAPTER_PATH, ADAPTER_KIND=dsg); "
                    "'v2d' = v2 + trained arrow detector")
    ap.add_argument("--explain-version", default=None, help=f"tagged runs: explanation version (default {EXPLAIN_VERSION!r}; "
                    "'' = the one-call explain-and-box answers of the first v2 run)")
    ap.add_argument("--exclude", default="", help="file of AI2D image ids to leave out of the sample (e.g. the dev set)")
    ap.add_argument("--write-ids", default="", help="write the sampled image ids to this file")
    args = ap.parse_args(argv)
    exclude = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    diagrams, items, golds = sample(args.n_parts, args.n_flow, args.seed, args.q_per_diagram, exclude)
    if args.write_ids:
        Path(args.write_ids).write_text("\n".join(d.image_id for d in diagrams) + "\n")
    K = key_fn(args.tag, args.explain_version)
    ev = EXPLAIN_VERSION if args.explain_version is None else args.explain_version
    print(f"{len(diagrams)} diagrams, {len(items)} questions", flush=True)
    cache = Cache(CACHE)
    vlm = ocr = dsg = None
    if not args.summarize_only:
        from studylens.dsg import DSGExtractor
        from studylens.extractors.qwen_vl import QwenVLExtractor
        from studylens.ocr import get_ocr

        vlm = QwenVLExtractor()
        vlm._load()
        ocr = get_ocr()
        # tags ending in "d" use the trained arrow detector (studylens/arrows.py) instead of the VLM's arrow list
        dsg = DSGExtractor(vlm, ocr, arrow_detector="auto" if args.tag.endswith("d") else None)
        if args.tag.endswith("d") and dsg.arrow_detector is None:
            sys.exit("tag ends in 'd' but no arrow detector weights (checkpoints/arrow_detector.pt or ARROW_DETECTOR_PATH)")
    t0 = time.time()
    by_img = defaultdict(list)
    for d, q in items:
        by_img[d.image_id].append(q)
    for i, d in enumerate(diagrams):
        if vlm is None:
            break
        img = Image.open(d.image_path).convert("RGB")
        if cache.get(K("dsg", d.image_id)) is None:
            g = dsg.extract(img)
            cache.put(K("dsg", d.image_id), {"graph": g.to_dict(), "latency_s": round(dsg.last_latency_s, 1)})
        g = DiagramGraph.from_dict(cache.get(K("dsg", d.image_id))["graph"])
        if cache.get(f"b2|{d.image_id}") is None:
            ts = time.perf_counter()
            raw, scale = vlm._generate(img, B2_PROMPT, 700)
            cache.put(f"b2|{d.image_id}", {"raw": raw, "parts": b2_parts(raw, scale), "latency_s": round(time.perf_counter() - ts, 1)})
        derive_x3(cache, K, d.image_id)
        if cache.get(K("ours", d.image_id)) is None:
            # tagged systems box parts with the same call as B2 (identical prompt, greedy decoding): reuse it
            boxes = ([{"name": p["label"], "bbox": p["bbox"]} for p in cache.get(f"b2|{d.image_id}")["parts"]]
                     if args.tag else None)
            r = collect(run_explain(vlm, img, g, WALK_Q, locator=dsg.ground, box_parts=boxes))
            cache.put(K("ours", d.image_id), r)
        for q in by_img[d.image_id]:
            if cache.get(f"E2|{q.qid}") is None:
                cache.put(f"E2|{q.qid}", qa_nl.direct_answer(vlm, img, q.text, q.choices))
            if cache.get(K("Q", q.qid)) is None:
                cache.put(K("Q", q.qid), structured_answer(q.text, g, q.choices, parser=vlm))
            s = cache.get(K("Q", q.qid))
            if not args.tag and s["status"] != "answered" and cache.get(f"F|{q.qid}") is None:
                cache.put(f"F|{q.qid}", qa_nl.grounded_answer(vlm, img, q.text, g, q.choices))
        print(f"[{i + 1}/{len(diagrams)}] {d.image_id} ({d.category}) {time.time() - t0:.0f}s", flush=True)
    out = summarize(diagrams, items, golds, cache, args)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"grounding_eval{'_' + args.tag if args.tag else ''}{'_' + ev if args.tag and ev else ''}{'_test' if args.exclude else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    if ocr is not None:
        ocr.close()
    return 0


EXPLAIN_VERSION = "x3"  # x3 = x2 + unreliable boxes hidden; x2 = two calls (text, then boxes); "" = one call


def derive_x3(cache: Cache, K, image_id: str) -> None:
    """x3 differs from x2 only by a deterministic filter (explain.gate_boxes), so it is derived from the cached
    x2 explanation instead of re-running the model."""
    import copy

    k3 = K("ours", image_id)
    if not k3.startswith("ours_") or "_x3|" not in k3 or cache.get(k3) is not None:
        return
    x2, dg = cache.get(k3.replace("_x3|", "_x2|")), cache.get(K("dsg", image_id))
    if not (x2 and dg):
        return
    r = copy.deepcopy(x2)
    r.setdefault("checks", {})["boxes_hidden"] = explain_mod.gate_boxes(r["references"], DiagramGraph.from_dict(dg["graph"]))
    cache.put(k3, r)


def key_fn(tag: str, explain_version: str | None = None):
    """Cache key per system version; the B2 baseline and E2 answers do not depend on it."""
    ev = EXPLAIN_VERSION if explain_version is None else explain_version

    def key(kind, ident):
        if not tag or kind not in ("dsg", "ours", "Q"):
            return f"{kind}|{ident}"
        return f"{kind}_{tag}{'_' + ev if kind == 'ours' and ev else ''}|{ident}"
    return key


def summarize(diagrams, items, golds, cache: Cache, args) -> dict:
    K = key_fn(getattr(args, "tag", ""), getattr(args, "explain_version", None))
    per_diag = []
    agg = {s: {"refs": 0, "matched": 0, "hits": 0, "gold": 0, "covered": 0, "unmatched": 0} for s in ("OURS", "B2")}
    flow = {"cited": 0, "gold": 0}
    invented = {"OURS": 0, "B2": 0}
    region = defaultdict(lambda: [0, 0])
    for d in diagrams:
        derive_x3(cache, K, d.image_id)
        dg, ours, b2 = cache.get(K("dsg", d.image_id)), cache.get(K("ours", d.image_id)), cache.get(f"b2|{d.image_id}")
        if not (dg and ours and b2):
            continue
        gold = golds.get(d.image_id, {})
        g = DiagramGraph.from_dict(dg["graph"])
        size = tuple(g.image_size)
        # scene-graph regions vs gold, by how the region was obtained
        for n in g.nodes:
            k = match_gold(n.label, gold)
            if k:
                region[n.region_source or "none"][1] += 1
                region[n.region_source or "none"][0] += ai2d.location_hit(n.bbox, gold[k], size)
        ours_parts = [r for r in ours.get("references", []) if r["kind"] == "part" and r.get("bbox")]
        systems = {"OURS": [{"label": r["label"], "bbox": r["bbox"]} for r in ours_parts], "B2": b2["parts"]}
        texts = {norm(t["text"]) for t in g.texts}
        row = {"image_id": d.image_id, "category": d.category, "n_gold_labels": len(gold), "dsg_s": dg["latency_s"],
               "n_parts": len(g.nodes), "n_arrows": len(g.edges)}
        for sname, parts in systems.items():
            seen_hit = set()
            a = agg[sname]
            a["gold"] += len(gold)
            hits = matched = 0
            for p in parts:
                a["refs"] += 1
                k = match_gold(p["label"], gold)
                if k is None:
                    a["unmatched"] += 1
                    if sname == "B2" and norm(p["label"]) not in texts and not any(norm(p["label"]) in t or t in norm(p["label"]) for t in texts if t):
                        invented["B2"] += 1
                    continue
                matched += 1
                h = ai2d.location_hit(p["bbox"], gold[k], size)
                hits += h
                if h:
                    seen_hit.add(k)
            a["matched"] += matched
            a["hits"] += hits
            a["covered"] += len(seen_hit)
            row[sname] = {"refs": len(parts), "matched": matched, "hits": hits, "covered": len(seen_hit)}
        invented["OURS"] += len(ours.get("checks", {}).get("invalid_ids", []))
        # flow references vs gold arrows
        gold_edges = {(norm(d.graph.label_of(e.source)), norm(d.graph.label_of(e.target))) for e in d.graph.edges}
        for r in ours.get("references", []):
            if r["kind"] in ("flow", "arrow"):
                for a_ in r["arrows"]:
                    flow["cited"] += 1
                    flow["gold"] += (norm(g.label_of(a_["source"])), norm(g.label_of(a_["target"]))) in gold_edges
        per_diag.append(row)
    summary = {"n_diagrams_scored": len(per_diag), "pointing": {}, "flow_reference_precision": None,
               "invented_references": invented}
    for s, a in agg.items():
        summary["pointing"][s] = {**a, "pointing_accuracy": round(a["hits"] / a["matched"], 3) if a["matched"] else None,
                                  "coverage": round(a["covered"] / a["gold"], 3) if a["gold"] else None}
    if flow["cited"]:
        summary["flow_reference_precision"] = {"cited_arrows": flow["cited"], "gold_arrows": flow["gold"],
                                               "precision": round(flow["gold"] / flow["cited"], 3)}
    summary["region_accuracy_by_source"] = {k: {"hits": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 3) if v[1] else None}
                                            for k, v in region.items()}
    # multiple choice
    recs = []
    tagged = bool(getattr(args, "tag", ""))
    for d, q in items:
        e2, s = cache.get(f"E2|{q.qid}"), cache.get(K("Q", q.qid))
        dg = cache.get(K("dsg", d.image_id))
        if not (e2 and s and dg):
            continue
        if tagged:  # routed policy: graph answer for rule-parsed questions, else the image-only answer (E2)
            res = route_e3(s, e2)
            recs.append({"qid": q.qid, "category": d.category, "E2": int(e2.get("choice_index") == q.answer),
                         "E3": int(res.get("choice_index") == q.answer), "E3_abstained": 0,
                         "E3_no_abstain": int(res.get("choice_index") == q.answer), "query_status": s["status"],
                         "graph_used": int(res.get("source") == "graph_query")})
            continue
        f = cache.get(f"F|{q.qid}")
        if s["status"] != "answered" and f is None:
            continue
        g = DiagramGraph.from_dict(dg["graph"])
        res = combine_e3(s, f, g)
        res_na = combine_e3(s, f, g, abstain_on_query_abstain=False)
        recs.append({"qid": q.qid, "category": d.category, "E2": int(e2.get("choice_index") == q.answer),
                     "E3": int(res.get("choice_index") == q.answer), "E3_abstained": int(res.get("choice_index") is None),
                     "E3_no_abstain": int(res_na.get("choice_index") == q.answer), "query_status": s["status"]})
    if recs:
        summary["multiple_choice"] = {"n": len(recs), "E2": accuracy([r["E2"] for r in recs]), "E3": accuracy([r["E3"] for r in recs]),
                                      "E2_vs_E3": mcnemar_exact([r["E2"] for r in recs], [r["E3"] for r in recs]),
                                      "answered_by_query": sum(r["query_status"] == "answered" for r in recs),
                                      "E3_abstentions": sum(r["E3_abstained"] for r in recs),
                                      # same pipeline, but when the query finds no supporting arrow, use the model fallback
                                      "E3_no_abstain": accuracy([r["E3_no_abstain"] for r in recs]),
                                      "E2_vs_E3_no_abstain": mcnemar_exact([r["E2"] for r in recs], [r["E3_no_abstain"] for r in recs])}
        for cat_name, keep in (("parts", lambda r: r["category"] == "partsOfA"), ("flow", lambda r: r["category"] != "partsOfA")):
            sub = [r for r in recs if keep(r)]
            if sub:
                summary["multiple_choice"][f"by_category_{cat_name}"] = {
                    "n": len(sub), "E2": accuracy([r["E2"] for r in sub])["accuracy"],
                    "E3": accuracy([r["E3"] for r in sub])["accuracy"],
                    "E3_no_abstain": accuracy([r["E3_no_abstain"] for r in sub])["accuracy"]}
    return {"title": "Grounded explanations on AI2D: StudyLens vs the same VLM with native grounding",
            "dataset": "ai2d-grounding", "created": str(date.today()), "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "settings": vars(args), "summary": summary, "per_diagram": per_diag, "records": recs,
            "note": "Explanations and question text are not included (AI2D licence)."}


if __name__ == "__main__":
    sys.exit(main())
