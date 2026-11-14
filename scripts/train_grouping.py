"""Learn how OCR lines group into part labels, from AI2D *train* diagrams.

The hand-written rule in ``dsg.group_parts`` (merge stacked lines; merge lines inside one shape) was
designed on box-and-arrow diagrams. On AI2D anatomy diagrams it merged columns of separate labels
("nuclear envelope nucleolus chromatin ...") and lost 27% of labels (docs/evaluation.md, section 2).

Supervision: AI2D stores every label as ONE text entry with a rectangle, also when it spans several
lines. OCR lines (scripts/ocr_ai2d_train.py) are assigned to the AI2D text entry containing their centre;
two neighbouring lines belong to the same label iff they map to the same entry. A logistic-regression
classifier on layout features decides "same label" for candidate pairs (lines above/below each other,
or in the same shape); it is small enough to store as a JSON of weights.

    python scripts/train_grouping.py            # writes checkpoints/grouping_model.json + report

The weights are derived from AI2D annotations (non-commercial licence) and stay out of the repository.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from studylens.datasets import ai2d  # noqa: E402
from studylens.dsg import candidate_pairs, group_parts, number_texts, pair_features  # noqa: E402
from studylens.regions import binarise, enclosing_box  # noqa: E402

OCR = ROOT / "external" / "grouping" / "ocr_train.jsonl"
OUT = ROOT / "checkpoints" / "grouping_model.json"


def gold_ids(texts: list[dict], ann: dict) -> dict[str, str | None]:
    """OCR text id -> AI2D text entry id whose rectangle contains the line's centre (smallest such)."""
    rects = []
    for tid, t in ann.get("text", {}).items():
        (x1, y1), (x2, y2) = t["rectangle"]
        rects.append((tid, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    out = {}
    for t in texts:
        b = t["box"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        inside = [r for r in rects if r[1] - 3 <= cx <= r[3] + 3 and r[2] - 3 <= cy <= r[4] + 3]
        out[t["id"]] = min(inside, key=lambda r: (r[3] - r[1]) * (r[4] - r[2]))[0] if inside else None
    return out


def load():
    rows = []
    for line in OCR.read_text().splitlines():
        r = json.loads(line)
        ann = json.loads((ai2d.AI2D_DIR / "annotations" / f"{r['image_id']}.json").read_text())
        img = Image.open(ai2d.AI2D_DIR / "images" / r["image_id"]).convert("RGB")
        texts = number_texts(r["lines"])
        if len(texts) < 2:
            continue
        ink = binarise(img)
        shapes = {t["id"]: enclosing_box(ink, t["box"]) for t in texts}
        rows.append({"id": r["image_id"], "texts": texts, "shapes": shapes, "gold": gold_ids(texts, ann),
                     "size": img.size, "ink": ink})
    return rows


def pairs_xy(rows):
    X, y = [], []
    for r in rows:
        heights = sorted(t["box"][3] - t["box"][1] for t in r["texts"])
        med_h = heights[len(heights) // 2]
        for a, b in candidate_pairs(r["texts"], r["shapes"], med_h):
            ga, gb = r["gold"][a["id"]], r["gold"][b["id"]]
            if ga is None or gb is None:
                continue
            X.append(pair_features(a, b, r["shapes"], max(1.0, med_h), r["size"], r["ink"]))
            y.append(int(ga == gb))
    return np.array(X, dtype=float), np.array(y)


def label_scores(rows, model):
    """Exact label recovery: a gold label counts if some predicted part has exactly its OCR lines."""
    found = total = 0
    for r in rows:
        parts, _ = group_parts(r["texts"], r["ink"], r["size"], model=model)
        pred = {frozenset(p["text_ids"]) for p in parts}
        by_gold: dict[str, set] = {}
        for tid, g in r["gold"].items():
            if g is not None:
                by_gold.setdefault(g, set()).add(tid)
        for members in by_gold.values():
            total += 1
            found += frozenset(members) in pred
    return found, total


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = load()
    random.Random(458).shuffle(rows)
    k = int(0.8 * len(rows))
    tr, te = rows[:k], rows[k:]
    Xtr, ytr = pairs_xy(tr)
    Xte, yte = pairs_xy(te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    acc = float((clf.predict(sc.transform(Xte)) == yte).mean())
    model = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(),
             "intercept": float(clf.intercept_[0]), "threshold": 0.5}
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        gb_acc = float((gb.predict(Xte) == yte).mean())
    except Exception:  # noqa: BLE001
        gb_acc = None
    # decision threshold chosen on the held-out TRAIN diagrams (never on evaluation diagrams)
    sweep = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        model["threshold"] = t
        sweep[t] = label_scores(te, model)[0]
    model["threshold"] = max(sweep, key=lambda t: (sweep[t], -abs(t - 0.5)))
    base_f, base_t = label_scores(te, None)
    new_f, new_t = label_scores(te, model)
    report = {"diagrams_train": len(tr), "diagrams_test": len(te), "pairs_train": int(len(ytr)), "pairs_test": int(len(yte)),
              "same_label_rate_test": round(float(yte.mean()), 3), "pair_accuracy_logreg": round(acc, 3),
              "pair_accuracy_gboost": round(gb_acc, 3) if gb_acc is not None else None,
              "threshold": model["threshold"], "threshold_sweep": {str(k): v for k, v in sweep.items()},
              "labels_recovered_exactly": {"hand_rule": f"{base_f}/{base_t} ({base_f / base_t:.1%})",
                                           "learned": f"{new_f}/{new_t} ({new_f / new_t:.1%})"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**model, "report": report}, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
