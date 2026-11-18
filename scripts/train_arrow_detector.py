"""Train the arrow detector (studylens/arrows.py) on AI2D's arrow and arrowhead annotations.

Mask R-CNN (torchvision, COCO start), two classes: arrow (polygon mask) and arrowhead (box). Only the AI2D
TRAIN split is used; 150 train diagrams (seeded) are held out for validation. The AI2D test split is never
used here. Resumable: a checkpoint is written every epoch to checkpoints/<run>/last.pt.

    python scripts/train_arrow_detector.py --probe                 # seconds per step, nothing saved
    python scripts/train_arrow_detector.py --run arrows --epochs 12
    python scripts/train_arrow_detector.py --run arrows --eval     # validation on the held-out diagrams

Outputs checkpoints/<run>/arrow_detector.pt (model weights) and checkpoints/<run>/val_metrics.json.
Copy arrow_detector.pt to checkpoints/ on the laptop (or set ARROW_DETECTOR_PATH). The weights are
derived from AI2D (non-commercial licence): keep them private, do not commit or publish them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from studylens.arrows import build_model, link_arrows  # noqa: E402
from studylens.datasets import ai2d  # noqa: E402
from studylens.evalkit import norm  # noqa: E402
from studylens.schema import Node  # noqa: E402

CKPT = ROOT / "checkpoints"


def split(n_holdout: int, seed: int = 458):
    ids = sorted(d.image_id for d in ai2d.load("train"))
    random.Random(seed).shuffle(ids)
    return ids[n_holdout:], ids[:n_holdout]


def annotation(image_id: str) -> dict:
    return json.loads((ai2d.AI2D_DIR / "annotations" / f"{image_id}.json").read_text())


def targets(ann: dict, size: tuple[int, int]):
    """Boxes, labels (1 arrow, 2 arrowhead) and masks from the annotation."""
    W, H = size
    boxes, labels, masks = [], [], []

    def add(poly, label):
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        b = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))]
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            return
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
        boxes.append(b)
        labels.append(label)
        masks.append(m)

    for a in ann.get("arrows", {}).values():
        if len(a.get("polygon", [])) >= 3:
            add(a["polygon"], 1)
    for h in ann.get("arrowHeads", {}).values():
        (x1, y1), (x2, y2) = h["rectangle"]
        add([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], 2)
    return boxes, labels, masks


class AI2DArrows:
    def __init__(self, ids, train: bool):
        self.ids, self.train = ids, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import numpy as np
        import torch
        from torchvision.transforms.functional import to_tensor

        iid = self.ids[i]
        img = Image.open(ai2d.AI2D_DIR / "images" / iid).convert("RGB")
        boxes, labels, masks = targets(annotation(iid), img.size)
        if self.train and random.random() < 0.5:  # horizontal flip
            W = img.size[0]
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            masks = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in masks]
            boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]
        W, H = img.size
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(labels, dtype=torch.int64),
             "masks": torch.from_numpy(np.stack([np.array(m) for m in masks])) if masks else torch.zeros((0, H, W), dtype=torch.uint8)}
        return to_tensor(img), t


def collate(b):
    return tuple(zip(*b))


def train(args) -> None:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, _ = split(args.n_holdout)
    if args.limit:
        tr = tr[: args.limit]
    dl = torch.utils.data.DataLoader(AI2DArrows(tr, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True)
    model = build_model(pretrained=True).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(dl)
    warm = min(500, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    out = CKPT / args.run
    out.mkdir(parents=True, exist_ok=True)
    start = 0
    if (out / "last.pt").exists() and not args.probe:
        st = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["epoch"] + 1
        print(f"resumed after epoch {start}", flush=True)
    model.train()
    for ep in range(start, args.epochs):
        t0, run = time.time(), 0.0
        for k, (imgs, tgts) in enumerate(dl):
            imgs = [x.to(dev) for x in imgs]
            tgts = [{kk: v.to(dev) for kk, v in t.items()} for t in tgts]
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach())
            if args.probe and k == 9:
                print(f"probe: {(time.time() - t0) / 10:.2f} s/step at batch {args.batch}; "
                      f"~{(time.time() - t0) / 10 * len(dl) / 60:.0f} min/epoch; "
                      f"max memory {torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0:.1f} GB", flush=True)
                return
            if k % 100 == 0:
                print(f"epoch {ep + 1} step {k}/{len(dl)} loss {run / (k + 1):.3f} {time.time() - t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, out / "last.pt")
        torch.save({"model": model.state_dict()}, out / "arrow_detector.pt")
        print(f"epoch {ep + 1} done: loss {run / len(dl):.3f}, {time.time() - t0:.0f}s", flush=True)


def gold_nodes(ann: dict) -> tuple[list[Node], set[tuple[str, str]]]:
    """Gold parts (label text box; region = the labelled blob when there is one) and gold directed arrows."""
    g, _ = ai2d.build_graph(ann)
    blobs = ann.get("blobs", {})
    region = {}
    for r in ann.get("relationships", {}).values():
        if r["category"] in ("intraObjectLabel", "intraObjectRegionLabel"):
            t, b = (r["origin"], r["destination"]) if r["origin"].startswith("T") else (r["destination"], r["origin"])
            if t.startswith("T") and b in blobs:
                xs, ys = [p[0] for p in blobs[b]["polygon"]], [p[1] for p in blobs[b]["polygon"]]
                region[norm(ann["text"][t]["value"])] = [min(xs), min(ys), max(xs), max(ys)]
    nodes = [Node(id=n.id, label=n.label, bbox=region.get(norm(n.label), n.bbox), label_bbox=n.bbox) for n in g.nodes]
    edges = {(norm(g.label_of(e.source)), norm(g.label_of(e.target))) for e in g.edges}
    return nodes, edges


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def evaluate(args) -> dict:
    from studylens.arrows import ArrowDetector

    _, held = split(args.n_holdout)
    det = ArrowDetector(CKPT / args.run / "arrow_detector.pt", score=args.score)
    box = {c: [0, 0, 0] for c in ("arrow", "arrowhead")}  # tp, pred, gold
    link = {k: [0, 0, 0] for k in ("detector", "gold_outlines")}
    t0 = time.time()
    for k, iid in enumerate(held[: args.n_eval]):
        ann = annotation(iid)
        img = Image.open(ai2d.AI2D_DIR / "images" / iid).convert("RGB")
        arrows, heads = det.detect(img)
        gb, gl, _ = targets(ann, img.size)
        for c, lab, pred in (("arrow", 1, [a["bbox"] for a in arrows]), ("arrowhead", 2, heads)):
            gold = [b for b, l_ in zip(gb, gl) if l_ == lab]
            used = set()
            for p in pred:
                j = max(range(len(gold)), key=lambda i: _iou(p, gold[i]), default=None)
                if j is not None and j not in used and _iou(p, gold[j]) >= 0.5:
                    used.add(j)
            box[c][0] += len(used)
            box[c][1] += len(pred)
            box[c][2] += len(gold)
        nodes, gold_edges = gold_nodes(ann)
        lab = {n.id: norm(n.label) for n in nodes}
        gold_arrows = [{"points": a["polygon"]} for a in ann.get("arrows", {}).values()]
        gold_heads = [[h["rectangle"][0][0], h["rectangle"][0][1], h["rectangle"][1][0], h["rectangle"][1][1]] for h in ann.get("arrowHeads", {}).values()]
        for name, (A, Hd) in (("detector", (arrows, heads)), ("gold_outlines", (gold_arrows, gold_heads))):
            pred = {(lab[s], lab[t]) for s, t in link_arrows(A, Hd, nodes, img.size)}
            link[name][0] += len(pred & gold_edges)
            link[name][1] += len(pred)
            link[name][2] += len(gold_edges)
        if k % 20 == 0:
            print(f"[{k + 1}/{min(args.n_eval, len(held))}] {time.time() - t0:.0f}s", flush=True)

    def prf(tp, np_, ng):
        p, r = (tp / np_ if np_ else 0.0), (tp / ng if ng else 0.0)
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                "tp": tp, "pred": np_, "gold": ng}

    res = {"n_diagrams": min(args.n_eval, len(held)), "score_threshold": args.score,
           "detection_iou50": {c: prf(*v) for c, v in box.items()},
           "arrow_f1_on_gold_parts": {k: prf(*v) for k, v in link.items()},
           "note": "arrow_f1_on_gold_parts: detected (or true) arrow outlines attached to the GOLD parts with "
                   "studylens.arrows.link_arrows; gold_outlines is the ceiling of the linking rule."}
    (CKPT / args.run / "val_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="arrows")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-holdout", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="train on only this many diagrams (smoke tests)")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--score", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.eval:
        evaluate(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
