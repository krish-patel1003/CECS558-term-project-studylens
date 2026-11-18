"""Arrow detector (optional): a Mask R-CNN trained on AI2D's arrow and arrowhead annotations
(``scripts/train_arrow_detector.py``) finds the drawn arrows; a deterministic rule attaches each arrow's
tail and head to the nearest scene-graph part.

Why: the VLM's arrow list is the weakest stage (arrow F1 ~0.30 on the AI2D dev set). Attaching the TRUE
AI2D arrow outlines to our parts with the same rule gives F1 0.53 on those diagrams, so a good detector
can replace the VLM's arrow call. When no weights are present, the scene graph keeps the VLM call.

Weights are derived from AI2D (non-commercial licence) and are not committed.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "checkpoints" / "arrow_detector.pt"
CLASSES = ("background", "arrow", "arrowhead")
MAX_LINK_FRAC = 0.15  # an arrow end farther than this (x the image's longer side) from every part is not linked
HEAD_FRAC = 0.08  # an arrowhead must lie within this distance of an arrow end to give its direction


def build_model(num_classes: int = len(CLASSES), pretrained: bool = False, min_size: int = 640, max_size: int = 1024):
    """torchvision Mask R-CNN (BSD licence); ``pretrained`` starts from COCO weights for training."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return m


def arrow_ends(points: list[list[float]]) -> tuple[list[float], list[float]] | None:
    """The two points of the outline farthest apart (the arrow's ends; adequate for gently curved arrows)."""
    pts = points if len(points) <= 400 else points[:: math.ceil(len(points) / 400)]
    best, pair = -1.0, None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > best:
                best, pair = d, (list(pts[i]), list(pts[j]))
    return pair


def _dist_to_box(p, b) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def link_arrows(arrows: list[dict], heads: list[list[float]], nodes, size: tuple[int, int]) -> list[tuple[str, str]]:
    """``arrows``: [{"points": outline or mask pixels}], ``heads``: arrowhead boxes, ``nodes``: scene-graph
    nodes (bbox, label_bbox). Returns directed (source id, target id) pairs, deduplicated. An arrow whose
    direction is unknown (no arrowhead near either end) or whose ends meet the same part is skipped."""
    L = max(size)
    out: list[tuple[str, str]] = []
    if len(nodes) < 2:
        return out

    def nearest(p):
        best = min(nodes, key=lambda n: min(_dist_to_box(p, n.bbox), _dist_to_box(p, n.label_bbox or n.bbox)))
        d = min(_dist_to_box(p, best.bbox), _dist_to_box(p, best.label_bbox or best.bbox))
        return best if d <= MAX_LINK_FRAC * L else None

    centres = [((h[0] + h[2]) / 2, (h[1] + h[3]) / 2) for h in heads]
    for a in arrows:
        ends = arrow_ends(a["points"])
        if not ends or not centres:
            continue
        p, q = ends
        dp = min(math.dist(p, c) for c in centres)
        dq = min(math.dist(q, c) for c in centres)
        if min(dp, dq) > HEAD_FRAC * L:
            continue
        tail, head = (p, q) if dq < dp else (q, p)
        s, t = nearest(tail), nearest(head)
        if s is None or t is None or s.id == t.id or (s.id, t.id) in out:
            continue
        out.append((s.id, t.id))
    return out


class ArrowDetector:
    """Loads the trained detector; ``detect(image)`` -> (arrows, arrowhead boxes) in original pixels."""

    def __init__(self, path: str | os.PathLike = DEFAULT_WEIGHTS, device: str | None = None, score: float = 0.5):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score = score
        self.model = build_model()
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state.get("model", state))
        self.model.eval().to(self.device)

    def detect(self, image) -> tuple[list[dict], list[list[float]]]:
        import numpy as np
        from torchvision.transforms.functional import to_tensor

        with self.torch.no_grad():
            out = self.model([to_tensor(image.convert("RGB")).to(self.device)])[0]
        arrows, heads = [], []
        for box, label, sc, mask in zip(out["boxes"], out["labels"], out["scores"], out["masks"]):
            if float(sc) < self.score:
                continue
            if int(label) == 2:
                heads.append([float(v) for v in box])
            elif int(label) == 1:
                ys, xs = np.nonzero(mask[0].cpu().numpy() > 0.5)
                if len(xs) >= 2:
                    pts = np.stack([xs, ys], 1)[:: max(1, len(xs) // 400)].tolist()
                    arrows.append({"points": pts, "score": float(sc), "bbox": [float(v) for v in box]})
        return arrows, heads


@functools.lru_cache(maxsize=1)
def load_detector() -> ArrowDetector | None:
    """The detector if weights exist (ARROW_DETECTOR_PATH or checkpoints/arrow_detector.pt), else None."""
    path = Path(os.environ.get("ARROW_DETECTOR_PATH", DEFAULT_WEIGHTS))
    if os.environ.get("ARROW_DETECTOR", "1") == "0" or not path.exists():
        return None
    return ArrowDetector(path)
