"""Deterministic region finding (no AI): the shape that encloses a label.

In flowcharts and architecture diagrams a label usually sits inside a box, circle or cylinder. From
the label's OCR box we cast rays outward (left, right, up, down; several rays per side) on a
binarised image and stop at the first dark pixel run, i.e. the shape's border. If borders are
found on all four sides, within a size limit, and the rays on each side agree, the enclosing box
is returned. Otherwise ``None``: the label is not inside a closed shape (e.g. an anatomy label
connected to a part by a leader line), and the caller must locate the part another way.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def binarise(image: Image.Image, threshold: int = 140) -> np.ndarray:
    """True where the pixel is dark (ink)."""
    return np.asarray(image.convert("L")) < threshold


def _first_ink(line: np.ndarray, skip: int) -> int | None:
    """Index of the first ink pixel after ``skip`` pixels; None if there is none."""
    idx = np.flatnonzero(line[skip:])
    return int(idx[0]) + skip if idx.size else None


def enclosing_box(ink: np.ndarray, box: list[float], max_grow: float = 4.0, pad: int = 3,
                  agree_px: int = 6) -> list[float] | None:
    """Box of the closed shape around the text ``box`` [x1, y1, x2, y2], or None.

    ``max_grow``: the shape may be at most this many times the text size (+ 60 px) on each axis.
    ``agree_px``: the rays on one side must hit the border within this many pixels of each other.
    """
    h, w = ink.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    tw, th = x2 - x1, y2 - y1
    reach_x, reach_y = int(tw * max_grow + 60), int(th * max_grow + 60)
    ys = [y1 + th // 4, (y1 + y2) // 2, y2 - th // 4]
    xs = [x1 + tw // 4, (x1 + x2) // 2, x2 - tw // 4]
    sides = {}
    # left / right: scan rows outward from the text box
    left = [_first_ink(ink[y, max(0, x1 - pad - reach_x): max(0, x1 - pad)][::-1], 0) for y in ys]
    right = [_first_ink(ink[y, min(w, x2 + pad): min(w, x2 + pad + reach_x)], 0) for y in ys]
    up = [_first_ink(ink[max(0, y1 - pad - reach_y): max(0, y1 - pad), x][::-1], 0) for x in xs]
    down = [_first_ink(ink[min(h, y2 + pad): min(h, y2 + pad + reach_y), x], 0) for x in xs]
    for name, hits in (("l", left), ("r", right), ("u", up), ("d", down)):
        if any(v is None for v in hits) or max(hits) - min(hits) > agree_px:
            return None
        sides[name] = float(np.median(hits))
    bx1 = x1 - pad - sides["l"]
    bx2 = x2 + pad + sides["r"]
    by1 = y1 - pad - sides["u"]
    by2 = y2 + pad + sides["d"]
    # the found border must look like a line along the side, not a stray mark: check ink coverage
    def coverage(a: np.ndarray) -> float:
        return float(a.mean()) if a.size else 0.0
    xl, xr, yt, yb = int(bx1), int(min(w - 1, bx2)), int(by1), int(min(h - 1, by2))
    span_y = slice(int(by1 + (by2 - by1) * 0.2), int(by2 - (by2 - by1) * 0.2) + 1)
    span_x = slice(int(bx1 + (bx2 - bx1) * 0.2), int(bx2 - (bx2 - bx1) * 0.2) + 1)
    cov = min(coverage(ink[span_y, max(0, xl - 1): xl + 2].any(axis=1)), coverage(ink[span_y, max(0, xr - 1): xr + 2].any(axis=1)),
              coverage(ink[max(0, yt - 1): yt + 2, span_x].any(axis=0)), coverage(ink[max(0, yb - 1): yb + 2, span_x].any(axis=0)))
    if cov < 0.6:
        return None
    return [bx1, by1, bx2, by2]


def leader_tip(ink: np.ndarray, box: list[float], ring: tuple[int, int] = (2, 14), min_len: int = 15,
               max_len: int | None = None) -> tuple[float, float] | None:
    """Follow a leader line that starts next to a label and return the point where it ends.

    Anatomy-style diagrams connect a label to its part with a thin line. Ink pixels in a thin ring
    just outside the label box are line candidates; from each, we walk away from the label along the
    line's direction, re-centring on the ink at every step (lines may be slightly slanted), until the
    ink stops. The longest walk wins; its end point is the tip (the part being labelled).
    Returns None if no line of at least ``min_len`` pixels leaves the label.
    """
    h, w = ink.shape
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    max_len = max_len or int(0.6 * (h ** 2 + w ** 2) ** 0.5)
    r0, r1 = ring
    X1, Y1, X2, Y2 = int(max(0, x1 - r1)), int(max(0, y1 - r1)), int(min(w - 1, x2 + r1)), int(min(h - 1, y2 + r1))
    sub = ink[Y1:Y2 + 1, X1:X2 + 1]
    ys, xs = np.nonzero(sub)
    xs, ys = xs + X1, ys + Y1
    # ring: outside the box by at least r0 pixels on some axis
    dx_out = np.maximum(x1 - xs, xs - x2)
    dy_out = np.maximum(y1 - ys, ys - y2)
    out = np.maximum(dx_out, dy_out)
    sel = (out >= r0) & (out <= r1)
    if not sel.any():
        return None
    seeds = list(zip(xs[sel].tolist(), ys[sel].tolist()))
    # thin the seeds: one per 4x4 cell
    seen, uniq = set(), []
    for sx, sy in seeds:
        k = (sx // 4, sy // 4)
        if k not in seen:
            seen.add(k)
            uniq.append((sx, sy))
    best, best_score = None, 0.0
    for sx, sy in uniq[:60]:
        vx, vy = sx - cx, sy - cy
        n = (vx * vx + vy * vy) ** 0.5 or 1.0
        vx, vy = vx / n, vy / n
        px, py, length, misses = float(sx), float(sy), 0, 0
        widths: list[int] = []
        while length < max_len and misses < 3:
            nx_, ny_ = px + vx * 2, py + vy * 2
            found = []
            for o in range(-4, 5):
                qx, qy = int(round(nx_ - vy * o)), int(round(ny_ + vx * o))
                if 0 <= qx < w and 0 <= qy < h and ink[qy, qx]:
                    found.append((qx, qy))
            if found:
                mx = sum(f[0] for f in found) / len(found)
                my = sum(f[1] for f in found) / len(found)
                # the line meets the drawing where the ink suddenly gets wider: stop there
                if len(widths) >= 4 and len(found) > 2 * (sum(widths[-4:]) / 4) + 1:
                    px, py = mx, my
                    break
                widths.append(len(found))
                dvx, dvy = mx - px, my - py
                dn = (dvx * dvx + dvy * dvy) ** 0.5 or 1.0
                vx, vy = 0.85 * vx + 0.15 * dvx / dn, 0.85 * vy + 0.15 * dvy / dn
                vn = (vx * vx + vy * vy) ** 0.5 or 1.0
                vx, vy = vx / vn, vy / vn
                px, py, misses = mx, my, 0
            else:
                px, py, misses = nx_, ny_, misses + 1
            length += 2
        if length < min_len:
            continue
        straight = (((px - sx) ** 2 + (py - sy) ** 2) ** 0.5) / max(length, 1)
        score = length * straight ** 4  # leader lines are long AND straight; outlines curve
        if score > best_score:
            best, best_score = (px, py), score
    return best
