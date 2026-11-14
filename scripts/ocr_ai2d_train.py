"""OCR a sample of AI2D *train* diagrams (PaddleOCR worker) for learning how OCR lines group into labels.

    python scripts/ocr_ai2d_train.py --n 1500          # ~40 min on an M4 (resumable)

Writes external/grouping/ocr_train.jsonl (git-ignored: derived from AI2D, non-commercial licence).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from studylens.datasets import ai2d  # noqa: E402
from studylens.ocr import get_ocr  # noqa: E402

OUT = ROOT / "external" / "grouping" / "ocr_train.jsonl"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=458)
    args = ap.parse_args(argv)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["image_id"] for l in OUT.read_text().splitlines() if l.strip()}
    train = sorted(ai2d.load("train"), key=lambda d: d.image_id)
    random.Random(args.seed).shuffle(train)
    todo = [d for d in train[: args.n] if d.image_id not in done]
    ocr = get_ocr()
    t0 = time.perf_counter()
    with OUT.open("a") as f:
        for i, d in enumerate(todo, 1):
            img = Image.open(d.image_path).convert("RGB")
            lines = ocr.read(img)
            f.write(json.dumps({"image_id": d.image_id, "category": d.category, "size": img.size, "lines": lines}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[{i}/{len(todo)}] {time.perf_counter() - t0:.0f}s", flush=True)
    ocr.close()
    print("done", len(todo), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
