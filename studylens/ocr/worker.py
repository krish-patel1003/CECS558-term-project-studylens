"""PaddleOCR worker process (runs in its own virtual environment, see docs/code_guide.md).

PaddleOCR pins numpy/PyYAML versions that conflict with the PyTorch stack, so it runs in a
separate interpreter. Protocol: one JSON object per line on stdin
    {"image_b64": "<PNG bytes, base64>"}
and one JSON line per request on stdout
    {"lines": [{"text", "score", "box": [x1, y1, x2, y2]}], "latency_s"}  or  {"error": "..."}
Images are decoded in memory only; nothing is written to disk.
"""

import base64
import io
import json
import os
import sys
import time

# Keep the protocol channel clean: anything the libraries print goes to stderr.
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402
from PIL import Image  # noqa: E402

DET = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det")
REC = os.environ.get("OCR_REC_MODEL", "en_PP-OCRv5_mobile_rec")
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                text_detection_model_name=DET, text_recognition_model_name=REC)
_proto.write(json.dumps({"ready": True, "det": DET, "rec": REC}) + "\n")

for line in sys.stdin:
    try:
        req = json.loads(line)
        img = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
        t0 = time.perf_counter()
        r = ocr.predict(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]))[0]  # BGR, as PaddleOCR expects
        lines = [{"text": t, "score": round(float(s), 3), "box": [int(v) for v in b]}
                 for t, s, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]) if str(t).strip()]
        _proto.write(json.dumps({"lines": lines, "latency_s": round(time.perf_counter() - t0, 3)}) + "\n")
    except Exception as exc:  # noqa: BLE001 - reported to the client
        _proto.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
