"""PaddleOCR client for the E1 baseline (OCR text only, no image).

The OCR engine runs in a separate interpreter (``OCR_PYTHON``, default
``external/venv-ocr/bin/python``) because its dependency pins conflict with the PyTorch stack.
The worker starts on first use and stays alive; images are sent over a pipe and never written to
disk.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import threading
from pathlib import Path

from PIL import Image

from ..config import ROOT, settings

WORKER = Path(__file__).with_name("worker.py")


class OCRUnavailable(RuntimeError):
    pass


class OCRClient:
    def __init__(self, python: str | None = None):
        self.python = python or settings.ocr_python
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.last_latency_s: float | None = None

    def available(self) -> bool:
        return Path(self.python).exists()

    def _start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.available():
            raise OCRUnavailable(f"OCR interpreter not found at {self.python}; see docs/code_guide.md (E1 setup).")
        env = dict(os.environ, PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True")
        self._proc = subprocess.Popen([self.python, "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, cwd=str(ROOT), env=env)
        hello = self._proc.stdout.readline()
        if not hello or not json.loads(hello).get("ready"):
            self.close()
            raise OCRUnavailable("OCR worker failed to start.")

    def read(self, image: Image.Image) -> list[dict]:
        """Return OCR lines [{"text", "score", "box"}] for an image (pixel coordinates)."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        payload = json.dumps({"image_b64": base64.b64encode(buf.getvalue()).decode()}) + "\n"
        with self._lock:
            self._start()
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
            reply = self._proc.stdout.readline()
        if not reply:
            self.close()
            raise OCRUnavailable("OCR worker exited unexpectedly.")
        res = json.loads(reply)
        if "error" in res:
            raise OCRUnavailable(res["error"])
        self.last_latency_s = res.get("latency_s")
        return res["lines"]

    def close(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self._proc = None


_client: OCRClient | None = None


def get_ocr() -> OCRClient:
    global _client
    if _client is None:
        _client = OCRClient()
    return _client
