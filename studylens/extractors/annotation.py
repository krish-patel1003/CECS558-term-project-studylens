"""NON-AI fallback backend: returns the ground-truth graph of a bundled synthetic diagram.

The synthetic generator (``scripts/make_synthetic.py``) writes ``data/synthetic/manifest.json``
with the SHA-256 of every PNG and its gold graph. This backend hashes the uploaded bytes and,
if they match a bundled sample, returns that gold graph labelled ``source="annotation (ground
truth, not AI)"``. Any other image is rejected with a clear message. It exists so the upload ->
graph -> question -> evidence workflow can be demonstrated without the 3B-parameter model, and
it is the "E3*" (ground-truth graph) condition in the evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from ..config import settings
from ..schema import DiagramGraph
from .base import ExtractorUnavailable


class AnnotationExtractor:
    name = "annotation"
    is_ai = False

    def __init__(self, manifest_path: Path | None = None):
        self.manifest_path = manifest_path or settings.synthetic_dir / "manifest.json"
        self._by_hash: dict[str, dict] = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            for item in manifest["diagrams"]:
                self._by_hash[item["sha256"]] = item

    def extract(self, image: Image.Image, image_bytes: bytes | None = None) -> DiagramGraph:
        if image_bytes is None:
            raise ExtractorUnavailable("annotation backend needs the raw image bytes")
        digest = hashlib.sha256(image_bytes).hexdigest()
        item = self._by_hash.get(digest)
        if item is None:
            raise ExtractorUnavailable(
                "The 'annotation' backend is a non-AI fallback that only knows the bundled sample "
                "diagrams in data/synthetic/. To extract a graph from a new image, run with "
                "EXTRACTOR_BACKEND=qwen (requires the Qwen2.5-VL-3B weights)."
            )
        g = DiagramGraph.from_dict(item["graph"])
        g.source = "annotation (ground truth, not AI)"
        g.image_size = list(image.size)
        return g
