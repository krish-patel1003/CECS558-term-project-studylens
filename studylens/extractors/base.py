from __future__ import annotations

from typing import Protocol

from PIL import Image

from ..schema import DiagramGraph


class ExtractorUnavailable(RuntimeError):
    """Raised when a backend cannot run in this environment (missing weights, no torch, ...)."""


class Extractor(Protocol):
    name: str
    is_ai: bool  # True only for backends that run a learned model

    def extract(self, image: Image.Image, image_bytes: bytes | None = None) -> DiagramGraph: ...
