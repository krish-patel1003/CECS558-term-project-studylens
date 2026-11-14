"""Configuration from environment variables (optionally loaded from a local, untracked .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    extractor_backend: str = os.environ.get("EXTRACTOR_BACKEND", "annotation")
    qwen_model_id: str = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
    device: str = os.environ.get("STUDYLENS_DEVICE", "auto")  # auto | cuda | mps | cpu
    # Each digit of a box coordinate is its own token (~28 tokens per entity), so dense diagrams
    # need a large budget; output cut off at the limit is salvaged (see parse_extraction).
    max_new_tokens: int = _int("QWEN_MAX_NEW_TOKENS", 3072)
    # Longest image side passed to the VLM. Training used 448 px (memory), but at inference a
    # larger side keeps small labels readable. scripts/evaluate_ai2d.py pins 448 to match training.
    max_image_side: int = _int("QWEN_MAX_IMAGE_SIDE", 1024)
    # LoRA adapter for extraction produced by scripts/train_lora.py ("" = zero-shot base model)
    qwen_adapter_path: str = os.environ.get("QWEN_ADAPTER_PATH", "")
    # "extraction": legacy whole-graph adapter (scripts/train_lora.py); "dsg": grounding + arrows adapter
    # (scripts/train_dsg_lora.py), used by the Diagram Scene Graph calls only
    adapter_kind: str = os.environ.get("ADAPTER_KIND", "extraction")
    # "dsg" = OCR-anchored Diagram Scene Graph (needs the OCR venv); "legacy" = one-call JSON extraction
    extraction_mode: str = os.environ.get("EXTRACTION_MODE", "dsg")
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 10)
    max_image_pixels: int = _int("MAX_IMAGE_PIXELS", 40_000_000)
    enable_direct_vlm: bool = os.environ.get("ENABLE_DIRECT_VLM", "1") == "1"
    # Interpreter of the separate PaddleOCR environment used by the E1 baseline (see docs/code_guide.md)
    ocr_python: str = str(ROOT / os.environ.get("OCR_PYTHON", "external/venv-ocr/bin/python"))  # relative to the repo root
    synthetic_dir: Path = ROOT / "data" / "synthetic"


settings = Settings()
