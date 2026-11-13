"""AI backend: Qwen2.5-VL-3B-Instruct, run locally with Hugging Face transformers.

Two uses:
1. ``extract``       image -> JSON entities (with boxes) + arrows -> ``DiagramGraph``  (used by E3 / the app)
2. ``answer_direct`` image + question -> free-text answer, no graph                  (baseline E2)

The prompts are module-level constants so they are documented in one place
(see docs/model_and_prompts.md). Model weights are pulled from the Hugging Face Hub
on first use into the standard HF cache; nothing is written to the repository.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
import time

from PIL import Image

from ..config import settings
from ..schema import DiagramGraph, Edge, Node
from .base import ExtractorUnavailable

EXTRACTION_PROMPT = """You are reading a diagram. Extract its structure as a directed graph.

1. List every labelled entity (a box, shape or object with a text label). Copy each label exactly as written. Give its bounding box as [x1, y1, x2, y2] in pixel coordinates of this image.
2. List every arrow that is actually drawn. An arrow goes FROM the entity at its tail TO the entity its arrowhead touches. Look carefully at which end has the arrowhead. Do not add arrows that are not drawn, even if they would make sense.

Return ONLY compact JSON (no indentation, no code fences) in exactly this format, with no explanation:
{"entities": [{"label": "...", "bbox_2d": [x1, y1, x2, y2]}], "arrows": [{"from": "<entity label>", "to": "<entity label>"}]}"""

DIRECT_QA_PROMPT = """Answer the question about this diagram using only what the diagram shows. Answer in one short sentence and name entities exactly as they are labelled. If the diagram does not show enough to answer, reply exactly: Not shown in the diagram.

Question: {question}"""


class QwenVLExtractor:
    name = "qwen2.5-vl-3b"
    is_ai = True

    def __init__(self, model_id: str | None = None, adapter_path: str | None = None):
        self.model_id = model_id or settings.qwen_model_id
        # Optional LoRA adapter (scripts/train_lora.py). Used ONLY for graph extraction; question
        # answering always runs with the adapter disabled (base weights).
        self.adapter_path = adapter_path if adapter_path is not None else settings.qwen_adapter_path
        self.has_adapter = False
        self._model = None
        self._processor = None
        self._device = None
        self._lock = threading.Lock()
        self.last_raw_output: str | None = None
        self.last_latency_s: float | None = None
        self.last_truncated: bool = False

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ExtractorUnavailable(
                f"Qwen backend needs torch + transformers (pip install -r requirements-model.txt): {exc}"
            ) from exc

        dev = settings.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if dev == "cpu":
            dtype = torch.float32
        elif dev == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16  # e.g. Colab's free T4
        else:
            dtype = torch.bfloat16
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, dtype=dtype, attn_implementation="sdpa"
            )
            processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:  # pragma: no cover - depends on env / network
            raise ExtractorUnavailable(f"Could not load {self.model_id}: {exc}") from exc
        model = model.to(dev)
        if self.adapter_path:
            from pathlib import Path

            from peft import PeftModel

            if not (Path(self.adapter_path) / "adapter_config.json").exists():
                raise ExtractorUnavailable(f"LoRA adapter not found at {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)
            self.has_adapter = True
            self.name = "qwen2.5-vl-3b + AI2D LoRA"
        self._model = model.eval()
        self._processor = processor
        self._device = dev

    # ------------------------------------------------------------------ generation
    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int, use_adapter: bool = False) -> tuple[str, tuple[float, float]]:
        """Run one image+text turn. Returns (decoded text, scale from original image to model grid)."""
        import contextlib

        import torch

        self._load()
        img = prepare_image(image, settings.max_image_side)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[img], return_tensors="pt").to(self._device)
        # The processor resizes to multiples of 28 px; Qwen2.5-VL boxes refer to THAT grid.
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        seen_size = (int(gw) * 14, int(gh) * 14)
        ctx = self._model.disable_adapter() if (self.has_adapter and not use_adapter) else contextlib.nullcontext()
        with self._lock, torch.inference_mode(), ctx:
            t0 = time.perf_counter()
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            self.last_latency_s = time.perf_counter() - t0
        gen = out[0, inputs["input_ids"].shape[1] :]
        self.last_truncated = _hit_limit(gen, max_new_tokens, self._processor.tokenizer)
        decoded = self._processor.decode(gen, skip_special_tokens=True)
        self.last_raw_output = decoded
        # scale factor from the model's grid back to the ORIGINAL image
        return decoded, (seen_size[0] / image.size[0], seen_size[1] / image.size[1])

    # ------------------------------------------------------------------ public API
    def extract(self, image: Image.Image, image_bytes: bytes | None = None, use_adapter: bool = True) -> DiagramGraph:
        """Image -> graph. Uses the LoRA adapter if one is configured (and use_adapter is True)."""
        self._load()
        use = use_adapter and self.has_adapter
        raw, (sx, sy) = self._generate(image, EXTRACTION_PROMPT, settings.max_new_tokens, use_adapter=use)
        g = parse_extraction(raw, scale=(sx, sy), truncated=self.last_truncated)
        g.source = "qwen2.5-vl-3b + AI2D LoRA" if use else "qwen2.5-vl-3b (zero-shot)"
        g.image_size = list(image.size)
        return g

    def extract_stream(self, image: Image.Image, use_adapter: bool = True):
        """Same as ``extract`` but yields progress events while the model writes its output.

        Events (dicts): {"stage": "loading_model"} (first call only), {"stage": "encoding"},
        {"stage": "generating", "tokens", "entities", "arrows", "elapsed_s"} (every few tokens;
        entity/arrow counts are read from the partial JSON), and finally
        {"stage": "done", "graph": DiagramGraph}. Errors in the generation thread are re-raised.
        """
        import contextlib
        import threading

        import torch
        from transformers import TextIteratorStreamer

        if self._model is None:
            yield {"stage": "loading_model"}
        self._load()
        use = use_adapter and self.has_adapter
        yield {"stage": "encoding"}
        img = prepare_image(image, settings.max_image_side)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": EXTRACTION_PROMPT}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[img], return_tensors="pt").to(self._device)
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        scale = (int(gw) * 14 / image.size[0], int(gh) * 14 / image.size[1])
        streamer = TextIteratorStreamer(self._processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600)
        ctx = self._model.disable_adapter() if (self.has_adapter and not use) else contextlib.nullcontext()
        err: list[BaseException] = []
        result: list = []

        def run():
            try:
                with self._lock, torch.inference_mode(), ctx:
                    result.append(self._model.generate(**inputs, max_new_tokens=settings.max_new_tokens,
                                                       do_sample=False, streamer=streamer))
            except BaseException as exc:  # surfaced to the caller below
                err.append(exc)
                streamer.end()

        t0 = time.perf_counter()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        out, n = "", 0
        for piece in streamer:
            out += piece
            n += 1
            if n % 6 == 0:
                yield {"stage": "generating", "tokens": n, "entities": out.count('"label"'),
                       "arrows": out.count('"from"'), "elapsed_s": round(time.perf_counter() - t0, 1)}
        worker.join()
        if err:
            raise err[0]
        self.last_latency_s = time.perf_counter() - t0
        self.last_raw_output = out
        self.last_truncated = bool(result) and _hit_limit(
            result[0][0, inputs["input_ids"].shape[1]:], settings.max_new_tokens, self._processor.tokenizer)
        g = parse_extraction(out, scale=scale, truncated=self.last_truncated)
        g.source = "qwen2.5-vl-3b + AI2D LoRA" if use else "qwen2.5-vl-3b (zero-shot)"
        g.image_size = list(image.size)
        yield {"stage": "done", "graph": g}

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 96) -> str:
        """Free-form image+text generation with the BASE weights (adapter disabled). Used for QA."""
        raw, _ = self._generate(image, prompt, max_new_tokens, use_adapter=False)
        return raw.strip()

    def generate_text(self, prompt: str, max_new_tokens: int = 96) -> str:
        """Text-only generation (no image) with the BASE weights. Used by the E1 OCR baseline and by
        the graph-query parser, neither of which may see the image."""
        import contextlib

        import torch

        self._load()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], return_tensors="pt").to(self._device)
        ctx = self._model.disable_adapter() if self.has_adapter else contextlib.nullcontext()
        with self._lock, torch.inference_mode(), ctx:
            t0 = time.perf_counter()
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            self.last_latency_s = time.perf_counter() - t0
        decoded = self._processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_raw_output = decoded
        return decoded.strip()

    def answer_direct(self, image: Image.Image, question: str) -> str:
        return self.generate(image, DIRECT_QA_PROMPT.format(question=question), 96)


# ---------------------------------------------------------------------- helpers
def prepare_image(image: Image.Image, max_side: int) -> Image.Image:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    return img


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def _hit_limit(gen_ids, max_new_tokens: int, tokenizer) -> bool:
    """True if generation stopped because it used the whole token budget (no end-of-turn token)."""
    if len(gen_ids) < max_new_tokens:
        return False
    eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return int(gen_ids[-1]) not in eos


_INNER_OBJ = re.compile(r"\{[^{}]*\}")


def _salvage(raw: str) -> dict | None:
    """Recover the COMPLETE entity/arrow objects from output that is not valid JSON as a whole,
    typically because generation was cut off at the token limit. Every innermost {...} that parses
    is kept; the unfinished last item is dropped. Returns None if nothing could be recovered."""
    ents, arrows = [], []
    for m in _INNER_OBJ.finditer(raw or ""):
        frag = re.sub(r",\s*}", "}", m.group(0))
        try:
            obj = json.loads(frag)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "label" in obj:
            ents.append(obj)
        elif ("from" in obj and "to" in obj) or ("source" in obj and "target" in obj):
            arrows.append(obj)
    return {"entities": ents, "arrows": arrows} if (ents or arrows) else None


def _load_json(raw: str) -> dict:
    """Parse model output into a dict, tolerating code fences, prose and trailing commas."""
    txt = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    blob = txt[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(blob)


def parse_extraction(raw: str, scale: tuple[float, float] = (1.0, 1.0), truncated: bool = False) -> DiagramGraph:
    """Convert the model's JSON text into a validated ``DiagramGraph``.

    * Entity boxes are mapped from the model's resized-image grid back to original pixels.
    * Arrow endpoints are matched to entity labels (exact, then fuzzy); unmatched endpoints are
      dropped with a warning rather than invented as new nodes.
    * Output that is not valid JSON as a whole (usually cut off at the token limit) is salvaged:
      every complete entity/arrow object is kept and a warning says the graph may be incomplete.
    """
    warnings: list[str] = []
    try:
        data = _load_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        data = _salvage(raw)
        if data is None:
            return DiagramGraph(warnings=[f"could not parse model output as JSON: {exc}"])
        why = ("the model reached its output limit (QWEN_MAX_NEW_TOKENS) before finishing"
               if truncated else "the model output was not valid JSON")
        warnings.append(f"Partial graph: {why}; kept the {len(data['entities'])} entities and "
                        f"{len(data['arrows'])} arrows that were complete. Entities or arrows may be missing.")
    else:
        if truncated:
            warnings.append("The model reached its output limit; the graph may be incomplete.")

    sx, sy = scale
    nodes: list[Node] = []
    by_label: dict[str, str] = {}
    for ent in data.get("entities", []) or []:
        if isinstance(ent, str):
            ent = {"label": ent}
        label = str(ent.get("label", "")).strip()
        if not label or _norm(label) in by_label:
            continue
        nid = f"n{len(nodes) + 1}"
        bbox = ent.get("bbox_2d") or ent.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                bbox = [float(bbox[0]) / sx, float(bbox[1]) / sy, float(bbox[2]) / sx, float(bbox[3]) / sy]
            except (TypeError, ValueError):
                bbox = None
        else:
            bbox = None
        nodes.append(Node(id=nid, label=label, bbox=bbox))
        by_label[_norm(label)] = nid

    def resolve(lbl) -> str | None:
        key = _norm(lbl)
        if key in by_label:
            return by_label[key]
        close = difflib.get_close_matches(key, by_label.keys(), n=1, cutoff=0.8)
        return by_label[close[0]] if close else None

    edges: list[Edge] = []
    for arr in data.get("arrows", []) or data.get("edges", []) or []:
        if not isinstance(arr, dict):
            continue
        s_lbl = arr.get("from", arr.get("source"))
        t_lbl = arr.get("to", arr.get("target"))
        s, t = resolve(s_lbl), resolve(t_lbl)
        if s is None or t is None:
            warnings.append(f"dropped arrow {s_lbl!r} -> {t_lbl!r}: endpoint is not a listed entity")
            continue
        edges.append(Edge(source=s, target=t, relation="points_to"))

    g = DiagramGraph(nodes=nodes, edges=edges, warnings=warnings)
    g.validate()
    return g
