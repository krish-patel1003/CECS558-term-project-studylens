"""FastAPI backend + static single-page UI.

Endpoints
---------
GET  /                          the web UI (studylens/static/index.html): "Ask" and "Evaluation" tabs
GET  /api/health                backend, whether it is AI, LoRA adapter, AI2D availability
GET  /api/samples               bundled synthetic sample diagrams
GET  /api/samples/{f}           a bundled sample PNG
POST /api/extract               multipart image -> extracted graph JSON
POST /api/extract_stream        same, streamed as NDJSON progress events (used by the UI progress bar)
POST /api/ask_stream            multipart image + question + graph (+ options, cached OCR) -> NDJSON
                                  progress events of the three pipelines E1 (OCR), E2 (image) and E3
                                  (graph query), used by the live pipeline view
POST /api/ask                   multipart image + question + graph (+ options) ->
                                  "nl":    natural-language answer from Qwen with cited arrows,
                                           verified deterministically (E3)          [qwen backend]
                                  "exact": rule-based NetworkX query, if the question matches a
                                           supported structural pattern (cross-check)
POST /api/ask_exact             {graph, question} -> rule-based answer only (no model)
POST /api/ask_direct            multipart image + question (+ options) -> image-only answer (E2)
GET  /api/eval                  list of evaluation runs in results/
GET  /api/eval/{run}            one run: summary + per-question records (ids only)
GET  /api/eval/{run}/item/{qid} question text/options for a record, from the LOCAL AI2D copy
GET  /api/ai2d/image/{image}    an AI2D image from the LOCAL copy (never bundled in the repo)

The server is stateless: the graph lives in the browser and is sent back with each question,
so the user can correct the graph before asking.
"""

from __future__ import annotations

import json
import re
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.formparsers import MultiPartParser

from . import pipelines, qa_nl
from .config import ROOT, settings
from .extractors import ExtractorUnavailable, get_extractor
from .graph_qa import answer_question
from .ocr import get_ocr
from .schema import DiagramGraph
from .uploads import ALLOWED_FORMATS, UploadError, load_upload

STATIC = ROOT / "studylens" / "static"
RESULTS = ROOT / "results"

_LIMIT = settings.max_upload_mb * 1024 * 1024

# Starlette spools multipart files larger than 1 MB to an OS temp file. Raise that threshold
# to the upload limit so accepted uploads stay in memory only (see README "Data handling").
MultiPartParser.max_file_size = _LIMIT + 1

app = FastAPI(title="StudyLens", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def reject_large_bodies(request: Request, call_next):
    """Refuse oversized requests before the multipart body is parsed (and possibly spooled)."""
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > _LIMIT + 64 * 1024:
        return JSONResponse({"detail": f"File too large (limit {settings.max_upload_mb} MB)."}, status_code=413)
    return await call_next(request)


class AskExactRequest(BaseModel):
    graph: dict
    question: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    from .datasets import ai2d

    ext = get_extractor(settings.extractor_backend)
    return {
        "status": "ok",
        "extractor": ext.name,
        "extractor_is_ai": ext.is_ai,
        "adapter_path": getattr(ext, "adapter_path", "") or None,
        "nl_qa_available": ext.is_ai,
        "direct_vlm_available": settings.enable_direct_vlm and ext.is_ai,
        "ai2d_local": ai2d.available(),
        "ocr_available": get_ocr().available(),
        "formats": sorted(ALLOWED_FORMATS),
        "max_upload_mb": settings.max_upload_mb,
    }


@app.get("/api/samples")
def samples():
    path = settings.synthetic_dir / "manifest.json"
    if not path.exists():
        return {"samples": []}
    manifest = json.loads(path.read_text())
    return {
        "samples": [
            {"file": d["file"], "title": d["title"], "example_questions": [q["question"] for q in d["questions"]][:4]}
            for d in manifest["diagrams"]
        ]
    }


@app.get("/api/samples/{name}")
def sample_image(name: str):
    path = (settings.synthetic_dir / name).resolve()
    if path.parent != settings.synthetic_dir.resolve() or path.suffix != ".png" or not path.exists():
        raise HTTPException(404, "sample not found")
    return FileResponse(path)


async def _read_upload(file: UploadFile) -> bytes:
    # Read at most limit+1 bytes so an oversized upload is rejected without buffering all of it.
    data = await file.read(_LIMIT + 1)
    if len(data) > _LIMIT:
        raise HTTPException(413, f"File too large (limit {settings.max_upload_mb} MB).")
    return data


async def _image(file: UploadFile):
    data = await _read_upload(file)
    try:
        return load_upload(data), data
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc


def _parse_graph(graph_json: str | dict) -> DiagramGraph:
    try:
        d = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        return DiagramGraph.from_dict(d)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"invalid graph: {exc}") from exc


def _parse_choices(choices: str | None) -> list[str] | None:
    if not choices:
        return None
    try:
        c = json.loads(choices)
    except json.JSONDecodeError:
        c = [x.strip() for x in choices.split("\n")]
    c = [str(x).strip() for x in c if str(x).strip()]
    if len(c) > 8:
        raise HTTPException(400, "at most 8 options")
    return c or None


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    img, data = await _image(file)
    ext = get_extractor(settings.extractor_backend)
    try:
        graph = ext.extract(img, image_bytes=data)
    except ExtractorUnavailable as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        del data  # uploads are never persisted
    out = graph.to_dict()
    out["is_ai"] = ext.is_ai
    out["latency_s"] = getattr(ext, "last_latency_s", None)
    out["raw_output"] = getattr(ext, "last_raw_output", None)
    return out


@app.post("/api/extract_stream")
async def extract_stream(file: UploadFile = File(...)):
    """Like /api/extract, but streams newline-delimited JSON progress events so the UI can show
    real progress: loading_model -> encoding -> generating (token / entity / arrow counts) -> done.
    The last line is {"stage": "done", "graph": {...}} or {"stage": "error", "detail": ...}."""
    img, data = await _image(file)
    ext = get_extractor(settings.extractor_backend)
    use_dsg = (ext.is_ai and hasattr(ext, "_generate") and settings.extraction_mode == "dsg"
               and get_ocr().available())

    t0 = time.perf_counter()

    def events():
        def line(d):
            return json.dumps(d) + "\n"

        try:
            if use_dsg:
                from .dsg import DSGExtractor

                if getattr(ext, "_model", None) is None:
                    yield line({"stage": "loading_model"})
                    ext._load()
                for ev in DSGExtractor(ext, get_ocr()).extract_stream(img):
                    if ev["stage"] == "done":
                        out = ev["graph"].to_dict()
                        out.update(is_ai=True, latency_s=round(time.perf_counter() - t0, 1), raw_output=None)
                        yield line({"stage": "done", "graph": out})
                    else:
                        yield line(ev)
            elif hasattr(ext, "extract_stream"):
                for ev in ext.extract_stream(img):
                    if ev["stage"] == "done":
                        out = ev["graph"].to_dict()
                        out.update(is_ai=ext.is_ai, latency_s=ext.last_latency_s, raw_output=ext.last_raw_output)
                        yield line({"stage": "done", "graph": out})
                    else:
                        yield line(ev)
            else:  # non-streaming backend (annotation fallback): one step
                yield line({"stage": "encoding"})
                g = ext.extract(img, image_bytes=data).to_dict()
                g.update(is_ai=ext.is_ai, latency_s=None, raw_output=None)
                yield line({"stage": "done", "graph": g})
        except ExtractorUnavailable as exc:
            yield line({"stage": "error", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report to the UI instead of a broken stream
            yield line({"stage": "error", "detail": f"extraction failed: {exc}"})

    # Starlette runs a sync generator in a thread pool, so the event loop is not blocked.
    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/ask_stream")
async def ask_stream(file: UploadFile = File(...), question: str = Form(...), graph: str = Form(...),
                     choices: str | None = Form(None), ocr: str | None = Form(None),
                     pipelines_: str = Form("E3,E2,E1", alias="pipelines")):
    """Run the selected pipelines one after another (they share one GPU) and stream every stage
    change as a JSON line: {"pipeline", "stage", "state", ...}; each pipeline ends with
    {"pipeline", "result"}; the stream ends with {"done": true}. ``ocr`` lets the browser send back
    OCR lines from an earlier question about the same image (OCR runs once per image)."""
    g = _parse_graph(graph)
    opts = _parse_choices(choices)
    if not question.strip():
        raise HTTPException(400, "empty question")
    img, data = await _image(file)
    del data
    ocr_lines = None
    if ocr:
        try:
            ocr_lines = [{"text": str(x["text"]), "score": float(x.get("score", 0)), "box": [int(v) for v in x["box"]][:4]}
                         for x in json.loads(ocr)][:300]
        except (ValueError, KeyError, TypeError):
            ocr_lines = None
    order = [p for p in (x.strip() for x in pipelines_.split(",")) if p in pipelines.STAGES]
    ext = get_extractor(settings.extractor_backend)
    vlm = ext if ext.is_ai else None

    def events():
        yield json.dumps({"start": True, "order": order, "stages": {p: pipelines.STAGES[p] for p in order}}) + "\n"
        for p in order:
            try:
                if p == "EX":
                    loc = None
                    if vlm is not None:
                        from .dsg import DSGExtractor
                        loc = DSGExtractor(vlm, None, arrow_detector=None).ground
                    gen = pipelines.run_explain(vlm, img, g, question, opts, locator=loc)
                elif p == "E1":
                    gen = pipelines.run_e1(vlm, get_ocr(), img, question, opts, ocr_lines)
                elif p == "E2":
                    gen = pipelines.run_e2(vlm, img, question, opts)
                else:
                    gen = pipelines.run_e3(vlm, img, g, question, opts)
                for ev in gen:
                    yield json.dumps(ev, default=str) + "\n"
            except Exception as exc:  # noqa: BLE001 - one failing pipeline must not break the others
                yield json.dumps({"pipeline": p, "stage": "answer", "state": "error", "detail": str(exc)}) + "\n"
                yield json.dumps({"pipeline": p, "result": {"answer": None, "status": "error", "error": str(exc)}}) + "\n"
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _exact(question: str, graph: DiagramGraph) -> dict:
    r = answer_question(question, graph).to_dict()
    r["recognized"] = r["intent"] not in (None, "unknown", "unsupported")
    return r


@app.post("/api/ask")
async def ask(file: UploadFile | None = File(None), question: str = Form(...), graph: str = Form(...),
              choices: str | None = Form(None)):
    g = _parse_graph(graph)
    opts = _parse_choices(choices)
    out = {"exact": _exact(question, g), "nl": None}
    ext = get_extractor(settings.extractor_backend)
    if ext.is_ai and file is not None:
        img, data = await _image(file)
        del data
        try:
            out["nl"] = qa_nl.grounded_answer(ext, img, question, g, opts)
        except ExtractorUnavailable as exc:
            raise HTTPException(422, str(exc)) from exc
    return out


@app.post("/api/ask_exact")
def ask_exact(req: AskExactRequest):
    return _exact(req.question, _parse_graph(req.graph))


@app.post("/api/ask_direct")
async def ask_direct(file: UploadFile = File(...), question: str = Form(...), choices: str | None = Form(None)):
    ext = get_extractor(settings.extractor_backend)
    if not (settings.enable_direct_vlm and ext.is_ai):
        raise HTTPException(422, "Direct VLM answers need EXTRACTOR_BACKEND=qwen.")
    img, data = await _image(file)
    del data
    try:
        return qa_nl.direct_answer(ext, img, question, _parse_choices(choices))
    except ExtractorUnavailable as exc:
        raise HTTPException(422, str(exc)) from exc


# ------------------------------------------------------------------------------------ evaluation
_RUN_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _run_path(run: str):
    if not _RUN_RE.match(run):
        raise HTTPException(404, "unknown run")
    p = RESULTS / f"{run}.json"
    if not p.exists():
        raise HTTPException(404, "unknown run")
    return p


@app.get("/api/eval")
def eval_runs():
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        runs.append({"run": p.stem, "title": d.get("title", p.stem), "dataset": d.get("dataset", "synthetic"),
                     "created": d.get("created"), "n_questions": d.get("summary", {}).get("n_questions")})
    # AI2D runs first
    runs.sort(key=lambda r: (r["dataset"] != "ai2d", r["run"]))
    return {"runs": runs}


@app.get("/api/eval/{run}")
def eval_run(run: str):
    return json.loads(_run_path(run).read_text())


@app.get("/api/eval/{run}/item/{qid}")
def eval_item(run: str, qid: str):
    """Join a result record with question text from the user's LOCAL AI2D copy (not redistributed)."""
    from .datasets import ai2d

    _run_path(run)
    if not ai2d.available():
        raise HTTPException(404, "AI2D is not downloaded on this machine (python scripts/get_ai2d.py)")
    img_id = qid.split("-")[0]
    qpath = ai2d.AI2D_DIR / "questions" / f"{img_id}.json"
    if not re.match(r"^\d+\.png$", img_id) or not qpath.exists():
        raise HTTPException(404, "unknown question")
    for text, q in json.loads(qpath.read_text())["questions"].items():
        if q["questionId"] == qid:
            return {"question": text, "choices": q["answerTexts"], "answer_index": q["correctAnswer"], "image_id": img_id}
    raise HTTPException(404, "unknown question")


@app.get("/api/ai2d/image/{image_id}")
def ai2d_image(image_id: str):
    from .datasets import ai2d

    if not re.match(r"^\d+\.png$", image_id) or not ai2d.available():
        raise HTTPException(404, "not available")
    p = ai2d.AI2D_DIR / "images" / image_id
    if not p.exists():
        raise HTTPException(404, "not available")
    return FileResponse(p)
