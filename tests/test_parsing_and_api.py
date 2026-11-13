import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studylens.app import app
from studylens.config import settings
from studylens.extractors.qwen_vl import parse_extraction
from studylens.uploads import UploadError, load_upload

client = TestClient(app)
MANIFEST = json.loads((settings.synthetic_dir / "manifest.json").read_text())


# ---------------------------------------------------------------- VLM output parsing (no model needed)
def test_parse_extraction_fenced_and_trailing_comma():
    raw = '```json\n{"entities":[{"label":"A","bbox_2d":[0,0,10,10]},{"label":"B"}],"arrows":[{"from":"A","to":"B"},]}\n```'
    g = parse_extraction(raw, scale=(0.5, 0.5))
    assert [n.label for n in g.nodes] == ["A", "B"]
    assert g.nodes[0].bbox == [0, 0, 20, 20]  # mapped back from the model's grid
    assert [(e.source, e.target) for e in g.edges] == [("n1", "n2")]


def test_parse_extraction_drops_unknown_endpoint():
    g = parse_extraction('{"entities":[{"label":"A"}],"arrows":[{"from":"A","to":"Ghost"}]}')
    assert g.edges == [] and "Ghost" in g.warnings[0]


def test_parse_extraction_salvages_output_cut_off_at_token_limit():
    # the kind of output that hit QWEN_MAX_NEW_TOKENS on a dense architecture diagram
    raw = ('{"entities":[{"label":"Queue","bbox_2d":[10,10,50,30]},{"label":"Transcoder","bbox_2d":[60,10,120,30]},'
           '{"label":"S3","bbox_2d":[130,10,160,30]},{"label":"Kaf')
    g = parse_extraction(raw, truncated=True)
    assert [n.label for n in g.nodes] == ["Queue", "Transcoder", "S3"]
    assert any("output limit" in w and "3 entities" in w for w in g.warnings)


def test_parse_extraction_salvages_arrows_and_skips_unfinished_arrow():
    raw = ('{"entities":[{"label":"A"},{"label":"B"},{"label":"C"}],'
           '"arrows":[{"from":"A","to":"B"},{"from":"B","to":"C"},{"from":"C","to"')
    g = parse_extraction(raw, truncated=True)
    assert [(g.label_of(e.source), g.label_of(e.target)) for e in g.edges] == [("A", "B"), ("B", "C")]


def test_parse_extraction_complete_but_truncated_flag_warns():
    g = parse_extraction('{"entities":[{"label":"A"}],"arrows":[]}', truncated=True)
    assert len(g.nodes) == 1 and any("output limit" in w for w in g.warnings)


def test_parse_extraction_garbage():
    g = parse_extraction("I cannot see a diagram.")
    assert g.nodes == [] and g.warnings


# ---------------------------------------------------------------- upload validation
def _png(size=(20, 20)):
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_upload_rejects_non_image():
    with pytest.raises(UploadError):
        load_upload(b"%PDF-1.4 not an image")


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF"])
def test_upload_accepts_common_formats(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(buf, format=fmt)
    img = load_upload(buf.getvalue())
    assert img.mode == "RGB" and img.size == (120, 80)


def test_upload_flattens_transparency_and_upscales_tiny():
    buf = io.BytesIO()
    Image.new("RGBA", (20, 10), (255, 0, 0, 0)).save(buf, format="PNG")
    img = load_upload(buf.getvalue())
    assert img.mode == "RGB" and min(img.size) >= 64
    assert img.getpixel((0, 0)) == (255, 255, 255)  # transparent -> white


def test_upload_applies_exif_rotation():
    buf = io.BytesIO()
    im = Image.new("RGB", (200, 100), "white")
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 degrees
    im.save(buf, format="JPEG", exif=exif)
    assert load_upload(buf.getvalue()).size == (100, 200)


def test_upload_rejects_pdf_bytes_and_ico():
    with pytest.raises(UploadError):
        load_upload(b"%PDF-1.7\n%garbage")
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="ICO")
    with pytest.raises(UploadError):
        load_upload(buf.getvalue())


def test_upload_accepts_png():
    assert load_upload(_png((100, 100))).size == (100, 100)


# ---------------------------------------------------------------- API end-to-end with the annotation backend
def test_end_to_end_annotation_backend(monkeypatch):
    monkeypatch.setattr("studylens.app.settings", settings.__class__(extractor_backend="annotation"))
    d = MANIFEST["diagrams"][0]
    data = (settings.synthetic_dir / d["file"]).read_bytes()
    r = client.post("/api/extract", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 200
    graph = r.json()
    assert graph["is_ai"] is False and "not AI" in graph["source"]

    r = client.post("/api/ask", data={"graph": json.dumps(graph), "question": "Which component directly feeds into C?"})
    body = r.json()["exact"]
    assert r.json()["nl"] is None  # no model in the annotation backend
    assert body["status"] == "answered" and body["evidence"][0]["text"] == "B -> C"

    r = client.post("/api/ask_exact", json={"graph": graph, "question": "Which component directly feeds into A?"})
    assert r.json()["status"] == "abstained"


def test_annotation_backend_rejects_unknown_image(monkeypatch):
    monkeypatch.setattr("studylens.app.settings", settings.__class__(extractor_backend="annotation"))
    r = client.post("/api/extract", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 422 and "qwen" in r.json()["detail"]


def test_extract_rejects_bad_file():
    r = client.post("/api/extract", files={"file": ("x.png", b"hello", "image/png")})
    assert r.status_code == 400


def test_sample_path_traversal_blocked():
    assert client.get("/api/samples/..%2F..%2FREADME.md").status_code == 404
    assert client.get("/api/samples/diagram_01.png").status_code == 200


def test_oversized_request_rejected_before_parsing():
    big = b"0" * (settings.max_upload_mb * 1024 * 1024 + 200_000)
    r = client.post("/api/extract", files={"file": ("x.png", big, "image/png")})
    assert r.status_code == 413


def test_uploads_kept_in_memory():
    from starlette.formparsers import MultiPartParser
    assert MultiPartParser.max_file_size > settings.max_upload_mb * 1024 * 1024


def _ndjson(r):
    return [json.loads(l) for l in r.text.splitlines() if l.strip()]


def test_extract_stream_annotation_backend(monkeypatch):
    monkeypatch.setattr("studylens.app.settings", settings.__class__(extractor_backend="annotation"))
    data = (settings.synthetic_dir / MANIFEST["diagrams"][0]["file"]).read_bytes()
    r = client.post("/api/extract_stream", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    ev = _ndjson(r)
    assert ev[-1]["stage"] == "done" and len(ev[-1]["graph"]["edges"]) == 3


def test_extract_stream_reports_progress_and_errors(monkeypatch):
    from studylens.schema import DiagramGraph

    class StreamStub:
        name, is_ai, last_latency_s, last_raw_output = "stub", True, 1.0, "{}"

        def __init__(self, fail=False):
            self.fail = fail

        def extract_stream(self, image):
            yield {"stage": "loading_model"}
            yield {"stage": "encoding"}
            yield {"stage": "generating", "tokens": 12, "entities": 2, "arrows": 1, "elapsed_s": 0.5}
            if self.fail:
                raise RuntimeError("boom")
            yield {"stage": "done", "graph": DiagramGraph(source="stub")}

    for fail in (False, True):
        monkeypatch.setattr("studylens.app.get_extractor", lambda name, f=fail: StreamStub(f))
        r = client.post("/api/extract_stream", files={"file": ("x.png", _png((100, 100)), "image/png")})
        stages = [e["stage"] for e in _ndjson(r)]
        assert stages[:3] == ["loading_model", "encoding", "generating"]
        assert stages[-1] == ("error" if fail else "done")


def test_extract_stream_rejects_bad_upload():
    r = client.post("/api/extract_stream", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
