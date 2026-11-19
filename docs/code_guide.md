# Code guide

```
studylens/
  app.py                 FastAPI app: extract(_stream) / ask_stream / ask / ask_direct / ask_exact / eval endpoints, upload caps
  config.py              All settings, read from environment variables / .env
  schema.py              DiagramGraph / Node / Edge: the shared graph format (+ to_networkx)
  uploads.py             Image intake: format detection, EXIF, transparency, colour modes, size limits (in memory)
  dsg.py                 Diagram Scene Graph extraction: OCR parts (no AI), arrows (detector, else Qwen by name), regions
  arrows.py              Arrow detector: Mask R-CNN model, ArrowDetector.detect, link_arrows (ends -> nearest parts)
  regions.py             Image analysis (no AI): enclosing shape of a label; experimental leader-line tracer
  explain.py             StudyLens answer: graph facts + Qwen explanation + deterministic reference linking;
                         gate_boxes hides highlights whose position came only from the label text
  pipelines.py           Pipelines as event generators: EX (StudyLens answer), E1 / E2 / E3 (comparison);
                         combine_e3 (E3 decision rule)
  query.py               E3 core: question -> structured graph query (rules, food-web rules, Qwen parser) ->
                         NetworkX -> answer + arrows; deterministic option matching (incl. NOT questions)
  graph_qa.py            Keyword rules (rule_parse) and NetworkX operations (execute): no AI
  qa_nl.py               Prompts and parsing for E1 (OCR text), E2 (image), E3 fallback (cited arrows) and
                         the explanation step; citation verifier
  evalkit.py             Shared evaluation helpers: resumable cache, bootstrap CI, McNemar, F1, scoring
  ocr/
    __init__.py          OCRClient: talks to the PaddleOCR worker over a pipe (images never touch disk)
    worker.py            Runs inside the separate OCR venv (external/venv-ocr), loads PaddleOCR once
  extractors/
    __init__.py          get_extractor(name): backend registry
    base.py              Extractor protocol, ExtractorUnavailable
    qwen_vl.py           Qwen2.5-VL loading (+ optional LoRA adapter), EXTRACTION_PROMPT, parse_extraction, generate()
    annotation.py        NON-AI fallback: returns gold graphs of the bundled synthetic samples
  datasets/
    ai2d.py              AI2D loader: gold graphs from annotations, questions, official splits
  static/
    index.html           Page layout: Ask and Evaluation tabs
    app.js               UI logic (upload, extraction progress, overlay, edge editing, suggestions,
                         live pipeline diagrams, evaluation tables/charts/browsers)
    style.css
scripts/
  get_ai2d.py            Download AI2D (requires --accept-license) into external/ (git-ignored)
  train_lora.py          LoRA fine-tuning of the extractor on AI2D train (resumable; --probe for a memory check)
  evaluate_ai2d.py       Main evaluation on the AI2D test split -> results/ai2d_eval.json
  make_synthetic.py      Pilot set: 27 diagrams (incl. counterfactual + knowledge-free), 304 questions, varied wording
  evaluate.py            Pilot evaluation E1 / E2 / E3 (resumable) -> results/pilot_eval.json
  evaluate_grounding.py  Grounded explanations on AI2D vs native VLM grounding + MC -> results/grounding_eval*.json
  train_arrow_detector.py  Arrow/arrowhead Mask R-CNN on AI2D train (150 held out) -> checkpoints/arrow_detector.pt
notebooks/
  colab_full_eval.ipynb  Full AI2D evaluations on a Colab GPU (Drive-backed caches)
  colab_train_arrows_inline.ipynb  Train the arrow detector on a Colab T4 (progress bars, threshold sweep)
  colab_eval_grounding.ipynb       Final grounding evaluation on fresh AI2D test diagrams (dev set excluded)
  make_figures.py        docs/figures/architecture.png
tests/                   pytest; no model or dataset needed (VLM replaced by stubs)
  test_qa_nl.py          prompts, JSON/option parsing, citation verification
  test_graph_qa.py       exact graph queries, abstention, entity linking
  test_parsing_and_api.py  extraction parsing, image formats/EXIF/transparency, API end-to-end (fallback backend)
  test_ai2d_and_eval.py  gold-graph construction, statistics, evaluation API
  test_query_and_pipelines.py  query parsing/execution, option matching, E3 decision rule, pipelines and
                         the streaming endpoint with stub models, scoring
  test_dsg_and_explain.py  shape finding, part grouping, arrow parsing, reference linking, AI2D gold
                         locations, the Explain pipeline (stub model)
external/                (git-ignored) AI2D download + cached raw model outputs
checkpoints/             (git-ignored) LoRA adapter, training log
results/                 committed evaluation outputs (ids + metrics only for AI2D); results/pilot_v1/ = first pilot
```

## Where things live

| Concern | Location |
|---|---|
| **Graph extraction (AI)** | `extractors/qwen_vl.py`: `EXTRACTION_PROMPT`, `QwenVLExtractor.extract(use_adapter=…)`, `parse_extraction` |
| **Fine-tuning** | `scripts/train_lora.py`: `target_json` (training targets), `encode`, `loss_on_targets`, LoRA config |
| **Scene graph (DSG)** | `dsg.DSGExtractor`: `number_texts`, `group_parts`, `ARROW_PROMPT` + `parse_arrows`, `ground`; `regions.enclosing_box` |
| **Arrow detector** | `arrows.py`: `build_model`, `ArrowDetector.detect`, `arrow_ends`, `link_arrows`; trained by `scripts/train_arrow_detector.py` |
| **Explanations + references** | `explain.py`: `EXPLAIN_PROMPT`, `link_mentions`, `build_references`; `pipelines.run_explain` |
| **E3 graph query** | `query.structured_answer` → `graph_qa.rule_parse` / `query.food_web_parse` / `PARSE_PROMPT` → `graph_qa.execute` → `query.match_options` |
| **E3 decision rule** | `pipelines.combine_e3` (graph answer / abstain / model fallback) |
| **Natural-language QA (AI)** | `qa_nl.py`: `GROUNDED_PROMPT` + `grounded_answer` (E3 fallback), `EXPLAIN_PROMPT`, `DIRECT_PROMPT` (E2), `OCR_PROMPT` (E1) |
| **Evidence checks (no AI)** | `qa_nl.verify`, which produces the status, valid/invalid citations and `answer_in_evidence` |
| **OCR (E1)** | `studylens/ocr/` (client + worker); setup below |
| **Real data** | `datasets/ai2d.py`: `build_graph`, `load` |
| **Evaluation** | `scripts/evaluate_ai2d.py` (`sample`, `is_relation_question`), `scripts/evaluate.py`, `studylens/evalkit.py` |
| **UI** | `static/app.js`: `loadImage`, `extractGraph`, `renderGraph`, `PIPES` (pipeline diagrams), `onPipeEvent`, `renderResult`, `renderAI2D`, `renderPilot` |
| **Image handling** | `uploads.py` (`load_upload`, `to_rgb`); `qwen_vl.prepare_image` (downscale before the model) |
| **Configuration** | `config.py`, `.env.example` |

## Model usage at a glance

| Call | Weights | Prompt | Output |
|---|---|---|---|
| `extract(image)` | base + LoRA (if configured) | `EXTRACTION_PROMPT` | JSON entities + arrows → `DiagramGraph` |
| `extract(image, use_adapter=False)` | base | same | zero-shot graph (evaluation condition E3-zs) |
| `grounded_answer(...)` | base | `GROUNDED_PROMPT` with numbered arrows | JSON answer/choice + cited arrow ids |
| `direct_answer(...)` | base | `DIRECT_PROMPT` | answer / option letter (E2) |
| `generate_text(PARSE_PROMPT)` | base, **text only** | entity names + question (no arrows, no image) | `{"op", "x", "y", "negate"}` |
| `explain(...)` | base, text only | `EXPLAIN_PROMPT` with the cited arrows | 1–2 sentence explanation |
| `ocr_answer(...)` | base, text only | `OCR_PROMPT` with OCR lines + positions | answer / option letter (E1) |

## E1 OCR setup (separate environment)

PaddleOCR pins numpy and PyYAML versions that conflict with the PyTorch stack, so it lives in its own
virtual environment and runs as a worker process:

```bash
python3 -m venv external/venv-ocr
external/venv-ocr/bin/pip install -r requirements-ocr.txt
```

The first OCR call downloads the PP-OCRv5 mobile detection and English recognition models (Apache-2.0)
to `~/.paddlex/official_models/`. Set `OCR_PYTHON` if the venv lives elsewhere. Without it, E1 shows
"OCR unavailable" and E2/E3 still work.
