# StudyLens: graph-grounded Q&A over diagrams

CECS 458/558 term project · **Phase 2 prototype** · Repository: `https://github.com/krish-patel1003/CECS558-term-project-studylens`

StudyLens helps students understand complex diagrams: flowcharts, system architectures, food webs,
life cycles, anatomy. Upload any diagram image and StudyLens builds a **Diagram Scene Graph**:
- the labelled parts (from OCR), each with the region to highlight;
- the arrows between them (from a vision-language model).

The student can review and correct it. Then ask anything in your own words: StudyLens explains the
diagram, and **every part or flow it mentions is a clickable reference** that spotlights that part on the
image or animates the flow. Structural facts ("what feeds into X", "is there a path…") are computed
**deterministically by NetworkX** from the graph, not guessed.

A **"Compare with plain AI"** panel runs the evaluated conditions on the same question, each drawn live
as an architecture diagram:

| | Pipeline | What answers | Evidence |
|---|---|---|---|
| **E1** | OCR baseline | PaddleOCR reads the text; Qwen answers from the text only | none |
| **E2** | Image baseline | Qwen answers from the image, like a general AI assistant | none |
| **E3** | Graph query | Question → graph query → NetworkX; model fallback with checked arrow citations; abstains when no arrow supports an answer | the arrows used |

> **Where AI is used.**
> - OCR (PaddleOCR).
> - Arrows and the regions of parts outside shapes (Qwen2.5-VL-3B).
> - Explanations.
> - Question parsing when the keyword rules do not match (Qwen, text only).
> - Fallback answers, and the E1/E2 answers.
> - The legacy extractor with our LoRA adapter fine-tuned on AI2D.
>
> **Not AI:**
> - grouping text into parts, and finding the shape around a label;
> - the NetworkX queries;
> - **linking references to parts**;
> - option matching, citation checking and the abstention rule. Details: [`docs/model_and_prompts.md`](docs/model_and_prompts.md).

## Features (implemented and tested)

| Feature | Status |
|---|---|
| Upload any common image (PNG, JPEG, WebP, BMP, GIF, TIFF, HEIC; ≤ 10 MB / 40 MP), EXIF rotation, transparency, in memory only | ✅ |
| **Extract graph** button with a progress bar driven by real model events (stages, live entity/arrow counts) | ✅ |
| Graph extraction with Qwen2.5-VL-3B + **LoRA adapter trained on AI2D** (zero-shot also available) | ✅ |
| Truncated model output salvaged (complete entities/arrows kept, warning shown) | ✅ |
| Overlay of boxes and arrows; edit the graph (add/remove arrows) before asking | ✅ |
| **Diagram Scene Graph**: OCR parts (exact text and boxes), enclosing-shape regions, arrows by name, grounded regions for parts outside shapes | ✅ |
| **Explanations with clickable references**: spotlight a part on the image; animate a flow along graph arrows; references linked deterministically | ✅ |
| Free-form questions (any wording) and optional multiple-choice options; suggestions generated from the graph | ✅ |
| **E3 graph query**: rules / Qwen parser → NetworkX → answer + arrows; deterministic option matching incl. "NOT" questions; abstention | ✅ |
| E3 explanation (Qwen, from the cited arrows only) and checked-citation fallback | ✅ |
| **E1 PaddleOCR baseline** and **E2 image baseline** run on the same question | ✅ |
| **Live architecture diagrams of E1 / E2 / E3** with per-stage state, timings and intermediate outputs | ✅ |
| Evaluation tab: AI2D test-split results, synthetic pilot, training curve, per-question browsers | ✅ |
| AI2D (real data) loader, gold graphs, LoRA training script, evaluation with CIs and McNemar tests | ✅ |
| 304-question synthetic pilot (knowledge-free and counterfactual diagrams) | ✅ |
| Unit / API tests (stubs replace the models) | ✅ `pytest` |
| Grounded-explanation evaluation on AI2D (vs the same VLM's native grounding); Colab notebook for full runs | ✅ |
| Guided walkthrough and quiz generator | ❌ planned for the final submission |
| PDF input, learning history | ❌ not implemented (stretch goals) |

## Prerequisites

* Python 3.10+ (developed on 3.12), macOS / Linux.
* **UI, fallback backend and tests only:** about 100 MB of packages, no GPU.
* **With the AI model:** torch + transformers + peft, about 7.5 GB disk for the weights (downloaded
  once from Hugging Face) and about 10 GB free RAM/VRAM. Tested only on an Apple M4 with 16 GB
  (MPS, bf16).
* **E1 OCR baseline (optional):** PaddleOCR in a **separate** virtual environment (its numpy/PyYAML
  pins conflict with PyTorch); about 1 GB.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # web app, graph query, fallback backend, tests
pip install -r requirements-model.txt      # Qwen2.5-VL backend, LoRA, evaluation
cp .env.example .env                       # optional; no secrets are needed

# optional: E1 OCR baseline in its own venv
python3 -m venv external/venv-ocr
external/venv-ocr/bin/pip install -r requirements-ocr.txt
```

The LoRA adapter is not committed (it is a derivative of AI2D, whose licence forbids
redistribution). Train it with `scripts/get_ai2d.py --accept-license` then `scripts/train_lora.py`
(see [`docs/model_and_prompts.md`](docs/model_and_prompts.md#fine-tuning-what-we-trained)), or run
zero-shot by leaving `QWEN_ADAPTER_PATH` empty.

## Configuration

All settings are environment variables (or a local, git-ignored `.env`). See [`.env.example`](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `EXTRACTOR_BACKEND` | `annotation` | `qwen` (AI) or `annotation` (non-AI fallback, bundled samples only) |
| `QWEN_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF id or local path |
| `QWEN_ADAPTER_PATH` | empty | LoRA adapter for extraction (e.g. `checkpoints/ai2d_lora_r16`) |
| `STUDYLENS_DEVICE` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `QWEN_MAX_NEW_TOKENS` / `QWEN_MAX_IMAGE_SIDE` | `3072` / `1024` | extraction output limit / downscale limit (the evaluations pin `1024` / `448`) |
| `OCR_PYTHON` | `external/venv-ocr/bin/python` | interpreter of the PaddleOCR venv (E1) |
| `EXTRACTION_MODE` | `dsg` | `dsg` = OCR-anchored scene graph (needs the OCR venv); `legacy` = one-call JSON extraction (uses the LoRA adapter) |
| `MAX_UPLOAD_MB` / `MAX_IMAGE_PIXELS` | `10` / `40000000` | upload limits |
| `ENABLE_DIRECT_VLM` | `1` | allow image-only answers (E2) |

No API keys are needed. If you hit Hugging Face rate limits, put `HF_TOKEN` in `.env` (never commit it).

## Run the prototype

```bash
# AI backend with the fine-tuned extractor
EXTRACTOR_BACKEND=qwen QWEN_ADAPTER_PATH=checkpoints/ai2d_lora_r16 python -m uvicorn studylens.app:app --port 8000

# Non-AI fallback: bundled samples only; E3 answers rule-recognised questions, E1/E2 need the model
EXTRACTOR_BACKEND=annotation python -m uvicorn studylens.app:app --port 8000
```

Open <http://localhost:8000>. The badge in the header shows the active backend. Only one model
process fits in 16 GB, so stop the app before training or evaluating.

### Sample usage

1. **Diagram:** upload any diagram (or pick **"Pond food web"** in the sample menu) and click **Extract graph**.
   The progress bar shows OCR → parts → arrows → regions. Check the parts and arrows on the right, and fix
   any wrong arrow.
2. **Question:** e.g. **"How does energy move from the algae to the heron?"**, then **Ask**.
3. **Answer:**
   - A green box shows what NetworkX computed from the graph, if the question is structural.
   - The explanation follows with numbered references. Click a number: a part is spotlighted on the image,
     and a flow is animated arrow by arrow.
   - If the graph has a wrong arrow, the computed fact says so (e.g. "no path"). Fix the arrow and ask again.
4. **Compare with plain AI:** runs E1 / E2 / E3 on the same question and shows their live pipelines.

More test images and expected answers: [`data/demo_inputs/README.md`](data/demo_inputs/README.md).

## Reproduce the evaluation

```bash
python -m pytest                                                     # tests, no model needed
python scripts/evaluate.py --no-model                                # pilot, gold graph + rules only, seconds
python scripts/evaluate.py --adapter checkpoints/ai2d_lora_r16       # pilot E1/E2/E3 (~2.5 h on an M4, resumable)
python scripts/get_ai2d.py --accept-license                          # AI2D into external/ (git-ignored)
python scripts/evaluate_grounding.py --n-parts 30 --n-flow 20       # grounded explanations on AI2D (~5 h on an M4)
python scripts/evaluate_ai2d.py --adapter checkpoints/ai2d_lora_r16  # AI2D test split (several hours, resumable)
# full runs on a GPU: notebooks/colab_full_eval.ipynb
```

Results go to `results/pilot_eval.json` and `results/ai2d_eval.json` and appear in the Evaluation
tab. Raw model outputs are cached in `external/` (git-ignored; AI2D text may not be
redistributed). Greedy decoding is used, but MPS/CUDA kernels are not guaranteed bit-identical.

Results and their interpretation: [`docs/evaluation.md`](docs/evaluation.md).

## Data handling and privacy

* Uploads are validated by decoding the image (format, size and pixel limits) and are **processed
  in memory**. Starlette's spool-to-disk threshold is raised to the upload limit and oversized
  requests are refused before parsing. Images sent to the OCR worker go over a pipe, not via files.
  Nothing is retained after the request; the server is stateless.
* The extracted graph and OCR lines live only in the browser tab and disappear on reload.
* No accounts, analytics, database or logs of uploaded content.
* No credentials are stored in the repository; `.env` is git-ignored.

## Licences

* **Qwen2.5-VL-3B-Instruct: Qwen Research License, non-commercial only.** Weights are not redistributed.
* **AI2D: AI2 non-commercial licence, no redistribution.** The data, raw outputs and the adapter
  trained on it are not committed; results contain ids and metrics only.
* PaddleOCR / PP-OCRv5 models: Apache-2.0. Other dependencies: Apache-2.0 / BSD / MIT-style.
* The synthetic data in `data/synthetic/` was generated for this project.

This is an academic prototype; we make no commercial-readiness claims. Details:
[`docs/model_and_prompts.md`](docs/model_and_prompts.md#licences) and [`docs/data.md`](docs/data.md).

## Limitations (summary)

* **E3 is only as good as the extracted graph.** The query is deterministic, but a missing or
  reversed arrow gives a wrong answer or an abstention. Extraction is the bottleneck on dense,
  out-of-domain diagrams (e.g. large software architectures): the adapter was trained on AI2D
  science diagrams with ≤ 30 labels.
* Diagrams made of labelled parts without arrows (cells, leaves) get no graph evidence; E3 then
  answers from the image and says so.
* The keyword rules were debugged on the synthetic pilot's questions, so pilot E3 numbers are
  optimistic; AI2D is the unbiased test. The 3B parser can misread direction.
* Dense diagrams take minutes to extract on a laptop (each box costs ~28 output tokens).
* English only; handwriting and photos of whiteboards were not systematically evaluated.

## Documentation

* [`docs/architecture.md`](docs/architecture.md): system diagram, data flow, deviations from Phase 1
* [`docs/code_guide.md`](docs/code_guide.md): main files, where each concern lives, OCR setup
* [`docs/model_and_prompts.md`](docs/model_and_prompts.md): models, training, prompts (verbatim), I/O, limitations, licences, team contribution
* [`docs/data.md`](docs/data.md): AI2D and synthetic data, licences
* [`docs/evaluation.md`](docs/evaluation.md): evaluation method and results
* [`reports/phase2/phase2_report.pdf`](reports/phase2/phase2_report.pdf): Phase 2 progress report (≤ 2 pages)

## Team

`<TEAM_MEMBERS_PLACEHOLDER>`: names and contributions to be filled in by the team.
