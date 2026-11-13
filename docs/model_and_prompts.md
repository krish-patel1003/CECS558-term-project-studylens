# Model, training, prompts and licences

## Where AI is used (and where it is not)

| Step | AI? | Weights | Code |
|---|---|---|---|
| Scene graph: text items and their boxes | **Yes** (OCR models) | PaddleOCR PP-OCRv5 mobile | `studylens/ocr/`, `dsg.number_texts` |
| Scene graph: grouping text into parts, enclosing shapes | **No** | – | `dsg.group_parts`, `regions.enclosing_box` |
| Scene graph: arrows between parts (by name) | **Yes** | Qwen2.5-VL-3B base | `dsg.ARROW_PROMPT` |
| Scene graph: regions of parts outside shapes | **Yes** | Qwen2.5-VL-3B base (grounding) | `dsg.GROUND_PROMPT` |
| Explanation of the diagram for the question | **Yes** | Qwen2.5-VL-3B base | `explain.EXPLAIN_PROMPT` |
| Linking the explanation's mentions to parts / flows | **No** | – | `explain.build_references` |
| Legacy one-call extraction (`EXTRACTION_MODE=legacy`, evaluation conditions E3-zs/ft) | **Yes** | Qwen2.5-VL-3B **+ our LoRA adapter** (or zero-shot) | `extractors/qwen_vl.py` |
| Question → graph query, when the keyword rules do not match (E3) | **Yes**, as a parser only | Qwen base, **text only** (sees entity names, not arrows) | `query.PARSE_PROMPT` |
| Question → graph query by keyword / food-web rules (E3) | No | – | `graph_qa.rule_parse`, `query.food_web_parse` |
| Answer the query on the graph, match options (E3) | **No** | – | `graph_qa.execute` (NetworkX), `query.match_options` |
| Explain a graph-query answer from its cited arrows (E3) | **Yes** | Qwen base, text only | `qa_nl.explain` |
| Fallback answer citing arrow ids, when the query cannot apply (E3) | **Yes** | Qwen base (adapter disabled), image + arrows | `qa_nl.grounded_answer` |
| Image-only answer (E2) | **Yes** | same base weights | `qa_nl.direct_answer` |
| OCR of the image (E1) | **Yes** (OCR models) | PaddleOCR PP-OCRv5 mobile det + English rec | `studylens/ocr/` |
| Answer from OCR text only (E1) | **Yes** | Qwen base, text only | `qa_nl.ocr_answer` |
| Parse model JSON, validate graph, map options | No | – | `parse_extraction`, `qa_nl.parse_json_obj`, `match_choice`, `query.parse_llm_query` |
| Check citations, assign grounded / unsupported / image-only / abstained | **No** | – | `qa_nl.verify`, `pipelines.combine_e3` |
| Annotation fallback backend | **No** | – | `extractors/annotation.py` |

## Fine-tuning (what we trained)

* **Task:** diagram → graph extraction. We fine-tuned only this step, because extraction
  errors (wrong or reversed arrows) were the main failure in the synthetic pilot, and because
  AI2D supplies gold graphs to supervise it.
* **Data:** AI2D train split. Diagrams with more than 30 labels were excluded to keep targets
  short. The pool was 1,188 diagrams with arrows and 600 without; the model saw 800 sampled
  examples (200 steps × 4). 20 diagrams were held out for validation loss.
* **Input:** the image (≤ 448 px) and the same `EXTRACTION_PROMPT` the app uses.
* **Target:** compact JSON from the gold graph. Boxes are in the model's resized-image pixel
  grid. Arrows run from tail label to head label.
* **Method:** LoRA with rank 16, α 32 and dropout 0.05. Targets are the q/k/v/o projections of
  the language model; the vision encoder is frozen. 7.4 M trainable parameters (0.20 %).
* **Optimisation:**
  * AdamW, lr 2e-4, 5 % warm-up, cosine decay;
  * 200 optimiser steps × 4 examples (gradient accumulation), about 0.45 epoch of the pool;
  * bf16 on Apple M4 (MPS), gradient checkpointing;
  * loss only on target tokens, with logits computed only there.
* **Not trained:** the QA behaviour. The answering prompt runs on base weights, so the
  comparison between graph sources (zero-shot vs fine-tuned vs gold) isolates the graph.
* **Outputs:** the adapter and `train_log.jsonl` go to `checkpoints/ai2d_lora_r16/`
  (git-ignored). The loss curve is shown in the Evaluation tab.

## Prompts (verbatim)

### Scene graph: arrows by name (`dsg.ARROW_PROMPT`)

```
You are reading a diagram. These are the names of its labelled parts: {names}.

List every arrow that is drawn in the diagram. An arrow goes FROM the part at its tail TO the part its arrowhead touches; look carefully at which end has the arrowhead. Only arrows that are actually drawn, not connections you would expect. Use the part names exactly as given.

Reply with ONLY JSON: {{"arrows": [{{"from": "<part name>", "to": "<part name>"}}]}}
```

### Scene graph: regions of parts outside shapes (`dsg.GROUND_PROMPT`, one call per diagram)

```
For each label below, locate the part of the diagram it names and give its bounding box. If a label is connected by a line to a drawing, give the box of the PART the line points to, not of the label text. Otherwise give the box of the object or region the label names.
Labels: {labels}
Reply with ONLY a JSON list: [{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]
```

### Explanation (`explain.EXPLAIN_PROMPT`, the StudyLens answer)

```
You are a tutor helping a student understand this diagram.
Parts labelled in the diagram: {names}
Arrows extracted from the diagram (A -> B means an arrow from A to B; the extraction may contain mistakes, the image is the final authority):
{arrows}
{facts}
Student's question: {question}

Answer in 3 to 6 sentences (use numbered steps if you describe a process or a flow). Mention parts by their names exactly as labelled in the diagram. Explain what the diagram shows; if the diagram does not show what is asked, say so.
```

`{facts}` holds the NetworkX result and its arrows when the question is structural. References are not
requested from the model: they are linked afterwards by exact name matching (see `docs/architecture.md`,
"Design decisions forced by measurements").

### Extraction (`EXTRACTION_PROMPT`; same for zero-shot and fine-tuned)

```
You are reading a diagram. Extract its structure as a directed graph.

1. List every labelled entity (a box, shape or object with a text label). Copy each label exactly as written. Give its bounding box as [x1, y1, x2, y2] in pixel coordinates of this image.
2. List every arrow that is actually drawn. An arrow goes FROM the entity at its tail TO the entity its arrowhead touches. Look carefully at which end has the arrowhead. Do not add arrows that are not drawn, even if they would make sense.

Return ONLY compact JSON (no indentation, no code fences) in exactly this format, with no explanation:
{"entities": [{"label": "...", "bbox_2d": [x1, y1, x2, y2]}], "arrows": [{"from": "<entity label>", "to": "<entity label>"}]}
```

### Graph-grounded answer (`GROUNDED_PROMPT`, condition E3 and the app)

```
Answer a question about this diagram. You are also given a graph that was extracted automatically from the diagram (it may contain mistakes). Arrow "A -> B" means the diagram shows an arrow from A to B.

Entities: {entities}
Arrows:
e1: {A} -> {B}
...

Question: {question}
[Options:
a) ...
b) ...]
Use the diagram and the arrows. Reply with ONLY a JSON object:
{"answer" | "choice": ..., "evidence": [ids of the arrows your answer relies on, e.g. "e1"], "graph_supported": true or false}
Set "graph_supported" to false and "evidence" to [] if none of the listed arrows supports your answer. If the question cannot be answered from the diagram, set "answer" to "cannot answer".
```

### Question → graph query (`query.PARSE_PROMPT`, E3; text only, used only when the rules do not match)

```
Convert a question about a diagram into ONE graph operation. Do not answer the question.

The diagram's entities: {entities}

Operations (an arrow A -> B means the diagram draws an arrow from A to B):
- successors(x): what x points to directly
- predecessors(x): what points to x directly
- descendants(x): everything reachable from x by following arrows
- ancestors(x): everything that leads to x
- edge(x, y): is there an arrow x -> y
- path(x, y): is there a chain of arrows from x to y
- count_in(x) / count_out(x): number of arrows into / out of x
- sources: entities with no incoming arrows (start points; in a food web: producers)
- sinks: entities with no outgoing arrows (end points; in a food web: top predators)
- none: the question is not about arrows or connections (e.g. colour, shape, a definition, what a label means)

Arrow conventions: in food chains and food webs arrows go from the food to the eater (direction of energy flow), so "what eats x" = successors(x), "what does x eat" = predecessors(x), "which organism would be affected if x disappeared" = successors(x). In cycles and processes arrows go from a stage to the next one, so "what comes after x" = successors(x) and "what comes before x" = predecessors(x).
Use entity names exactly as listed. Set "negate" to true if the question asks which option is NOT / does NOT.

Examples (entities from other diagrams):
"What does the frog eat?" -> {"op": "predecessors", "x": "frog", "y": null, "negate": false}
"Which organism eats the frog?" -> {"op": "successors", "x": "frog", "y": null, "negate": false}
"Which of these is NOT eaten by the hawk?" -> {"op": "predecessors", "x": "hawk", "y": null, "negate": true}
"Does energy flow from grass to the snake?" -> {"op": "path", "x": "grass", "y": "snake", "negate": false}
"What stage comes after the pupa?" -> {"op": "successors", "x": "pupa", "y": null, "negate": false}
"Which organisms get energy from the sun, directly or indirectly?" -> {"op": "descendants", "x": "sun", "y": null, "negate": false}
"What is the function of the leaf?" -> {"op": "none", "x": "leaf", "y": null, "negate": false}

Question: {question}
{options}Reply with ONLY JSON: {"op": "...", "x": "<entity or null>", "y": "<entity or null>", "negate": false}
```

The reply is validated deterministically: unknown operations become `none`; entity names are matched
to graph nodes; a second entity is ignored for one-entity operations.

### Explanation of a graph-query answer (`qa_nl.EXPLAIN_PROMPT`, E3)

```
A student asked about a diagram: {question}
The answer, computed from the arrows extracted from the diagram, is: {answer}
The arrows used as evidence:
{arrows}

Explain this answer to the student in one or two sentences, using only these arrows (an arrow A -> B means the diagram draws an arrow from A to B). Do not add facts the arrows do not show.
```

### OCR-text answer (`qa_nl.OCR_PROMPT`, condition E1; no image)

```
You cannot see the diagram. An OCR system read the following pieces of text from it. Each line gives the text and the position of its centre in pixels from the top-left corner of a {w}x{h} image.

{lines}

Question: {question}
{options}
{instruction}
```

Each OCR line is written as `"text" at (x, y)` (centre of its box), sorted top-to-bottom, at most 80 lines.

### Image-only answer (`DIRECT_PROMPT`, condition E2)

```
Answer a question about this diagram using what the diagram shows.

Question: {question}
[Options: a) ... b) ...]
Reply with ONLY the letter of the correct option.     (multiple choice)
Answer in one short sentence. If the diagram does not show enough to answer, reply exactly: Not shown in the diagram.   (free form)
```

## Input / output behaviour

* **Images:** any of PNG, JPEG, WebP, BMP, GIF, TIFF and HEIC/HEIF up to 10 MB / 40 MP. Images
  are normalised to RGB (EXIF rotation applied, transparency flattened onto white) and
  downscaled to ≤ `QWEN_MAX_IMAGE_SIDE` on the longer side before the model: 1,024 px in the app
  (so small labels stay readable), 448 px in training and in `scripts/evaluate_ai2d.py`. Boxes are mapped back to the
  original pixels.
* **Extraction output:** a `DiagramGraph`. Warnings list arrows dropped because an endpoint
  was not a listed entity, and say when the output was cut off at the token limit. In that case
  the complete entities and arrows are kept (a partial graph) instead of discarding everything. The model provides no confidence scores, so none are shown.
* **QA output:** `answer`, `choice_index` (for multiple choice), `status`, `evidence` (valid
  cited arrows), `invalid_citations`, `answer_in_evidence` and `raw` (model text).
* **Decoding:** greedy (`do_sample=False`), with up to 3,072 new tokens for extraction (1,024
  in the AI2D evaluation), 96 for fallback answers, 64 for E1/E2 answers, 60 for the query parser
  and 90 for explanations. Qwen tokenises every digit separately, so one entity
  with a box costs about 28 tokens; dense diagrams (40+ boxes) need well over 1,024 tokens and take
  minutes on a laptop.

## Known limitations

* **The LoRA adapter is not used by the scene-graph extractor.** It was trained on the legacy one-call
  output format; the scene graph uses base weights. Retraining on scene-graph targets (parts, label→part
  links, arrows) is final-submission work. The adapter is still evaluated through the legacy conditions.
* **Explanation text is not verified.** References always point at real parts (computed links), but the
  3B model's sentences can contain statements the diagram does not show (seen in testing: invented
  "encryption" in an architecture explanation). The UI says so. Claim checking against the graph is
  planned.
* **Speed (Apple M4, 1,024 px):** scene graph 45 s (5 boxes) to 450 s (40 boxes); explanation 50–140 s.

* **The graph query is only as good as the graph.** A NetworkX answer is deterministic and cites
  arrows, but a wrong or missing arrow gives a confidently wrong answer or an abstention. On dense,
  out-of-domain diagrams (large software architectures) extraction is the bottleneck.
* **Rules were debugged on the synthetic pilot.** The keyword rules were fixed while checking the
  pilot's question templates on gold graphs, so pilot E3 numbers are optimistic. AI2D's
  human-written questions were not used to develop the rules and are the unbiased test.
* **The 3B parser can choose the wrong direction** (e.g. "what does X eat" vs "what eats X").
  The common food-web phrasings are therefore handled by deterministic rules first; the parser's
  output is shown in the UI so a wrong parse is visible.

* **Graph quality bounds grounding.** If an arrow is extracted in the wrong direction, a
  "grounded" answer can be confidently wrong. Grounded means *consistent with the extracted
  graph*, not *correct*. Correcting the graph in the UI fixes this case.
* **The model decides `graph_supported` itself.** The verifier only checks that cited arrows
  exist and involve the answer; it cannot prove that the reasoning used them.
* **Resolution mismatch:** the adapter was trained at 448 px (for memory) but the app extracts at
  1,024 px so small labels are readable. Accuracy at 1,024 px was not measured on AI2D; the
  evaluation uses 448 px.
* **Out-of-domain diagrams:** the adapter was trained only on AI2D science diagrams with ≤ 30
  labels. Software-architecture diagrams (elbow connectors, two-way arrows, arrows ending on
  lines, grouping regions) are outside that distribution.
* **Labels with leader lines, captions and titles are not extracted.** Only entities and the
  arrows between them are. Questions about labelled parts (e.g. a cell diagram) are answered
  from the image alone and shown as "Not supported by the graph".
* **Prior-knowledge override:** on the synthetic counterfactual food chain, the zero-shot
  extractor output the textbook arrow directions rather than the drawn ones.
* **Latency on a 16 GB M4:** tens of seconds per extraction and several seconds per answer.
  The measured values are in the Evaluation tab.
* **Scope of testing:** English only. Photos of whiteboards and handwriting were not
  systematically evaluated.

## Licences

| Item | Licence / terms (checked) | Consequence |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | **Qwen RESEARCH LICENSE AGREEMENT** (`LICENSE` file shipped with the weights): non-commercial research/evaluation only. Redistribution must include the licence and the notice "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved."; models trained on its outputs that are distributed must say "Built with Qwen" | Weights are not redistributed (downloaded from Hugging Face). Our adapter is a modification of Qwen Materials, so if it were shared, these terms would apply |
| AI2D | **AI2 non-commercial licence** (`license.txt` in the archive): no for-profit use, no redistribution, no public release of modified data | Not committed. Results contain ids and metrics only. The adapter is not shared. See `docs/data.md` |
| PEFT, transformers, torch, FastAPI, NetworkX, Pillow, pillow-heif | Apache-2.0 / BSD / MIT-style | Installed, not vendored |
| PaddleOCR 3.7, PaddlePaddle 3.3 and the PP-OCRv5 mobile models | Apache-2.0 | Installed into a separate venv; models downloaded on first use, not vendored |
| Synthetic diagrams | Created for this project | Committed |

**This is an academic prototype.** Both the model licence and the dataset licence exclude
commercial use, and we make no commercial-readiness claims.

## Team's technical contribution vs off-the-shelf components

* **Off-the-shelf:** the Qwen2.5-VL weights and inference code, PEFT's LoRA implementation, the
  AI2D data, NetworkX, FastAPI and Pillow.
* **Ours:**
  * the gold-graph conversion of AI2D annotations (blob→label mapping, direction handling);
  * the fine-tuning pipeline that fits a 3B VLM on a 16 GB laptop (target-only logits, masking,
    resumable training);
  * the extraction and grounded-QA prompts;
  * the JSON repair and coordinate mapping;
  * the deterministic citation verifier and answer statuses;
  * the graph-query engine for E3 (keyword and food-web rules, an LLM used strictly as a parser with
    deterministic validation, NetworkX execution, deterministic multiple-choice matching including
    negation, the abstain/fallback rule);
  * the three-pipeline design (E1 OCR, E2 image, E3 graph) with live streamed progress, and the
    isolated OCR worker;
  * robust image intake;
  * the web UI with graph editing, evidence highlighting and the evaluation dashboard;
  * the evaluation design (E1 / E2 / E3-vlm / E3-zs / E3-ft / E3*, relationship-question subset,
    bootstrap CIs, McNemar) and the 304-question synthetic pilot with counterfactual diagrams;
  * licence-compliant result handling;
  * the tests.
