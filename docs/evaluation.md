# Evaluation

Two evaluations exist:
- the **synthetic pilot**, which is complete;
- the **AI2D grounded-explanation evaluation** (100 test diagrams), which is complete.

The full AI2D runs are prepared for Colab (`notebooks/colab_full_eval.ipynb`). The first, smaller pilot (79
templated questions) is kept for reference in `results/pilot_v1/`.

## 1. Synthetic pilot (complete): E1 vs E2 vs E3

**Data.** `scripts/make_synthetic.py --seed 458` produces 27 diagrams and 304 questions with exact gold
answers, in varied natural wording.
- 12 knowledge-free graphs with nonsense labels.
- 4 counterfactual diagrams whose arrows contradict textbook knowledge (reversed food chain, pond web,
  frog life cycle, water cycle).
- 12 question kinds, including 2 kinds that should be abstained. Details: `docs/data.md`.

**Conditions** (identical questions; `scripts/evaluate.py`; Qwen2.5-VL-3B, greedy decoding, images ≤ 448 px,
Apple M4, single run):

| Condition | What answers |
|---|---|
| E1 | PaddleOCR text + positions → Qwen, text only (no image, no graph) |
| E2 | image + question → Qwen (plain VLM) |
| E3-vlm | image + fine-tuned extracted graph → Qwen answers and cites arrows (no graph query) |
| E3-zs / E3-ft | question → graph query (keyword rules, else Qwen as parser only) → NetworkX on the zero-shot / fine-tuned graph. The model answers only when the query cannot apply; E3 abstains when no arrow supports an answer |
| E3* | the same on the gold graph (upper bound) |

**Scoring.**
- Graph-query answers are scored by their answer nodes, or yes/no, or count.
- Free-text answers (E1, E2, fallbacks) are scored by the entity labels they mention, by yes/no at the start
  of the answer, or by abstention phrases.
- Raw answers are in `results/pilot_eval.json` and can be browsed in the Evaluation tab.

### Accuracy (95% bootstrap CI)

| Condition | All 304 | Structural (254) | Knowledge-free (144) | Counterfactual (38) | Should abstain (50) |
|---|---|---|---|---|---|
| E1 | 96/304 = 31.6% [26.6, 36.8] | 18.1% | 30.6% | 34.2% | 100.0% |
| E2 | 151/304 = 49.7% [44.1, 55.3] | 45.3% | 39.6% | 34.2% | 72.0% |
| E3-vlm | 113/304 = 37.2% [31.9, 42.8] | 42.9% | 25.7% | 28.9% | 8.0% |
| E3-zs | 172/304 = 56.6% [51.0, 61.8] | 60.6% | 46.5% | 15.8% | 36.0% |
| E3-ft | 164/304 = 53.9% [48.4, 59.5] | 57.9% | 40.3% | 21.1% | 34.0% |
| E3* | 279/304 = 91.8% [88.8, 94.7] | 100.0% | 93.1% | 89.5% | 50.0% |

### Paired tests on all questions (exact McNemar)

| Comparison | Only first correct | Only second correct | p |
|---|---|---|---|
| E2 vs E1 | 85 | 30 | 0.0 |
| E2 vs E3-vlm | 65 | 27 | 0.0001 |
| E2 vs E3-zs | 44 | 65 | 0.0549 |
| E2 vs E3-ft | 49 | 62 | 0.2546 |
| E3-zs vs E3-ft | 40 | 32 | 0.4096 |
| E3-ft vs E3* | 2 | 117 | 0.0 |

On **structural questions only**:
- E3-zs is right on 58 that E2 gets wrong, and E2 on 19 that E3-zs gets wrong (p < 0.001);
- E3-ft vs E2 is 56 vs 24 (p = 0.0005);
- E3-vlm vs E2 is 27 vs 33 (p = 0.52);
- E1 vs E2 is 16 vs 85 (p < 0.001).

On **knowledge-free structural questions**, E3-zs scores 48.3% and E2 32.5% (p = 0.002).

### By question kind

| Kind | n | E1 | E2 | E3-vlm | E3-zs | E3-ft | E3* |
|---|---|---|---|---|---|---|---|
| ancestors | 23 | 0% | 30% | 26% | 48% | 48% | 100% |
| count_in | 27 | 0% | 52% | 33% | 74% | 59% | 100% |
| descendants | 23 | 0% | 26% | 26% | 44% | 48% | 100% |
| edge_reversed | 27 | 93% | 48% | 56% | 89% | 82% | 100% |
| edge_true | 27 | 11% | 70% | 74% | 63% | 70% | 100% |
| path | 27 | 59% | 82% | 48% | 74% | 74% | 100% |
| predecessors | 27 | 0% | 56% | 48% | 56% | 44% | 100% |
| sinks | 23 | 0% | 13% | 17% | 48% | 44% | 100% |
| sources | 23 | 0% | 22% | 26% | 65% | 56% | 100% |
| successors | 27 | 7% | 41% | 63% | 41% | 48% | 100% |
| unsupported_no_edge | 23 | 100% | 48% | 9% | 78% | 65% | 100% |
| unsupported_non_structural | 27 | 100% | 93% | 7% | 0% | 7% | 7% |

### Extraction (vs gold graphs, 27 diagrams)

| Extractor | Entity F1 | Arrow P / R / F1 | Reversed arrows | Mean time |
|---|---|---|---|---|
| Zero-shot | 0.98 | 0.641 / 0.59 / 0.614 | 19 | 48.5 s |
| Fine-tuned (LoRA) | 0.917 | 0.471 / 0.691 / 0.56 | 32 | 48.1 s |

**Graph-query coverage** (E3-ft):
- the query answered 260 of 304 questions;
- on those, E3-ft was right 56.5% of the time, and E2 48.5%.

### What the pilot shows

1. **Structure helps when code answers from it.** The graph query beats the plain VLM on structural
   questions, including knowledge-free ones. Handing the graph to the model (E3-vlm) does not help.
2. **Extraction is the bottleneck.** With the gold graph, the same query is right on 100% of structural
   questions.
3. **Prior knowledge corrupts extraction on counterfactual diagrams.** The extractor "corrects" reversed
   arrows to the textbook direction (E3 19–25% vs E3* 100% on structural questions). Correcting the graph
   in the UI is the remedy.
4. **The LoRA adapter did not improve extraction on these diagrams.** Arrow precision fell from 0.64 to
   0.47, and the QA difference vs zero-shot is not significant (p = 0.41).
5. **E3 abstains poorly on non-arrow questions.** The 3B parser mapped "How large is X?" to an arrow count;
   E2 says "Not shown". E1's perfect score on should-abstain questions is an artefact: with text only, it
   says "Not shown" to almost everything.

**Caveats.**
- Single run, clean synthetic images.
- The keyword rules were debugged on these question templates, so E3 numbers are optimistic.
- Scoring free text with string rules is imperfect; every raw answer is saved, so each decision can be audited.

## 2. AI2D grounded-explanation evaluation (complete)

`scripts/evaluate_grounding.py --n-parts 60 --n-flow 40`, seed 458, results in `results/grounding_eval.json`
(ids and metrics only; explanations and question text stay in `external/` because of the AI2D licence):
- 100 AI2D **test** diagrams: 60 "parts of" diagrams (anatomy-style labels with leader lines) and 40 with arrows
  (food webs, life cycles, rock/water cycles);
- 852 annotated part locations;
- 125 multiple-choice questions (up to 3 per diagram, picked per diagram so the sample is the same at any size).
- Single run on an Apple M4: 5.7 h for 83 new diagrams (17 were reused from the cache); about 90 s per scene graph.

**Systems compared:**

| System | What it does |
|---|---|
| StudyLens | scene graph → explanation → references linked deterministically |
| B2 | the same Qwen2.5-VL-3B asked to explain the diagram and give a box for each part it mentions (its native grounding, no scene graph, no checks) |

**Metrics:**
- **Pointing accuracy:** among references whose name matches an annotated label, the share whose highlight
  contains the gold leader-line tip (padded by 3% of the image diagonal) or whose centre lies inside the
  labelled object.
- **Coverage:** annotated parts that at least one correct highlight landed on.
- **Flow-reference precision** against AI2D's gold arrows.
- **Invented references:** references to names that are not in the diagram's text.
- **Scene-graph region accuracy**, by how each region was found.
- **Multiple-choice accuracy:** E2 vs E3 (exact McNemar).

### Pointing (references → annotated parts)

| | References | Matched to a gold label | Pointing accuracy | Coverage (of 852) |
|---|---|---|---|---|
| StudyLens | 533 | 349 | **49.9%** (174) | 20.4% |
| B2 (native grounding) | 662 | 493 | **65.3%** (322) | 36.7% |

| By category | StudyLens accuracy | B2 accuracy | StudyLens coverage | B2 coverage |
|---|---|---|---|---|
| "Parts of" (60 diagrams, 673 gold) | 44.8% | 65.8% | 15.5% | 34.2% |
| Arrows (40 diagrams, 179 gold) | 59.8% | 64.0% | 39.1% | 46.4% |

**Scene-graph regions by source** (all scene-graph parts that match a gold label, not only the cited ones):

| Region source | Correct | Accuracy |
|---|---|---|
| batch grounding call | 248 / 487 | 50.9% |
| enclosing shape | 3 / 14 | 21.4% |
| label box (fallback) | 3 / 55 | 5.5% |

**Other measures:**
- Invented references: StudyLens 0 (by construction: references are linked only to scene-graph parts); B2 38
  of 662 (5.7%) name text that OCR did not find in the diagram. OCR misses can inflate the B2 count.
- Flow-reference precision: 88 of 230 cited arrows (38.3%) are AI2D gold arrows.

### Multiple choice (125 questions)

| Condition | Accuracy [95% CI] | vs E2 |
|---|---|---|
| E2: image → Qwen | **94.4%** [90.4, 97.6] | -- |
| E3, default policy (abstain when the query finds no supporting arrow) | 67.2% [58.4, 75.2] | worse, 36 vs 2, p < 0.001 |
| E3, no abstention (model fallback instead) | 88.0% [82.4, 93.6] | worse, 10 vs 2, p = 0.039 |

- E3 abstained on 27 questions (21.6%), all counted as wrong; that policy was tuned on the synthetic pilot,
  where arrow questions dominate.
- The graph query answered only 29 of 125 questions (22 correct vs 27 for E2 on the same questions); the rest
  went to the model fallback or abstained.
- By category, without abstention: "parts of" 92.0% vs E2 96.0%; arrows 85.3% vs E2 93.3%.
- E2's 94.4% is higher than Qwen2.5-VL-3B's published AI2D score (81.6%) because this is a small, filtered
  sample (diagrams with at least 3 located labels or at least 2 gold arrows); the two are not comparable.

### What this evaluation shows

- **On real AI2D diagrams, the plain model's own boxes beat ours.** When Qwen explains and boxes the parts in
  one answer, 65% of its boxes land on the right part; our scene-graph regions land 50% of the time. Most of
  our regions come from the separate batch grounding call (50.9%); the label-box fallback is nearly always
  wrong (it marks the text, not the part), and AI2D anatomy parts are rarely inside a closed shape. On the
  synthetic architecture diagrams the opposite held (148/149 vs 11%), so region finding has to depend on the
  diagram type.
- **Our explanations cite fewer parts** (274 vs 376 references on "parts of" diagrams), which lowers coverage.
- **Our references are never invented**, while 5.7% of B2's are: the trade-off is fewer, checkable references
  vs more, partly unchecked ones.
- **Arrow extraction is the weak point on AI2D:** only 38% of cited arrows are gold arrows, so graph answers
  are often wrong or abstain, and on these mostly non-structural questions E3 loses to the plain model. This
  matches the pilot's conclusion that extraction is the bottleneck, and is worse on real diagrams.
- **Changes this points to:** use the model's in-answer boxes (or per-part grounding) for "parts of" diagrams
  and keep shapes for box-and-arrow diagrams; never fall back to the label box; abstain only on arrow
  questions; improve arrow extraction (planned retraining on scene-graph targets).

Limitations: one run, one seed, 100 diagrams; matching references to gold labels is by normalised name, so
references with paraphrased names are not scored (184 StudyLens and 169 B2 references were unmatched).
