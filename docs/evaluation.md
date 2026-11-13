# Evaluation

Two evaluations exist:
- the **synthetic pilot**, which is complete;
- the **AI2D grounded-explanation evaluation**, which is in progress.

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

## 2. AI2D grounded-explanation evaluation (in progress)

`scripts/evaluate_grounding.py`, seed 458:
- 50 AI2D **test** diagrams: 30 "parts of" diagrams and 20 with arrows (food webs, life cycles, cycles);
- 433 annotated part locations;
- 65 multiple-choice questions.

**Systems compared:**

| System | What it does |
|---|---|
| StudyLens | scene graph → explanation → references linked deterministically |
| B2 | the same Qwen2.5-VL-3B asked to explain the diagram and give a box for each part it mentions (its native grounding, no scene graph, no checks) |

**Metrics:**
- **Pointing accuracy:** a referenced part's highlight contains the gold leader-line tip (padded by 3% of
  the image diagonal), or its centre lies inside the labelled object.
- **Coverage** of annotated parts.
- **Flow-reference precision** against AI2D's gold arrows.
- **Invented references.**
- **Scene-graph region accuracy**, by how each region was found.
- **Multiple-choice accuracy:** E2 vs E3 (McNemar).

Results will be added here when the run finishes. Raw outputs stay in `external/` (AI2D licence).
