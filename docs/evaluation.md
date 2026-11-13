# Pilot evaluation (Phase 2)

**Status: small pilot on synthetic data. It shows that the pipeline and the measurement work.
It is not evidence that graph grounding generalises.** The full comparison (more diagrams,
AI2D, paraphrased questions, significance testing) is Phase 3 work.

## Data

`scripts/make_synthetic.py --seed 458` (committed output in `data/synthetic/`):

* **10 diagrams** at 900×600 px with rounded boxes, text labels and straight black arrows with
  filled heads:
  * 7 hand-designed: abstract A→B→C→D chain, data-processing flowchart, food chain, a
    **counterfactual food chain** with every arrow reversed (its rendered caption matches the
    real food chain, so the image gives no hint), water cycle (a cycle), web-service
    architecture, photosynthesis inputs/outputs;
  * 3 random layered DAGs with **nonsense labels** (Zorp, Blick…), so prior knowledge cannot help.
* **79 templated questions** with gold answers computed from the gold graph:

| Kind | Template | Gold | n |
|---|---|---|---|
| predecessors | Which component directly feeds into X? | set of labels | 20 |
| successors | What does X point to? | set of labels | 10 |
| edge_true | Does A point to B? (A→B drawn) | yes | 10 |
| edge_reversed | Does B point to A? (A→B drawn) | no (unless both drawn) | 10 |
| path | Is there a path from A to B? | yes/no | 10 |
| unsupported_no_edge | Which component directly feeds into S? (S has no incoming arrow) | abstain | 9 |
| unsupported_non_structural | What colour is X in real life? | abstain | 10 |

## Conditions (identical diagrams and questions)

| Condition | Graph | Answerer |
|---|---|---|
| **E2** direct VLM | none | Qwen2.5-VL-3B sees the image + question (`DIRECT_QA_PROMPT`) |
| **E3** graph-grounded | extracted by Qwen2.5-VL-3B (`EXTRACTION_PROMPT`) | deterministic `graph_qa.answer_question` |
| **E3\*** upper bound | gold graph | deterministic `graph_qa.answer_question` |

E1 (OCR text only) was not run because PaddleOCR is not integrated yet.

## Scoring rules (`scripts/evaluate.py`)

* **set**: E3/E3\* are correct iff the answer's node set equals the gold set exactly. E2 is
  correct iff the set of entity labels mentioned in its text, minus the question's own entities
  (unless they are gold), equals the gold set; "Not shown" counts as wrong.
* **yes/no**: E3/E3\* answers start with "Yes"/"No". E2 is correct iff its text *starts with*
  the right word; anything else (hedging, a list) counts as wrong.
* **abstain**: E3/E3\* are correct iff the status is `abstained`. E2 is correct iff its text
  contains an abstention phrase (`not shown`, `cannot`, `does not show`, `no component`,
  `nothing`, `none`, `no arrow`, …).
* Extraction: entity P/R/F1 on normalised labels, and directed-edge P/R/F1 on (source label,
  target label) pairs, micro-averaged. A *reversed edge* is a predicted B→A where the gold graph
  has A→B.

The E2 string-matching rules are a known source of noise. Every raw answer is stored in
`results/eval_full.json` so each scoring decision can be audited.

## Results

Single run: `python scripts/evaluate.py` on 2026-11-09, Apple M4 (16 GB, MPS, bf16), Qwen2.5-VL-3B-Instruct,
greedy decoding, wall time 36 min. Raw outputs: `results/eval_full.json`; table: `results/eval_full.md`.

### Question answering (same 79 questions for all conditions)

| Question kind | n | E2 direct VLM | E3 VLM graph + QA | E3* gold graph + QA |
|---|---|---|---|---|
| predecessors | 20 | 14 (70%) | 12 (60%) | 20 (100%) |
| successors | 10 | 6 (60%) | 5 (50%) | 10 (100%) |
| edge_true | 10 | 9 (90%) | 7 (70%) | 10 (100%) |
| edge_reversed | 10 | **1 (10%)** | **8 (80%)** | 10 (100%) |
| path | 10 | 7 (70%) | 6 (60%) | 10 (100%) |
| unsupported_no_edge (should abstain) | 9 | 4 (44%) | 6 (67%) | 9 (100%) |
| unsupported_non_structural (should abstain) | 10 | 8 (80%) | 10 (100%) | 10 (100%) |
| **All** | **79** | **49 (62%)** | **54 (68%)** | **79 (100%)** |

**Paired comparison, E2 vs E3:**

| Outcome | Questions |
|---|---|
| Both correct | 41 |
| Only E3 correct | 13 |
| Only E2 correct | 8 |
| Both wrong | 17 |

An exact two-sided sign test on the 21 discordant questions gives **p = 0.38**. With this sample
**there is no statistically meaningful difference between E2 and E3 overall.** We do not claim
that graph grounding improves accuracy yet.

### Extraction (Qwen2.5-VL-3B vs gold graph)

| | Precision | Recall | F1 |
|---|---|---|---|
| Entities (labels) | 1.00 | 1.00 | 1.00 |
| Directed edges | 0.70 | 0.67 | 0.68 |

* 5 of 10 diagrams were extracted perfectly: abstract chain, flowchart, food chain, web
  architecture and photosynthesis. All of them are acyclic, with labels whose relations match
  common knowledge.
* There were **8 reversed edges**, 5 of them on the counterfactual food chain.
* Mean latency was **86 s per extraction** and **16 s per direct answer**. The model ignored the
  "compact JSON, no code fences" instruction and pretty-printed its JSON, which costs tokens.

### What the pilot shows (observations, not general claims)

1. **E3\* = 100% while E3 = 68%.** Every E3 error comes from extraction, not from the graph QA.
   Note that E3\* questions use the same templates the parser was built for, so this does not
   show that the parser handles free-form questions.
2. **E2 has a strong "Yes" bias.** It answered "Yes" to 9 of 10 questions about arrows drawn
   in the *opposite* direction, so its 90% on true-edge questions is not evidence that it reads
   arrow direction. E3 answered 8 of 10 correctly and cited the actual arrow, e.g. *"No. There is no
   arrow Report -> Store; the arrow between them goes the other way (Store -> Report)."* E3's two
   misses are on mis-extracted diagrams.
3. **Abstention.** E3 abstained correctly on 16 of 19 unsupported questions; E2 on 12 of 19. E2
   answered "Green" or "Blue" to "What colour is X in real life?" and named non-existent
   predecessors ("Which component directly feeds into Client?" → "Load Balancer").
4. **Prior-knowledge override hurts E3 most when extraction is wrong.** On the counterfactual
   food chain, Qwen extracted the biologically "expected" arrows (Sun → Grass → Rabbit → Fox)
   instead of the drawn reversed ones. E3 then scored 1 of 8 on that diagram (E2: 4 of 8), with
   confident answers that cite wrong edges. Graph grounding makes this error **visible and
   correctable** in the UI, since the student can see the reversed arrows in the overlay, but it
   does not prevent it.
5. **Knowledge-free graphs are hard for both.** On the three nonsense-label graphs, edge F1 was
   0.40–0.55. E3 got 9 of 24 questions right and E2 got 10 of 24.
6. Of the 25 E3 errors, 20 were confident wrong answers built on wrong edges and 5 were wrong
   abstentions (missing edges).

### Known extraction errors (from `results/eval_full.md`)

| Diagram | Missing edges | Spurious edges |
|---|---|---|
| Counterfactual food chain | all 5 drawn edges | all 5 reversed |
| Water cycle (cycle) | collection→evaporation, precipitation→collection | evaporation→collection |
| Nonsense graph 1 | krell→zorp, yoma→krell, yoma→vint | krell→vint, krell→yoma |
| Nonsense graph 2 | quan→krell, silk→blick, tesh→silk | krell→silk, tesh→krell, tesh→quan |
| Nonsense graph 3 | krell→tesh, mivo→zorp, rulo→mivo | rulo→krell, tesh→mivo, zorp→mivo |

### Limits of this evaluation

* There is a single run, n = 79 templated questions on 10 clean synthetic diagrams, with one
  annotator (the generator). There are no real textbook diagrams, no paraphrases, and no
  human-written questions.
* E2 scoring uses string rules, and all 30 E2 failures were checked by hand.
* The prompts were written before the run and not tuned on these questions, but the same 10
  diagrams served as the development check for the parser (E3\*).
* Phase 3 needs more data, paraphrased questions, AI2D, and confidence intervals before any
  claim about E2 vs E3.
