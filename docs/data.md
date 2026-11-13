# Data

## AI2D: real data (training and main evaluation)

**Source:** A. Kembhavi et al., *A Diagram Is Worth a Dozen Images*, ECCV 2016. Allen Institute
for AI. Download with `python scripts/get_ai2d.py --accept-license`, which fetches:

* `ai2d-all.zip` (about 1 GB) from `ai2-public-datasets.s3.amazonaws.com/diagrams/`;
* the official test split ids, `ai2d_test_ids.csv` (982 ids, of which 981 have images).

Both go into `external/ai2d/`, which is git-ignored.

**Licence (checked against `ai2d/license.txt` in the archive):** "License for Non-Commercial
Use". Its terms:

* use is allowed by academic, non-profit or government-sponsored researchers, and for
  evaluation;
* no use in a for-profit enterprise;
* **no right to modify and publicly release the data in any form**;
* **no right to distribute the data to a third party**;
* only short segments or still images may appear in academic publications.

> The Phase 1 report said AI2D was "reported CC BY-SA; to confirm". That was **wrong**: the
> licence shipped with the data is the non-commercial licence above.

**What this repository does because of that licence:**
* No AI2D images, annotations, questions or converted graphs are committed.
* Raw model outputs on AI2D, which transcribe diagram text, are cached only in git-ignored
  `external/ai2d_runs/`.
* `results/ai2d_eval.json` contains only question/image ids, chosen option indices, statuses,
  counts and metrics.
* The Evaluation tab shows question text and images **only from the user's own local AI2D
  copy**, through `/api/eval/{run}/item/{qid}` and `/api/ai2d/image/{id}`.
* The fine-tuned LoRA adapter is a model derived from AI2D. It is kept in git-ignored
  `checkpoints/` and is not redistributed.

**Contents used:**

| | Train | Test (official) |
|---|---|---|
| Diagrams | 3,922 | 981 |
| Diagrams whose annotations contain ≥1 directed arrow (after label mapping) | 1,200 | 290 |
| Multiple-choice questions, excluding letter-label questions | 7,804 | 1,904 |

Questions are human-written, with 4 options. We exclude the 5,793 `abcLabel` questions ("What
is A in the diagram?"), which refer to letter labels that must be painted over the image.

**Gold graph construction** (`studylens/datasets/ai2d.py`):
* **Nodes** are text elements (label = text value). Identical labels are merged.
* **Blobs** (drawn objects) are mapped to a label through `intraObjectLabel`. If a blob has
  several labels, the nearest one is used.
* **Arrows** are directed `interObjectLinkage` relations, origin → destination. We checked on
  food-web diagrams that origin is the tail and destination the arrowhead. Endpoints that are
  unlabelled blobs are dropped: 1,769 links in train and 543 in test.

**Known annotation noise:**
* some blobs are linked to column headers rather than names (handled by the nearest-label rule);
* truncated labels ("Zoo-plankto") are kept as written;
* undirected linkages are ignored.

**Splits:**
* Training uses only the train split. 20 train diagrams are held out for validation loss.
* Evaluation uses only the official test split.
* The evaluation sample is seeded (seed 458): 100 test diagrams, 60 drawn from those with gold
  arrows and 40 from those without, with up to 8 questions each.

## Synthetic pilot (`data/synthetic/`, created for this project)

`scripts/make_synthetic.py --seed 458` generates **27 diagrams with 304 questions** whose gold
answers are computed exactly from the drawn graph. This is the proposal's ~200-question pilot.
Everything is created by this project and committed.

* **15 hand-designed diagrams:**
  * food chain and pond food web;
  * frog life cycle and water cycle;
  * photosynthesis;
  * data-processing flowchart, web-service architecture, CI/CD pipeline, home network,
    compiler stages, abstract chain;
  * **4 counterfactual versions** (food chain, pond food web, frog life cycle, water cycle) with
    **every arrow reversed**. Each shows the same caption as the real diagram, so only the
    arrows reveal the change.
* **12 knowledge-free graphs** with nonsense labels (Zorp, Blick…), 3 or 4 layers. Prior
  knowledge cannot help with these.
* **Questions:** 12 kinds, with varied natural wording rather than one template per kind.
  Food diagrams use food-web wording ("What eats the X?").
  * predecessors and successors;
  * ancestors and descendants ("directly or indirectly");
  * edge present / reversed;
  * path;
  * number of incoming arrows;
  * sources and sinks;
  * two kinds that should be **abstained**: nothing points into the entity, or the question is
    not structural ("What colour is X in real life?").

The first, smaller pilot (10 diagrams, 79 templated questions) and its results are kept in
`results/pilot_v1/` for reference.

**Limits.** These are clean, straight-arrow, English diagrams. The keyword rules of the graph
query were debugged on these questions, so the pilot measures the pipeline under favourable
conditions. AI2D is the test on real diagrams and human-written questions.
