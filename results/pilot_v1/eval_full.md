# Evaluation summary (full)

Diagrams: 10 · Questions: 79 · Wall time: 2169.1 s

| Question kind | n | E2 | E3 | E3* |
|---|---|---|---|---|
| edge_reversed | 10 | 1/10 (10%) | 8/10 (80%) | 10/10 (100%) |
| edge_true | 10 | 9/10 (90%) | 7/10 (70%) | 10/10 (100%) |
| path | 10 | 7/10 (70%) | 6/10 (60%) | 10/10 (100%) |
| predecessors | 20 | 14/20 (70%) | 12/20 (60%) | 20/20 (100%) |
| successors | 10 | 6/10 (60%) | 5/10 (50%) | 10/10 (100%) |
| unsupported_no_edge | 9 | 4/9 (44%) | 6/9 (67%) | 9/9 (100%) |
| unsupported_non_structural | 10 | 8/10 (80%) | 10/10 (100%) | 10/10 (100%) |
| **ALL** | 79 | 49/79 (62%) | 54/79 (68%) | 79/79 (100%) |

## Extraction (Qwen2.5-VL-3B vs gold graph, micro-averaged)

| | P | R | F1 |
|---|---|---|---|
| Entities | 1.0 | 1.0 | 1.0 |
| Edges (directed) | 0.696 | 0.667 | 0.681 |

Reversed edges (predicted B->A where gold is A->B): 8
Mean extraction latency: 86.4 s; mean direct-answer latency: 16.2 s

| Diagram | entity F1 | edge F1 | reversed | missing | extra |
|---|---|---|---|---|---|
| Abstract chain | 1.00 | 1.00 | 0 | – | – |
| Data-processing flowchart | 1.00 | 1.00 | 0 | – | – |
| Food chain (energy flow) | 1.00 | 1.00 | 0 | – | – |
| Counterfactual food chain (arrows reversed) | 1.00 | 0.00 | 5 | deer->grass, fox->rabbit, grass->sun, rabbit->grass, wolf->deer | deer->wolf, grass->deer, grass->rabbit, rabbit->fox, sun->grass |
| Water cycle | 1.00 | 0.57 | 1 | collection->evaporation, precipitation->collection | evaporation->collection |
| Web service architecture | 1.00 | 1.00 | 0 | – | – |
| Photosynthesis inputs and outputs | 1.00 | 1.00 | 0 | – | – |
| Nonsense-label graph 1 | 1.00 | 0.55 | 1 | krell->zorp, yoma->krell, yoma->vint | krell->vint, krell->yoma |
| Nonsense-label graph 2 | 1.00 | 0.40 | 0 | quan->krell, silk->blick, tesh->silk | krell->silk, tesh->krell, tesh->quan |
| Nonsense-label graph 3 | 1.00 | 0.40 | 1 | krell->tesh, mivo->zorp, rulo->mivo | rulo->krell, tesh->mivo, zorp->mivo |
