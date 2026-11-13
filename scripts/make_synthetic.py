"""Generate the synthetic StudyLens sample set: diagram PNGs + gold graphs + gold questions.

All data is synthetic and created by this script (no third-party dataset), so it can be
redistributed with the repository. Output is deterministic for a given --seed.

    python scripts/make_synthetic.py            # writes data/synthetic/*.png + manifest.json

Each diagram has: boxes with text labels, straight arrows with filled arrowheads, the exact
gold graph (node boxes in pixel coordinates), and templated questions with gold answers:
  set      : the exact set of entity labels that answers the question
  yesno    : true / false
  abstain  : the graph cannot support an answer (the correct behaviour is to abstain)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic"
W, H = 900, 600
BOX_W, BOX_H = 150, 56

# ---------------------------------------------------------------------------
# Hand-designed diagrams: (title, {label: (cx, cy)}, [(src, tgt), ...])
# Positions are box centres in a 900x600 canvas.
# ---------------------------------------------------------------------------
HANDMADE = [
    ("Abstract chain", {"A": (120, 300), "B": (340, 300), "C": (560, 300), "D": (780, 300)},
     [("A", "B"), ("B", "C"), ("C", "D")]),
    ("Data-processing flowchart",
     {"Input": (110, 300), "Parse": (330, 300), "Validate": (550, 300), "Store": (780, 160),
      "Reject": (780, 440), "Report": (550, 100)},
     [("Input", "Parse"), ("Parse", "Validate"), ("Validate", "Store"), ("Validate", "Reject"), ("Store", "Report")]),
    ("Food chain (energy flow)",
     {"Sun": (110, 300), "Grass": (330, 300), "Rabbit": (550, 200), "Deer": (550, 430), "Fox": (780, 200), "Wolf": (780, 430)},
     [("Sun", "Grass"), ("Grass", "Rabbit"), ("Grass", "Deer"), ("Rabbit", "Fox"), ("Deer", "Wolf")]),
    # Counterfactual: same entities, arrows deliberately reversed. Tests whether answers follow
    # the drawn arrows rather than prior biology knowledge.
    ("Counterfactual food chain (arrows reversed)",
     {"Sun": (110, 300), "Grass": (330, 300), "Rabbit": (550, 200), "Deer": (550, 430), "Fox": (780, 200), "Wolf": (780, 430)},
     [("Grass", "Sun"), ("Rabbit", "Grass"), ("Deer", "Grass"), ("Fox", "Rabbit"), ("Wolf", "Deer")]),
    ("Water cycle",
     {"Evaporation": (450, 90), "Condensation": (770, 300), "Precipitation": (450, 510), "Collection": (130, 300)},
     [("Evaporation", "Condensation"), ("Condensation", "Precipitation"), ("Precipitation", "Collection"), ("Collection", "Evaporation")]),
    ("Web service architecture",
     {"Client": (110, 300), "Load Balancer": (340, 300), "Server 1": (570, 160), "Server 2": (570, 440), "Database": (790, 300)},
     [("Client", "Load Balancer"), ("Load Balancer", "Server 1"), ("Load Balancer", "Server 2"), ("Server 1", "Database"), ("Server 2", "Database")]),
    ("Photosynthesis inputs and outputs",
     {"Light": (130, 110), "Water": (130, 300), "CO2": (130, 490), "Chloroplast": (450, 300), "Glucose": (770, 190), "Oxygen": (770, 410)},
     [("Light", "Chloroplast"), ("Water", "Chloroplast"), ("CO2", "Chloroplast"), ("Chloroplast", "Glucose"), ("Chloroplast", "Oxygen")]),
    # ---- added for the 200-question pilot (v2) ----
    ("Pond food web",
     {"Algae": (110, 300), "Water flea": (340, 160), "Snail": (340, 440), "Small fish": (570, 300), "Heron": (790, 300)},
     [("Algae", "Water flea"), ("Algae", "Snail"), ("Water flea", "Small fish"), ("Snail", "Small fish"), ("Small fish", "Heron")]),
    ("Counterfactual pond food web (arrows reversed)",
     {"Algae": (110, 300), "Water flea": (340, 160), "Snail": (340, 440), "Small fish": (570, 300), "Heron": (790, 300)},
     [("Water flea", "Algae"), ("Snail", "Algae"), ("Small fish", "Water flea"), ("Small fish", "Snail"), ("Heron", "Small fish")]),
    ("Frog life cycle",
     {"Egg": (450, 90), "Tadpole": (770, 300), "Froglet": (450, 510), "Adult frog": (130, 300)},
     [("Egg", "Tadpole"), ("Tadpole", "Froglet"), ("Froglet", "Adult frog"), ("Adult frog", "Egg")]),
    ("Counterfactual frog life cycle (arrows reversed)",
     {"Egg": (450, 90), "Tadpole": (770, 300), "Froglet": (450, 510), "Adult frog": (130, 300)},
     [("Tadpole", "Egg"), ("Froglet", "Tadpole"), ("Adult frog", "Froglet"), ("Egg", "Adult frog")]),
    ("CI/CD pipeline",
     {"Commit": (110, 300), "Build": (330, 300), "Unit tests": (550, 160), "Lint": (550, 440), "Deploy": (780, 160), "Notify": (780, 440)},
     [("Commit", "Build"), ("Build", "Unit tests"), ("Build", "Lint"), ("Unit tests", "Deploy"), ("Lint", "Notify"), ("Unit tests", "Notify")]),
    ("Home network",
     {"Internet": (110, 300), "Modem": (330, 300), "Router": (550, 300), "Laptop": (780, 110), "Printer": (780, 300), "Phone": (780, 490)},
     [("Internet", "Modem"), ("Modem", "Router"), ("Router", "Laptop"), ("Router", "Printer"), ("Router", "Phone")]),
    ("Compiler stages",
     {"Source code": (120, 130), "Lexer": (340, 130), "Parser": (560, 130), "Type checker": (780, 300), "Optimizer": (560, 470), "Code generator": (300, 470)},
     [("Source code", "Lexer"), ("Lexer", "Parser"), ("Parser", "Type checker"), ("Type checker", "Optimizer"), ("Optimizer", "Code generator")]),
    ("Counterfactual water cycle (arrows reversed)",
     {"Evaporation": (450, 90), "Condensation": (770, 300), "Precipitation": (450, 510), "Collection": (130, 300)},
     [("Condensation", "Evaporation"), ("Precipitation", "Condensation"), ("Collection", "Precipitation"), ("Evaporation", "Collection")]),
]

NONSENSE = ["Zorp", "Blick", "Quan", "Mivo", "Tesh", "Rulo", "Pax", "Vint", "Krell", "Yoma", "Dube", "Silk"]


def random_dag(rng: random.Random, idx: int):
    """Knowledge-free layered DAG with nonsense labels (no prior knowledge can help).
    3 or 4 layers of 2 nodes; arrows only between neighbouring layers (no arrow crosses a box)."""
    n_layers = rng.choice([3, 3, 4])
    labels = rng.sample(NONSENSE, 2 * n_layers)
    layers = [labels[2 * i : 2 * i + 2] for i in range(n_layers)]
    step = 310 if n_layers == 3 else 220
    x0 = 140 if n_layers == 3 else 115
    pos = {}
    for li, layer in enumerate(layers):
        for j, lab in enumerate(layer):
            pos[lab] = (x0 + li * step, 170 + j * 260 + rng.randint(-30, 30))
    edges = set()
    for li in range(n_layers - 1):
        for t in layers[li + 1]:
            for s in rng.sample(layers[li], rng.choice([1, 1, 2])):
                edges.add((s, t))

    # make sure every first-layer node has an outgoing arrow
    for s in layers[0]:
        if not any(e[0] == s for e in edges):
            edges.add((s, rng.choice(layers[1])))
    return (f"Nonsense-label graph {idx}", pos, sorted(edges))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 bundles a scalable font
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


def box_of(c):
    x, y = c
    return [x - BOX_W / 2, y - BOX_H / 2, x + BOX_W / 2, y + BOX_H / 2]


def border_point(b, tx, ty, pad=4):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dx, dy = tx - cx, ty - cy
    hw, hh = (b[2] - b[0]) / 2 + pad, (b[3] - b[1]) / 2 + pad
    s = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * s, cy + dy * s


# Text drawn on the image. The counterfactual diagram deliberately shows the SAME caption as the
# real food chain so the image itself gives no hint that the arrows were reversed.
DISPLAY_TITLE = {"Counterfactual food chain (arrows reversed)": "Food chain (energy flow)",
                 "Counterfactual pond food web (arrows reversed)": "Pond food web",
                 "Counterfactual frog life cycle (arrows reversed)": "Frog life cycle",
                 "Counterfactual water cycle (arrows reversed)": "Water cycle"}
FOOD = ("food", "Food")


def render(title, pos, edges, path: Path):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 14), DISPLAY_TITLE.get(title, title), fill="#444444", font=font(20))
    f = font(22)
    boxes = {lab: box_of(c) for lab, c in pos.items()}
    for s, t in edges:
        bs, bt = boxes[s], boxes[t]
        ct = ((bt[0] + bt[2]) / 2, (bt[1] + bt[3]) / 2)
        cs = ((bs[0] + bs[2]) / 2, (bs[1] + bs[3]) / 2)
        p0 = border_point(bs, *ct)
        p1 = border_point(bt, *cs)
        d.line([p0, p1], fill="black", width=3)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        L, Wd = 18, 9
        bx, by = p1[0] - L * math.cos(ang), p1[1] - L * math.sin(ang)
        d.polygon(
            [p1, (bx + Wd * math.sin(ang), by - Wd * math.cos(ang)), (bx - Wd * math.sin(ang), by + Wd * math.cos(ang))],
            fill="black",
        )
    for lab, b in boxes.items():
        d.rounded_rectangle(b, radius=10, outline="#1f3b73", width=3, fill="#eaf1ff")
        tw = d.textlength(lab, font=f)
        d.text(((b[0] + b[2]) / 2 - tw / 2, (b[1] + b[3]) / 2 - 13), lab, fill="black", font=f)
    img.save(path, format="PNG", optimize=False)
    return img


# ---------------------------------------------------------------------------
# Gold questions
# ---------------------------------------------------------------------------
# Varied natural wording (not one fixed template per question type), so the pilot also tests
# whether each pipeline understands free-form questions.
PRED = ["Which component directly feeds into {x}?", "What points to {x}?", "Which boxes have an arrow going into {x}?",
        "What comes right before {x}?", "Where do the arrows into {x} come from?"]
SUCC = ["What does {x} point to?", "Where do the arrows from {x} go?", "What comes right after {x}?",
        "Which boxes does {x} feed into directly?", "{x} leads directly to what?"]
FOOD_PRED = ["What does the {x} eat?", "What does {x} feed on?"]
FOOD_SUCC = ["What eats the {x}?", "Which organism eats {x}?"]
ANC = ["Which entities lead to {x}, directly or indirectly?", "List everything upstream of {x}."]
DESC = ["Name everything reachable from {x} by following the arrows.", "What is downstream of {x}, directly or indirectly?"]
EDGE = ["Does {a} point to {b}?", "Is there an arrow from {a} to {b}?", "Does an arrow go from {a} to {b}?"]
PATH = ["Is there a path from {a} to {b}?", "Can you get from {a} to {b} by following the arrows?"]
COUNT_IN = ["How many arrows point into {x}?", "How many arrows end at {x}?"]
SOURCES = ["Which entities have no incoming arrows?", "Where does the diagram start (no arrows coming in)?"]
SINKS = ["Which entities have no outgoing arrows?", "Which boxes are end points, with no arrows leaving them?"]
NONSTRUCT = ["What colour is {x} in real life?", "How large is {x} usually?", "What is {x} made of?"]


def make_questions(title, pos, edges, rng: random.Random):
    G = nx.DiGraph()
    G.add_nodes_from(pos)
    G.add_edges_from(edges)
    nodes = list(pos)
    food = any(w in title for w in FOOD)
    qs = []

    def add(text, kind, gold, ents):
        qs.append({"question": text, "kind": kind, "gold": gold, "entities": ents})

    with_in = [n for n in nodes if G.in_degree(n) > 0]
    with_out = [n for n in nodes if G.out_degree(n) > 0]
    no_in = [n for n in nodes if G.in_degree(n) == 0]
    x = rng.choice(with_in)
    add(rng.choice(FOOD_PRED if food else PRED).format(x=x), "predecessors", {"type": "set", "value": sorted(G.predecessors(x))}, [x])
    x = rng.choice(with_out)
    add(rng.choice(FOOD_SUCC if food else SUCC).format(x=x), "successors", {"type": "set", "value": sorted(G.successors(x))}, [x])
    a, b = rng.choice(edges)
    add(rng.choice(EDGE).format(a=a, b=b), "edge_true", {"type": "yesno", "value": True}, [a, b])
    a, b = rng.choice(edges)
    add(rng.choice(EDGE).format(a=b, b=a), "edge_reversed", {"type": "yesno", "value": G.has_edge(b, a)}, [b, a])
    a, b = rng.sample(nodes, 2)
    add(rng.choice(PATH).format(a=a, b=b), "path", {"type": "yesno", "value": nx.has_path(G, a, b)}, [a, b])
    is_dag = nx.is_directed_acyclic_graph(G)
    if is_dag:  # ancestors / descendants are not meaningful in a cycle
        x = rng.choice([n for n in nodes if len(nx.ancestors(G, n)) >= 2] or with_in)
        add(rng.choice(ANC).format(x=x), "ancestors", {"type": "set", "value": sorted(nx.ancestors(G, x))}, [x])
        x = rng.choice([n for n in nodes if len(nx.descendants(G, n)) >= 2] or with_out)
        add(rng.choice(DESC).format(x=x), "descendants", {"type": "set", "value": sorted(nx.descendants(G, x))}, [x])
        add(rng.choice(SOURCES), "sources", {"type": "set", "value": sorted(n for n in no_in if G.out_degree(n) > 0)}, [])
        add(rng.choice(SINKS), "sinks", {"type": "set", "value": sorted(n for n in nodes if G.out_degree(n) == 0 and G.in_degree(n) > 0)}, [])
    x = rng.choice(nodes)
    add(rng.choice(COUNT_IN).format(x=x), "count_in", {"type": "count", "value": G.in_degree(x)}, [x])
    if no_in:
        x = rng.choice(no_in)
        add(rng.choice(PRED).format(x=x), "unsupported_no_edge", {"type": "abstain"}, [x])
    x = rng.choice(nodes)
    add(rng.choice(NONSTRUCT).format(x=x), "unsupported_non_structural", {"type": "abstain"}, [x])
    return qs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=458)
    ap.add_argument("--n-random", type=int, default=12)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    specs = list(HANDMADE) + [random_dag(rng, i + 1) for i in range(args.n_random)]
    diagrams = []
    for i, (title, pos, edges) in enumerate(specs, start=1):
        fname = f"diagram_{i:02d}.png"
        render(title, pos, edges, OUT / fname)
        ids = {lab: f"n{k + 1}" for k, lab in enumerate(pos)}
        graph = {
            "nodes": [{"id": ids[lab], "label": lab, "bbox": box_of(c), "confidence": 1.0} for lab, c in pos.items()],
            "edges": [{"source": ids[s], "target": ids[t], "relation": "points_to", "confidence": 1.0} for s, t in edges],
            "source": "annotation",
            "image_size": [W, H],
        }
        diagrams.append({
            "file": fname,
            "title": title,
            "sha256": hashlib.sha256((OUT / fname).read_bytes()).hexdigest(),
            "graph": graph,
            "questions": make_questions(title, pos, edges, rng),
        })
    manifest = {"version": 2, "seed": args.seed, "generator": "scripts/make_synthetic.py", "license": "synthetic, created for this project",
                "diagrams": diagrams}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    nq = sum(len(d["questions"]) for d in diagrams)
    print(f"wrote {len(diagrams)} diagrams and {nq} questions to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
