"""Create demo inputs for trying the app by hand.

    python scripts/make_demo_inputs.py

1. data/demo_inputs/      NEW diagrams made for this project (not in AI2D, not used in training),
                          in several formats / conditions, plus README.md with questions and the
                          correct answers. Safe to commit.
2. external/demo_ai2d/    a few REAL textbook diagrams from the AI2D *test* split (never used for
                          training) with their human-written questions. Local only: the AI2D licence
                          forbids redistribution, and external/ is git-ignored.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_synthetic as ms  # noqa: E402

OUT = ROOT / "data" / "demo_inputs"
AI2D_OUT = ROOT / "external" / "demo_ai2d"

DIAGRAMS = [
    # file, title, positions, edges, how it is saved, questions [(question, correct answer)]
    ("butterfly_life_cycle.png", "Butterfly life cycle",
     {"Egg": (450, 110), "Caterpillar": (760, 300), "Pupa": (450, 500), "Butterfly": (140, 300)},
     [("Egg", "Caterpillar"), ("Caterpillar", "Pupa"), ("Pupa", "Butterfly"), ("Butterfly", "Egg")], "png",
     [("What stage comes after the caterpillar?", "Pupa (arrow Caterpillar -> Pupa)"),
      ("What does the butterfly produce in this cycle?", "Egg (arrow Butterfly -> Egg)"),
      ("Is this a cycle? Explain using the arrows.", "Yes: Egg -> Caterpillar -> Pupa -> Butterfly -> Egg")]),
    ("digestion_flow.jpg", "Path of food through the digestive system",
     {"Mouth": (110, 150), "Esophagus": (330, 150), "Stomach": (560, 150), "Small intestine": (560, 420), "Large intestine": (300, 420)},
     [("Mouth", "Esophagus"), ("Esophagus", "Stomach"), ("Stomach", "Small intestine"), ("Small intestine", "Large intestine")], "jpeg",
     [("Where does food go right after the stomach?", "Small intestine"),
      ("Which organ comes before the stomach?", "Esophagus"),
      ("What is the last organ shown?", "Large intestine"),
      ("Which organ absorbs the most nutrients?", "Not shown by the arrows: should be 'not supported by the graph' / image-only")]),
    ("electricity_grid.webp", "Electricity from power plant to home",
     {"Power plant": (130, 300), "Step-up transformer": (380, 160), "Transmission lines": (640, 160), "Step-down transformer": (640, 440), "Home": (380, 440)},
     [("Power plant", "Step-up transformer"), ("Step-up transformer", "Transmission lines"), ("Transmission lines", "Step-down transformer"), ("Step-down transformer", "Home")], "webp",
     [("What feeds directly into the home?", "Step-down transformer"),
      ("Is there a path from the power plant to the home?", "Yes: Power plant -> Step-up transformer -> Transmission lines -> Step-down transformer -> Home"),
      ("Does the home send electricity to the power plant?", "No (no such arrow)")]),
    ("nitrogen_transparent.png", "Simplified nitrogen cycle",
     {"Atmospheric N2": (450, 100), "Nitrogen-fixing bacteria": (760, 300), "Soil nitrates": (450, 500), "Plants": (140, 300)},
     [("Atmospheric N2", "Nitrogen-fixing bacteria"), ("Nitrogen-fixing bacteria", "Soil nitrates"), ("Soil nitrates", "Plants")], "png_transparent",
     [("Where do plants get nitrogen from in this diagram?", "Soil nitrates"),
      ("What converts atmospheric N2?", "Nitrogen-fixing bacteria"),
      ("What feeds into atmospheric N2?", "Nothing in this diagram: should abstain / not supported")]),
    ("photo_rotated.jpg", "Pond food chain",
     {"Algae": (130, 300), "Water flea": (370, 300), "Small fish": (610, 300), "Heron": (800, 140)},
     [("Algae", "Water flea"), ("Water flea", "Small fish"), ("Small fish", "Heron")], "photo",
     [("What eats the water flea?", "Small fish (arrow Water flea -> Small fish)"),
      ("If the algae disappeared, which animals would be affected?", "Water flea, then Small fish and Heron (downstream of Algae)"),
      ("What is at the top of this food chain?", "Heron")]),
    ("water_cycle.gif", "Water cycle",
     {"Ocean": (150, 470), "Evaporation": (150, 150), "Clouds": (520, 120), "Rain": (760, 330), "Rivers": (520, 480)},
     [("Ocean", "Evaporation"), ("Evaporation", "Clouds"), ("Clouds", "Rain"), ("Rain", "Rivers"), ("Rivers", "Ocean")], "gif",
     [("What happens after evaporation?", "Clouds form (Evaporation -> Clouds)"),
      ("Where does rain go?", "Rivers"),
      ("What does the river flow into?", "Ocean")]),
]


def save(img: Image.Image, path: Path, how: str) -> None:
    if how == "jpeg":
        img.save(path, "JPEG", quality=85)
    elif how == "webp":
        img.save(path, "WEBP", quality=85)
    elif how == "png_transparent":
        rgba = img.convert("RGBA")
        px = rgba.load()
        for y in range(rgba.height):
            for x in range(rgba.width):
                r, g, b, _ = px[x, y]
                if r > 245 and g > 245 and b > 245:
                    px[x, y] = (255, 255, 255, 0)  # white background -> transparent
        rgba.save(path, "PNG")
    elif how == "photo":
        # imitate a phone photo: slight rotation, blur, noise, warm tint, stored sideways with an
        # EXIF orientation tag (the app must rotate it back)
        rng = random.Random(7)
        ph = img.rotate(-3, resample=Image.BICUBIC, expand=True, fillcolor=(235, 230, 220))
        ph = ph.filter(ImageFilter.GaussianBlur(1.1))
        noise = Image.effect_noise(ph.size, 18).convert("RGB")
        ph = Image.blend(ph, noise, 0.08)
        ph = Image.blend(ph, Image.new("RGB", ph.size, (255, 236, 200)), 0.12)
        sideways = ph.transpose(Image.ROTATE_90)  # raw pixels sideways ...
        exif = sideways.getexif()
        exif[0x0112] = 6  # ... EXIF says "rotate 90° CW to display"
        sideways.save(path, "JPEG", quality=80, exif=exif)
        _ = rng
    elif how == "gif":
        frames = [img.convert("P", palette=Image.ADAPTIVE), img.point(lambda v: v * 0.9).convert("P", palette=Image.ADAPTIVE)]
        frames[0].save(path, "GIF", save_all=True, append_images=frames[1:], duration=600, loop=0)
    else:
        img.save(path, "PNG")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Demo inputs (new diagrams, not in AI2D, not used for training)", "",
             "Generated by `scripts/make_demo_inputs.py`. Upload them in the Ask tab and try the questions.",
             "\"Correct answer\" is what the drawn arrows show.", ""]
    tmp = OUT / "_tmp.png"
    for fname, title, pos, edges, how, qs in DIAGRAMS:
        ms.render(title, pos, edges, tmp)
        save(Image.open(tmp).convert("RGB"), OUT / fname, how)
        lines += [f"## `{fname}` ({ {'png': 'PNG', 'jpeg': 'JPEG', 'webp': 'WebP', 'png_transparent': 'PNG with transparent background', 'photo': 'phone-style photo: blurred, noisy, stored sideways with EXIF rotation', 'gif': 'animated GIF (first frame is used)'}[how]})",
                  "", "| Question | Correct answer |", "|---|---|"] + [f"| {q} | {a} |" for q, a in qs] + [""]
    tmp.unlink()
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"wrote {len(DIAGRAMS)} new diagrams to {OUT.relative_to(ROOT)}")

    # ---- real AI2D test diagrams (local only)
    try:
        from studylens.datasets import ai2d
    except Exception:  # pragma: no cover
        return 0
    if not ai2d.available():
        print("AI2D not downloaded; skipping external/demo_ai2d")
        return 0
    AI2D_OUT.mkdir(parents=True, exist_ok=True)
    test = [d for d in ai2d.load("test") if d.questions]
    picks, seen = [], set()
    want = ["foodChainsWebs", "lifeCycles", "rockCycle", "waterCNPCycle", "moonPhaseEquinox", "partsOfA", "photosynthesisRespiration", "typesOf"]
    for cat in want:
        cands = [d for d in test if d.category == cat and (d.graph.edges or cat in ("moonPhaseEquinox", "partsOfA", "typesOf"))]
        if cands:
            d = sorted(cands, key=lambda d: -len(d.questions))[0]
            picks.append(d)
    md = ["# Real AI2D test diagrams (LOCAL ONLY - AI2 non-commercial licence, do not redistribute)", "",
          "From the official AI2D *test* split: never used for training. Questions and answers are the dataset's own.", ""]
    for d in picks:
        shutil.copy(d.image_path, AI2D_OUT / f"ai2d_{d.image_id}")
        md += [f"## `ai2d_{d.image_id}` ({d.category}, {len(d.graph.edges)} annotated arrows)", ""]
        for q in d.questions[:4]:
            md.append(f"- **{q.text}** Options: {' / '.join(q.choices)}. Correct: **{q.choices[q.answer]}**")
        md.append("")
    (AI2D_OUT / "README.md").write_text("\n".join(md))
    print(f"copied {len(picks)} AI2D test diagrams to {AI2D_OUT.relative_to(ROOT)} (git-ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
