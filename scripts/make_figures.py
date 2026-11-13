"""Draw docs/figures/architecture.png (used by docs/architecture.md and the Phase 2 report)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
AI, DET, UI, DATA = "#d9ead3", "#dbe7fb", "#fff2cc", "#eeeeee"


def box(ax, x, y, w, h, text, color, bold=False, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, p, q, text=None, dashed=False, off=0.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#333",
                                 linestyle="--" if dashed else "-"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + off, text, ha="center", va="bottom", fontsize=6.8, color="#444")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.25)
    ax.axis("off")
    # step 1: intake + extraction (shared by E3; the graph is shown and editable in the UI)
    box(ax, 0.1, 6.0, 1.9, 1.15, "Browser UI\n1 upload + extract\n2 question\n3 live pipelines", UI, True, 7.4)
    box(ax, 2.4, 6.0, 1.9, 1.15, "Image intake\nany format, EXIF,\nin memory only", DET, fs=7.4)
    box(ax, 4.7, 6.0, 2.5, 1.15, "Qwen2.5-VL-3B + LoRA (AI)\nimage → entities, boxes,\narrows (step 1)", AI, True, 7.4)
    box(ax, 7.6, 6.0, 2.2, 1.15, "DiagramGraph\nparsed, validated,\nuser-correctable", DET, fs=7.4)
    arrow(ax, (2.0, 6.57), (2.4, 6.57))
    arrow(ax, (4.3, 6.57), (4.7, 6.57))
    arrow(ax, (7.2, 6.57), (7.6, 6.57))
    # E3 lane: main path on top, fallback below
    ax.text(0.1, 4.8, "E3  graph-grounded\n(proposed)", fontsize=7.6, weight="bold", va="center")
    box(ax, 1.9, 4.85, 2.2, 0.85, "Question → query\nrules, else Qwen as\nparser only (AI)", AI, fs=7.0)
    box(ax, 4.5, 4.85, 1.9, 0.85, "NetworkX query\n(no AI) → answer\n+ supporting arrows", DET, True, 7.0)
    box(ax, 6.8, 4.85, 2.0, 0.85, "Qwen explains\nfrom cited arrows\n(AI, no new facts)", AI, fs=7.0)
    box(ax, 6.8, 3.8, 2.6, 0.8, "Fallback (AI): Qwen + graph,\nmust cite arrow ids →\ncitation verifier (no AI)", AI, fs=6.8)
    box(ax, 11.3, 3.8, 1.6, 1.9, "Answer +\nevidence\nor abstain", UI, True, 7.2)
    arrow(ax, (4.1, 5.27), (4.5, 5.27))
    arrow(ax, (6.4, 5.27), (6.8, 5.27))
    arrow(ax, (8.8, 5.27), (11.3, 5.27), "answered from the graph")
    arrow(ax, (5.45, 4.85), (6.8, 4.2), dashed=True)
    ax.text(5.2, 4.25, "query cannot\napply", fontsize=6.4, color="#444", ha="center", va="center")
    arrow(ax, (9.4, 4.2), (11.3, 4.2))
    arrow(ax, (8.0, 6.0), (3.0, 5.7), "graph", dashed=True, off=0.02)
    # E2 lane
    ax.text(0.1, 3.25, "E2  image → VLM\n(main baseline)", fontsize=7.6, weight="bold", va="center")
    box(ax, 1.9, 2.85, 4.5, 0.8, "Qwen2.5-VL-3B base (AI): image + question → answer", AI, fs=7.0)
    box(ax, 11.3, 2.85, 1.6, 0.8, "Answer\n(no evidence)", UI, fs=7.0)
    arrow(ax, (6.4, 3.25), (11.3, 3.25))
    # E1 lane
    ax.text(0.1, 2.05, "E1  OCR text only\n(baseline)", fontsize=7.6, weight="bold", va="center")
    box(ax, 1.9, 1.65, 2.2, 0.8, "PaddleOCR (AI)\ntext + positions", AI, fs=7.0)
    box(ax, 4.5, 1.65, 3.0, 0.8, "Qwen base, text only (AI)\nOCR lines + question → answer", AI, fs=7.0)
    box(ax, 11.3, 1.65, 1.6, 0.8, "Answer\n(no evidence)", UI, fs=7.0)
    arrow(ax, (4.1, 2.05), (4.5, 2.05))
    arrow(ax, (7.5, 2.05), (11.3, 2.05))
    # offline
    ax.plot([0.1, 12.9], [1.4, 1.4], color="#999", lw=0.8, ls=":")
    ax.text(0.12, 1.25, "offline", fontsize=7, color="#666", style="italic")
    box(ax, 0.1, 0.05, 2.6, 1.0, "AI2D (4,903 real diagrams)\nannotations → gold graphs\n+ human questions", DATA, fs=7.0)
    box(ax, 3.0, 0.05, 2.5, 1.0, "LoRA fine-tuning of\nextraction (train split)", AI, fs=7.0)
    box(ax, 5.8, 0.05, 3.6, 1.0, "Evaluation: AI2D test split + 304-question\nsynthetic pilot; E1 / E2 / E3 (zs, ft, gold);\nextraction F1, bootstrap CIs, McNemar", DATA, fs=6.8)
    box(ax, 9.7, 0.05, 3.2, 1.0, "results/*.json → Evaluation tab\n(AI2D: ids + metrics only)", UI, fs=7.0)
    arrow(ax, (2.7, 0.55), (3.0, 0.55))
    arrow(ax, (5.5, 0.55), (5.8, 0.55))
    arrow(ax, (9.4, 0.55), (9.7, 0.55))
    for i, (c, t) in enumerate([(AI, "learned model"), (DET, "deterministic code"), (UI, "front end / output"), (DATA, "data / evaluation")]):
        x = 10.1 + (i % 2) * 1.45
        y = 6.9 - (i // 2) * 0.35
        ax.add_patch(FancyBboxPatch((x, y), 0.18, 0.18, boxstyle="square,pad=0", fc=c, ec="#333", lw=0.6))
        ax.text(x + 0.25, y + 0.09, t, fontsize=6.4, va="center")
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "architecture.png", dpi=200)
    print("wrote", OUT / "architecture.png")


if __name__ == "__main__":
    main()
