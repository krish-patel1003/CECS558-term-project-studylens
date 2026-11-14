"""Multi-task LoRA for the Diagram Scene Graph (DSG) model calls, trained on the AI2D *train* split.

The Phase-2 AI2D analysis (docs/evaluation.md, section 2) showed that the base Qwen2.5-VL-3B
(a) boxes the label text instead of the part it names, or the wrong place, for about half of the
labels, and (b) finds only ~40% of drawn arrows. The old adapter (scripts/train_lora.py) could not fix
this: it trained a different output format at 448 px, mostly on diagrams without arrows, and its
"part" boxes were the label text boxes. This script trains exactly the calls the DSG makes:

* GROUND  image + label names  -> per label: ``point_2d`` (where its leader line ends on the part) or
          ``bbox_2d`` (the object/region the label names). Supervision: AI2D intraObjectLinkage
          connector tips and intraObjectLabel / RegionLabel blobs (~25k labels in ~3.4k diagrams).
* ARROWS  image + part names -> ``{"arrows": [{"from", "to"}]}`` by name (the DSG arrow prompt).
          Supervision: AI2D directed interObjectLinkage relations; diagrams without arrows teach
          ``{"arrows": []}``; synthetic box-and-arrow diagrams with RANDOM arrow directions and AI2D
          vocabulary teach reading arrowheads instead of the textbook direction.

Images use the same resizing as inference (max side 1024 by default) and coordinates are in the
model's resized grid, as Qwen2.5-VL expects. 150 train diagrams are held out for validation
(``--eval``): grounding hit rate (same rule as scripts/evaluate_grounding.py) and arrow P/R/F1.

    python scripts/train_dsg_lora.py --probe                       # 3 steps: memory / speed check
    python scripts/train_dsg_lora.py --run dsg_lora --epochs 2     # full run (resumable), A100: ~3-5 h
    python scripts/train_dsg_lora.py --run dsg_lora --eval         # validation metrics, base vs adapter

Writes checkpoints/<run>/ (git-ignored; derived from AI2D (non-commercial) and Qwen (research licence),
so it is not committed or redistributed).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from studylens.datasets import ai2d  # noqa: E402
from studylens.dsg import ARROW_PROMPT, GROUND_PROMPT_V2, parse_arrows, parse_grounding  # noqa: E402
from studylens.extractors.qwen_vl import prepare_image  # noqa: E402

CKPT = ROOT / "checkpoints"
SYNTH_DIR = ROOT / "external" / "train_synth"
MAX_LABELS = 24  # labels per grounding example (the DSG grounds at most 24 parts per call)
N_VAL = 150


# ----------------------------------------------------------------------------------------- data
def label_locations(image_id: str) -> list[tuple[str, dict]]:
    """[(label text as written, location)] from AI2D annotations; location {"point": (x, y)} or {"box": [...]}."""
    ann = json.loads((ai2d.AI2D_DIR / "annotations" / f"{image_id}.json").read_text())
    texts, blobs, arrows = ann.get("text", {}), ann.get("blobs", {}), ann.get("arrows", {})
    out, seen = [], set()
    for r in ann.get("relationships", {}).values():
        cat = r.get("category")
        o, d = r.get("origin", ""), r.get("destination", "")
        tid = o if o.startswith("T") else d if d.startswith("T") else None
        if tid is None or tid not in texts:
            continue
        label = " ".join(str(texts[tid].get("value", "")).split())
        key = ai2d.norm(label)
        if not key or key in seen:
            continue
        (x1, y1), (x2, y2) = texts[tid]["rectangle"]
        tc = ((x1 + x2) / 2, (y1 + y2) / 2)
        if cat == "intraObjectLinkage" and r.get("connector") in arrows:
            poly = arrows[r["connector"]]["polygon"]
            tip = max(poly, key=lambda p: math.dist(p, tc))
            out.append((label, {"point": (float(tip[0]), float(tip[1]))}))
            seen.add(key)
        elif cat in ("intraObjectLabel", "intraObjectRegionLabel"):
            bid = d if d.startswith("B") else o if o.startswith("B") else None
            if bid in blobs:
                xs = [p[0] for p in blobs[bid]["polygon"]]
                ys = [p[1] for p in blobs[bid]["polygon"]]
                out.append((label, {"box": [min(xs), min(ys), max(xs), max(ys)]}))
                seen.add(key)
    return out


def ground_target(items: list[tuple[str, dict]], scale) -> str:
    sx, sy = scale
    out = []
    for label, loc in items:
        if "point" in loc:
            x, y = loc["point"]
            out.append({"label": label, "point_2d": [round(x * sx), round(y * sy)]})
        else:
            b = loc["box"]
            out.append({"label": label, "bbox_2d": [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)]})
    return json.dumps(out, ensure_ascii=False)


def arrow_target(pairs: list[tuple[str, str]]) -> str:
    return json.dumps({"arrows": [{"from": a, "to": b} for a, b in pairs]}, ensure_ascii=False)


def make_synthetic(n: int, seed: int, vocab: list[str]) -> list[dict]:
    """Box-and-arrow diagrams with AI2D-train vocabulary and random arrow directions (cached PNGs)."""
    import make_synthetic as ms

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        _, pos, edges = ms.random_dag(rng, i)
        names = rng.sample(vocab, len(pos))
        ren = dict(zip(pos, names))
        pos2 = {ren[k]: v for k, v in pos.items()}
        e2 = [(ren[b], ren[a]) if rng.random() < 0.5 else (ren[a], ren[b]) for a, b in edges]
        path = SYNTH_DIR / f"synth_{seed}_{i:05d}.png"
        if not path.exists():
            ms.render("", pos2, e2, path)
        out.append({"task": "arrows", "image": str(path), "names": list(pos2), "pairs": e2, "id": path.name})
    return out


def build_examples(seed: int, n_synth: int, n_no_arrow: int, val_ids: set[str], exclude: set[str]) -> list[dict]:
    rng = random.Random(seed)
    train = sorted(ai2d.load("train"), key=lambda d: d.image_id)
    ex = []
    vocab = set()
    no_arrow = []
    for d in train:
        if d.image_id in val_ids or d.image_id in exclude:
            continue
        for n in d.graph.nodes:
            if 2 < len(n.label) <= 14 and n.label.replace(" ", "").isalpha():
                vocab.add(n.label.strip().capitalize())
        locs = label_locations(d.image_id)
        rng.shuffle(locs)
        for i in range(0, len(locs), MAX_LABELS):
            chunk = locs[i : i + MAX_LABELS]
            ex.append({"task": "ground", "image": str(d.image_path), "items": chunk, "id": d.image_id})
        names = [n.label for n in d.graph.nodes]
        if d.graph.edges and len(names) <= 120:
            pairs = list(dict.fromkeys((d.graph.label_of(e.source), d.graph.label_of(e.target)) for e in d.graph.edges))
            ex.append({"task": "arrows", "image": str(d.image_path), "names": names, "pairs": pairs, "id": d.image_id})
        elif names and len(names) <= 60:
            no_arrow.append({"task": "arrows", "image": str(d.image_path), "names": names, "pairs": [], "id": d.image_id})
    rng.shuffle(no_arrow)
    ex += no_arrow[:n_no_arrow]
    ex += make_synthetic(n_synth, seed, sorted(vocab))
    rng.shuffle(ex)
    return ex


def val_diagrams(seed: int) -> list[ai2d.Diagram]:
    train = sorted(ai2d.load("train"), key=lambda d: d.image_id)
    random.Random(seed + 1).shuffle(train)
    return train[:N_VAL]


# ------------------------------------------------------------------------------------- encoding
def prompt_of(e: dict, rng: random.Random) -> str:
    if e["task"] == "ground":
        return GROUND_PROMPT_V2.format(labels=json.dumps([lab for lab, _ in e["items"]], ensure_ascii=False))
    names = list(e["names"])
    rng.shuffle(names)  # the DSG lists parts in reading order; shuffling avoids learning list order
    return ARROW_PROMPT.format(names=json.dumps(names, ensure_ascii=False))


def encode(processor, e: dict, max_side: int, rng: random.Random):
    raw = Image.open(e["image"])
    img = prepare_image(raw, max_side)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_of(e, rng)}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    probe = processor(text=[prompt], images=[img], return_tensors="pt")
    _, gh, gw = probe["image_grid_thw"][0].tolist()
    scale = (gw * 14 / raw.size[0], gh * 14 / raw.size[1])
    tgt = (ground_target(e["items"], scale) if e["task"] == "ground" else arrow_target(e["pairs"])) + "<|im_end|>"
    full = processor(text=[prompt + tgt], images=[img], return_tensors="pt")
    full["n_target"] = full["input_ids"].shape[1] - probe["input_ids"].shape[1]
    return full


def loss_on_targets(model, batch) -> torch.Tensor:
    n = int(batch.pop("n_target"))
    out = model(**batch, logits_to_keep=n + 1, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    tgt = batch["input_ids"][:, -n:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))


# ----------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, processor, dev, max_side: int, seed: int, use_adapter: bool, n: int, prompt_v: int) -> dict:
    """Grounding hit rate and arrow P/R/F1 on held-out train diagrams (base: v1 grounding prompt)."""
    import contextlib

    from studylens.dsg import GROUND_PROMPT

    ctx = contextlib.nullcontext() if use_adapter or not hasattr(model, "disable_adapter") else model.disable_adapter()
    hits = total = 0
    tp = fp = fn = 0
    with ctx:
        for d in val_diagrams(seed)[:n]:
            raw_img = Image.open(d.image_path)
            img = prepare_image(raw_img, max_side)
            size = raw_img.size

            def gen(text, max_new):
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                p = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = processor(text=[p], images=[img], return_tensors="pt").to(dev)
                _, gh, gw = inp["image_grid_thw"][0].tolist()
                o = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
                out = processor.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                return out, (gw * 14 / size[0], gh * 14 / size[1])

            locs = label_locations(d.image_id)[:MAX_LABELS]
            if locs:
                labels = [lab for lab, _ in locs]
                tmpl = GROUND_PROMPT_V2 if prompt_v == 2 else GROUND_PROMPT
                out, scale = gen(tmpl.format(labels=json.dumps(labels, ensure_ascii=False)), 30 * len(labels) + 40)
                boxes = parse_grounding(out, scale, size)
                gold = ai2d.gold_locations(json.loads((ai2d.AI2D_DIR / "annotations" / f"{d.image_id}.json").read_text()))
                for lab in labels:
                    k = ai2d.norm(lab)
                    b = boxes.get(lab.strip().lower())
                    total += 1
                    hits += bool(b) and k in gold and ai2d.location_hit(b, gold[k], size)
            if d.graph.edges:
                names = [n.label for n in d.graph.nodes][:120]
                out, _ = gen(ARROW_PROMPT.format(names=json.dumps(names, ensure_ascii=False)), min(1536, 18 * len(names) + 80))
                parts = [{"label": x} for x in names]
                pred, _ = parse_arrows(out, parts)
                pred = {(names[a], names[b]) for a, b in pred}
                gold_e = {(d.graph.label_of(e.source), d.graph.label_of(e.target)) for e in d.graph.edges}
                tp += len(pred & gold_e)
                fp += len(pred - gold_e)
                fn += len(gold_e - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"grounding_hit_rate": round(hits / total, 3) if total else None, "labels": total,
            "arrow_precision": round(p, 3), "arrow_recall": round(r, 3),
            "arrow_f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "gold_arrows": tp + fn}


# ----------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dsg_lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--image-side", type=int, default=1024)
    ap.add_argument("--n-synth", type=int, default=1500)
    ap.add_argument("--n-no-arrow", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=458)
    ap.add_argument("--exclude", default="", help="file with AI2D image ids to keep out of training")
    ap.add_argument("--probe", action="store_true", help="3 optimizer steps, report memory/speed, do not save")
    ap.add_argument("--eval", action="store_true", help="validation metrics for the base model and the adapter")
    ap.add_argument("--n-eval", type=int, default=60)
    args = ap.parse_args(argv)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from studylens.config import settings

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(settings.qwen_model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings.qwen_model_id, dtype=dtype, attn_implementation="sdpa").to(dev)
    out_dir = CKPT / args.run

    if args.eval:
        res = {"base_v1_prompt": evaluate(model, processor, dev, args.image_side, args.seed, False, args.n_eval, 1)}
        print(json.dumps(res), flush=True)
        if (out_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, out_dir)
            res["adapter"] = evaluate(model, processor, dev, args.image_side, args.seed, True, args.n_eval, 2)
        print(json.dumps(res, indent=1), flush=True)
        (out_dir / "val_metrics.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "val_metrics.json").write_text(json.dumps(res, indent=1))
        return 0

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    resume = (out_dir / "adapter_config.json").exists() and not args.probe
    if resume:
        model = PeftModel.from_pretrained(model, out_dir, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
            # language model attention + MLP; the vision encoder stays frozen
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model.print_trainable_parameters()

    val_ids = {d.image_id for d in val_diagrams(args.seed)}
    exclude = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    examples = build_examples(args.seed, args.n_synth, args.n_no_arrow, val_ids, exclude)
    counts = {t: sum(e["task"] == t for e in examples) for t in ("ground", "arrows")}
    total_steps = max(1, int(len(examples) * args.epochs) // args.grad_accum)
    if args.probe:
        total_steps = 3
    print(f"examples {len(examples)} {counts}; optimizer steps {total_steps}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    state_path = out_dir / "trainer_state.json"
    step = 0
    if resume and state_path.exists():
        st = json.loads(state_path.read_text())
        step = st["step"]
        if (out_dir / "optimizer.pt").exists():
            opt.load_state_dict(torch.load(out_dir / "optimizer.pt", map_location=dev))
    sched = lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))  # noqa: E731
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "train_log.jsonl").open("a")
    if not resume:
        (out_dir / "config.json").write_text(json.dumps({**vars(args), "examples": len(examples), "counts": counts,
                                                          "total_steps": total_steps, "device": dev, "dtype": str(dtype)}, indent=1))
    rng = random.Random(args.seed)
    model.train()
    t0 = time.perf_counter()
    i = step * args.grad_accum
    start_step = step
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        acc, ntok = 0.0, 0
        for _ in range(args.grad_accum):
            e = examples[i % len(examples)]
            i += 1
            try:
                batch = encode(processor, e, args.image_side, rng)
            except Exception as exc:  # noqa: BLE001 - skip unreadable images
                print(f"skip {e['id']}: {exc}", flush=True)
                continue
            n_t = int(batch["n_target"])
            batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss = loss_on_targets(model, batch) / args.grad_accum
            loss.backward()
            acc += loss.item()
            ntok += n_t
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        el = time.perf_counter() - t0
        rec = {"step": step, "loss": round(acc, 4), "lr": opt.param_groups[0]["lr"], "elapsed_s": round(el, 1), "target_tokens": ntok}
        if dev == "cuda":
            rec["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or args.probe:
            print(rec, f"~{el / max(1, step - start_step) * (total_steps - step) / 3600:.1f} h left", flush=True)
        if not args.probe and (step % args.save_every == 0 or step == total_steps):
            model.save_pretrained(out_dir)
            torch.save(opt.state_dict(), out_dir / "optimizer.pt")
            state_path.write_text(json.dumps({"step": step, "total_steps": total_steps}))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
