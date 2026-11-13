"""LoRA fine-tuning of Qwen2.5-VL-3B-Instruct for diagram -> graph extraction on AI2D (train split).

Input : AI2D diagram image + the SAME extraction prompt the app uses (EXTRACTION_PROMPT)
Target: compact JSON {"entities":[{"label","bbox_2d"}],"arrows":[{"from","to"}]} built from the
        AI2D gold annotations (studylens/datasets/ai2d.py). Loss only on the target tokens.

    python scripts/train_lora.py --steps 3 --probe            # memory / speed check
    python scripts/train_lora.py --run ai2d_lora_r16          # full run (resumable)

Writes checkpoints/<run>/ (git-ignored): adapter weights, train_log.jsonl, config.json.
The adapter is derived from AI2D (non-commercial licence) and Qwen (research licence) and is
therefore not committed or redistributed.
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

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from studylens.config import settings  # noqa: E402
from studylens.datasets import ai2d  # noqa: E402
from studylens.extractors.qwen_vl import EXTRACTION_PROMPT, prepare_image  # noqa: E402

CKPT = ROOT / "checkpoints"
MAX_NODES = 30  # skip very dense diagrams (long targets) during training


def target_json(d: ai2d.Diagram, scale: tuple[float, float]) -> str:
    sx, sy = scale
    ents = []
    for n in d.graph.nodes:
        b = n.bbox
        ents.append({"label": n.label, "bbox_2d": [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)]})
    arrows = [{"from": d.graph.label_of(e.source), "to": d.graph.label_of(e.target)} for e in d.graph.edges]
    return json.dumps({"entities": ents, "arrows": arrows}, separators=(",", ":"), ensure_ascii=False)


def select_examples(seed: int, n_no_edge: int, n_val: int):
    train = [d for d in ai2d.load("train") if d.graph.nodes and len(d.graph.nodes) <= MAX_NODES]
    rng = random.Random(seed)
    with_e = [d for d in train if d.graph.edges]
    no_e = [d for d in train if not d.graph.edges]
    rng.shuffle(with_e)
    rng.shuffle(no_e)
    val = with_e[: n_val // 2] + no_e[: n_val - n_val // 2]
    tr = with_e[n_val // 2 :] + no_e[n_val - n_val // 2 : n_val - n_val // 2 + n_no_edge]
    rng.shuffle(tr)
    return tr, val


def encode(processor, d: ai2d.Diagram, max_side: int):
    img = prepare_image(Image.open(d.image_path), max_side)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": EXTRACTION_PROMPT}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    probe = processor(text=[prompt], images=[img], return_tensors="pt")
    _, gh, gw = probe["image_grid_thw"][0].tolist()
    orig_w, orig_h = Image.open(d.image_path).size
    scale = (gw * 14 / orig_w, gh * 14 / orig_h)
    tgt = target_json(d, scale) + "<|im_end|>"
    full = processor(text=[prompt + tgt], images=[img], return_tensors="pt")
    full["n_target"] = full["input_ids"].shape[1] - probe["input_ids"].shape[1]
    return full


def loss_on_targets(model, batch) -> torch.Tensor:
    """Cross-entropy on the target tokens only. Computing logits for just those positions
    (logits_to_keep) avoids materialising a [seq x 151k-vocab] fp32 tensor: the main memory saver."""
    n = int(batch.pop("n_target"))
    out = model(**batch, logits_to_keep=n + 1, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    tgt = batch["input_ids"][:, -n:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ai2d_lora_r16")
    ap.add_argument("--steps", type=int, default=600, help="optimizer steps")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--image-side", type=int, default=448, help="max image side during training")
    ap.add_argument("--targets", default="attn", choices=["attn", "attn_mlp"])
    ap.add_argument("--n-no-edge", type=int, default=600, help="train diagrams without arrows to include")
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=458)
    ap.add_argument("--probe", action="store_true", help="few steps, report memory/speed, do not save")
    args = ap.parse_args(argv)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(settings.qwen_model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings.qwen_model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    out_dir = CKPT / args.run
    resume = (out_dir / "adapter_config.json").exists() and not args.probe
    if resume:
        model = PeftModel.from_pretrained(model, out_dir, is_trainable=True)
    else:
        cfg = LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
            # language-model projections only; the vision encoder stays frozen
            target_modules=(r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)" if args.targets == "attn"
                            else r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"),
        )
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    train, val = select_examples(args.seed, args.n_no_edge, args.n_val)
    print(f"train examples {len(train)} (with arrows {sum(bool(d.graph.edges) for d in train)}), val {len(val)}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    state_path = out_dir / "trainer_state.json"
    start_step = 0
    if resume and state_path.exists():
        st = json.loads(state_path.read_text())
        start_step = st["step"]
        if (out_dir / "optimizer.pt").exists():
            opt.load_state_dict(torch.load(out_dir / "optimizer.pt", map_location=dev))
        print("resuming at step", start_step)

    def lr_at(step):
        warm = max(1, int(0.05 * args.steps))
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for d in val:
                b = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in encode(processor, d, args.image_side).items()}
                tot += loss_on_targets(model, b).item()
                if dev == "mps":
                    torch.mps.empty_cache()
        model.train()
        return tot / len(val)

    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train_log.jsonl", "a") if not args.probe else None
    if not args.probe:
        (out_dir / "config.json").write_text(json.dumps({**vars(args), "model": settings.qwen_model_id,
                                                         "train_image_side": args.image_side,
                                                         "n_train": len(train), "n_val": len(val)}, indent=1))
    model.train()
    order = [train[i % len(train)] for i in range(args.steps * args.grad_accum)]
    t0 = time.time()
    total_steps = 3 if args.probe else args.steps
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        acc = 0.0
        ntok = 0
        for k in range(args.grad_accum):
            d = order[step * args.grad_accum + k]
            b = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in encode(processor, d, args.image_side).items()}
            ntok += int(b["input_ids"].shape[1])
            loss = loss_on_targets(model, b) / args.grad_accum
            loss.backward()
            acc += loss.item()
            del b, loss
            if dev == "mps":
                torch.mps.empty_cache()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        el = time.time() - t0
        rec = {"step": step + 1, "loss": round(acc, 4), "lr": lr_at(step), "elapsed_s": round(el, 1), "tokens": ntok}
        if dev == "mps":
            rec["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        print(json.dumps(rec), flush=True)
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if not args.probe and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            vl = val_loss()
            print(json.dumps({"step": step + 1, "val_loss": round(vl, 4)}), flush=True)
            log.write(json.dumps({"step": step + 1, "val_loss": round(vl, 4)}) + "\n")
            log.flush()
            model.save_pretrained(out_dir)
            torch.save(opt.state_dict(), out_dir / "optimizer.pt")
            state_path.write_text(json.dumps({"step": step + 1}))
    if args.probe:
        print(f"probe: {total_steps} steps x {args.grad_accum} examples in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
