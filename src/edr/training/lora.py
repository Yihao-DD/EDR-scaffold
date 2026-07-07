"""LoRA distillation trainer: SFT + optional KL anchor on replay rows.

Ported verbatim from the frozen round-1 trainer. Non-finite guards (skip
batch on nan loss / nan grad) are part of the frozen recipe.

CLI:
  python -m edr.training.lora --model-id ... --dataset ... --output-dir ... \
      --rank 16 --lr 5e-5 --epochs 3 --seed ... --kl-anchor-lambda 2.0 \
      [--base-adapter-dir <M1>]   # round 2: train A2 on top of merged M1
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from edr.io_utils import read_jsonl, write_json
from edr.modeling import (
    TARGET_MODULES,
    apply_lora,
    load_model_for_training,
    load_tokenizer,
    merge_base_adapter,
    render_training_text,
    render_user_prompt,
    set_seed,
)


def encode_row(tokenizer, row, max_length):
    prompt_text = render_user_prompt(tokenizer, row["input"])
    full_text = render_training_text(tokenizer, row["input"], row["output"])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": full_ids, "labels": labels}


def make_dataset(rows, tokenizer, max_length):
    import torch

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.rows = rows
            self.encoded = [encode_row(tokenizer, row, max_length) for row in rows]

        def __len__(self):
            return len(self.encoded)

        def __getitem__(self, index):
            item = dict(self.encoded[index])
            row = self.rows[index]
            item["is_replay"] = row.get("source") == "replay_base_success" or row.get("partition") == "replay"
            return item

    return SFTDataset()


def collate(features, pad_token_id):
    import torch

    max_len = max(len(item["input_ids"]) for item in features)
    input_ids = []
    labels = []
    attention_mask = []
    is_replay = []
    for item in features:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        is_replay.append(bool(item.get("is_replay", False)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "is_replay": torch.tensor(is_replay, dtype=torch.bool),
    }


def optimizer_and_scheduler(model, lr, total_steps, warmup_ratio):
    import torch

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    def lr_lambda(step):
        warmup = max(1, int(total_steps * warmup_ratio))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def kl_anchor_loss(model, batch, temperature):
    import torch
    import torch.nn.functional as F

    replay_rows = batch["is_replay"]
    if not replay_rows.any():
        return None
    model_inputs = {
        "input_ids": batch["input_ids"][replay_rows],
        "attention_mask": batch["attention_mask"][replay_rows],
    }
    labels = batch["labels"][replay_rows]
    outputs = model(**model_inputs)
    shifted_labels = labels[:, 1:]
    anchor_mask = shifted_labels.ne(-100)
    if not anchor_mask.any():
        return None
    with torch.no_grad():
        if not hasattr(model, "disable_adapter"):
            return None
        with model.disable_adapter():
            base_outputs = model(**model_inputs)
    adapted_logits = outputs.logits[:, :-1, :][anchor_mask].float() / temperature
    base_logits = base_outputs.logits[:, :-1, :][anchor_mask].float() / temperature
    if not torch.isfinite(adapted_logits).all() or not torch.isfinite(base_logits).all():
        return None
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    if not torch.isfinite(adapted_log_probs).all() or not torch.isfinite(base_log_probs).all():
        return None
    base_probs = base_log_probs.exp()
    token_kl = (base_probs * (base_log_probs - adapted_log_probs)).sum(dim=-1)
    if not torch.isfinite(token_kl).all():
        return None
    return token_kl.mean() * (temperature**2)


def train(args):
    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    rows = read_jsonl(args.dataset)
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    tokenizer = load_tokenizer(args.model_id)
    use_qlora = not args.no_qlora
    if args.base_adapter_dir and use_qlora:
        print(
            "[train:warn] --base-adapter-dir requires an in-memory merge before the new LoRA; "
            "loading bf16/full precision for the merge. Use a >=48GB GPU.",
            flush=True,
        )
        use_qlora = False
    model = load_model_for_training(args.model_id, qlora=use_qlora)
    model = merge_base_adapter(model, args.base_adapter_dir)
    model = apply_lora(model, args.rank, dropout=args.lora_dropout)
    dataset = make_dataset(rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    optimizer, scheduler = optimizer_and_scheduler(model, args.lr, total_steps, args.warmup_ratio)
    model.train()
    global_step = 0
    losses = []
    skipped_nonfinite_loss = 0
    skipped_nonfinite_grad = 0
    kl_anchor_batches = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    print(
        f"[train] grad_clip max_norm={args.max_grad_norm} "
        f"kl_anchor_lambda={args.kl_anchor_lambda} kl_anchor_temperature={args.kl_anchor_temperature}",
        flush=True,
    )
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            model_batch = {key: value for key, value in batch.items() if key != "is_replay"}
            loss = model(**model_batch).loss
            if args.kl_anchor_lambda > 0:
                anchor = kl_anchor_loss(model, batch, args.kl_anchor_temperature)
                if anchor is not None:
                    loss = loss + args.kl_anchor_lambda * anchor
                    kl_anchor_batches += 1
            if not torch.isfinite(loss.detach()):
                skipped_nonfinite_loss += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[train:warn] skipped non-finite loss epoch={epoch + 1}/{args.epochs} "
                    f"batch={batch_index}/{len(loader)} skipped={skipped_nonfinite_loss}",
                    flush=True,
                )
                continue
            loss = loss / args.grad_accum_steps
            loss.backward()
            if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skipped_nonfinite_grad += 1
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"[train:warn] skipped non-finite grad norm epoch={epoch + 1}/{args.epochs} "
                        f"step_candidate={global_step + 1}/{total_steps} skipped={skipped_nonfinite_grad}",
                        flush=True,
                    )
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                losses.append(float(loss.detach().cpu()) * args.grad_accum_steps)
                if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
                    print(
                        f"[train] epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                        f"loss={losses[-1]:.4f}",
                        flush=True,
                    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "dataset": args.dataset,
        "dataset_rows": len(rows),
        "model_id": args.model_id,
        "rank": args.rank,
        "lr": args.lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "qlora": use_qlora,
        "base_adapter_dir": args.base_adapter_dir,
        "kl_anchor_reference": "merged_base_adapter" if args.base_adapter_dir else "raw_base_disable_adapter",
        "target_modules": TARGET_MODULES,
        "kl_anchor_lambda": args.kl_anchor_lambda,
        "kl_anchor_temperature": args.kl_anchor_temperature,
        "max_grad_norm": args.max_grad_norm,
        "skipped_nonfinite_loss": skipped_nonfinite_loss,
        "skipped_nonfinite_grad": skipped_nonfinite_grad,
        "kl_anchor_batches": kl_anchor_batches,
        "train_seconds": time.time() - started,
        "losses": losses,
    }
    write_json(output_dir / "train_metadata.json", metadata)
    print(f"[train] saved {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-adapter-dir",
        default=None,
        help="Frozen adapter merged into base before adding the trainable LoRA (round 2: M1).",
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--kl-anchor-lambda",
        type=float,
        default=0.0,
        help="Optional KL(P_base || P_adapter) anchor on replay output tokens.",
    )
    parser.add_argument("--kl-anchor-temperature", type=float, default=1.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
