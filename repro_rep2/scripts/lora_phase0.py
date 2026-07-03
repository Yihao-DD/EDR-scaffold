import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.phase0_probe import (
    DEFAULT_SPLIT_SEED,
    assert_no_leakage,
    build_base_prompt,
    load_exp2,
    read_json,
    write_json,
)


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_training(model_id, qlora=True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if qlora:
        model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, rank, alpha=None, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank * 2,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def merge_base_adapter(model, adapter_dir):
    """Merge a frozen adapter into model weights before attaching a new LoRA.

    Round 2 trains A2 on top of M1 = M0 + A1. After this merge, the trainable
    adapter added by ``apply_lora`` is A2, and ``model.disable_adapter()`` inside
    the KL-anchor path evaluates the frozen M1 reference rather than raw M0.
    """

    from peft import PeftModel

    if not adapter_dir:
        return model
    print(f"[train] merging frozen base adapter into weights: {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model = model.merge_and_unload()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def render_user_prompt(tokenizer, input_text):
    messages = [{"role": "user", "content": input_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return input_text


def render_training_text(tokenizer, input_text, output_text):
    messages = [{"role": "user", "content": input_text}, {"role": "assistant", "content": output_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{input_text}\n{output_text}"


def encode_row(tokenizer, row, max_length):
    prompt_text = render_user_prompt(tokenizer, row["input"])
    full_text = render_training_text(tokenizer, row["input"], row["output"])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": full_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.encoded = [encode_row(tokenizer, row, max_length) for row in rows]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        item = dict(self.encoded[index])
        row = self.rows[index]
        item["is_replay"] = row.get("source") == "replay_base_success" or row.get("partition") == "replay"
        return item


def collate(features, pad_token_id):
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
    set_seed(args.seed)
    rows = read_jsonl(args.dataset)
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    tokenizer = load_tokenizer(args.model_id)
    use_qlora = not args.no_qlora
    if args.base_adapter_dir and use_qlora:
        print(
            "[train:warn] --base-adapter-dir requires an in-memory merge before A2; "
            "loading bf16/full precision for the merge. Use a >=49GB GPU.",
            flush=True,
        )
        use_qlora = False
    model = load_model_for_training(args.model_id, qlora=use_qlora)
    model = merge_base_adapter(model, args.base_adapter_dir)
    model = apply_lora(model, args.rank, dropout=args.lora_dropout)
    dataset = SFTDataset(rows, tokenizer, args.max_length)
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
            labels = batch["labels"]
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


def load_model_for_eval(model_id, adapter_dir=None, qlora=True):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def generate_one(model, tokenizer, prompt, max_new_tokens):
    text = render_user_prompt(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[:, inputs.input_ids.shape[-1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def evaluate_episode(evo, model, tokenizer, episode, prompt_template, max_new_tokens):
    raw = generate_one(model, tokenizer, build_base_prompt(episode, prompt_template), max_new_tokens)
    try:
        prediction = evo.parse_model_output(raw)
        parse_error = None
    except Exception as error:
        prediction = None
        parse_error = str(error)
    return {
        "episode_id": episode["episode_id"],
        "success": evo.call_success(prediction or {"name": "", "arguments": {}}, episode["ground_truth_call"]),
        "prediction": prediction,
        "raw_model_output": raw,
        "parse_error": parse_error,
        "failure_class": episode.get("failure_class", "unknown"),
    }


def rate(records):
    return sum(1 for row in records if row["success"]) / len(records) if records else 0.0


def evaluate(args):
    baseline = read_json(args.failures)
    evo = load_exp2(args.exp2_root)
    multiple_failures = [item for item in baseline["failures"] if item.get("split") == "multiple"]
    split = evo.deterministic_split(multiple_failures, seed=args.split_seed)
    inventory = read_json(args.s00_input)
    r_success_ids = set(inventory["regression_sets"]["r_success_eval"])
    heldout_ids = set(inventory["regression_sets"]["r_heldout"])
    val_episodes = split["validation"]
    r_success = [row for row in baseline["records"] if row["episode_id"] in r_success_ids]
    assert_no_leakage([], heldout_ids, r_success_ids)
    if args.eval_limit:
        val_episodes = val_episodes[: args.eval_limit]
        r_success = r_success[: args.eval_limit]
    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(args.model_id, adapter_dir=args.adapter_dir, qlora=not args.no_qlora)
    val_records = []
    for index, episode in enumerate(val_episodes, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(val_episodes):
            print(f"[eval:val] {index}/{len(val_episodes)}", flush=True)
        val_records.append(evaluate_episode(evo, model, tokenizer, episode, baseline["prompt_template"], args.max_new_tokens))
    success_records = []
    for index, episode in enumerate(r_success, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(r_success):
            print(f"[eval:r-success] {index}/{len(r_success)}", flush=True)
        success_records.append(evaluate_episode(evo, model, tokenizer, episode, baseline["prompt_template"], args.max_new_tokens))
    result = {
        "adapter_dir": args.adapter_dir,
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "val_n": len(val_records),
        "val_repair_rate": rate(val_records),
        "r_success_n": len(success_records),
        "r_success_rate": rate(success_records),
        "val_records": val_records,
        "r_success_records": success_records,
    }
    write_json(args.output, result)
    print(
        f"[eval] val_repair_rate={result['val_repair_rate']:.4f} "
        f"r_success_rate={result['r_success_rate']:.4f}",
        flush=True,
    )


def train_eval(args):
    train(args)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.adapter_dir = args.output_dir
    eval_args.output = args.eval_output
    evaluate(eval_args)


def build_parser():
    parser = argparse.ArgumentParser(description="EXP3 Phase 0 LoRA train/eval")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--s00-input", default="results/s00_inventory.json")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-qlora", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_train_args(train_parser)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--adapter-dir", default=None)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--eval-limit", type=int, default=None)
    eval_parser.add_argument("--progress-every", type=int, default=50)
    eval_parser.set_defaults(func=evaluate)

    train_eval_parser = subparsers.add_parser("train-eval")
    add_train_args(train_eval_parser)
    train_eval_parser.add_argument("--eval-output", required=True)
    train_eval_parser.add_argument("--eval-limit", type=int, default=None)
    train_eval_parser.add_argument("--progress-every", type=int, default=50)
    train_eval_parser.set_defaults(func=train_eval)
    return parser


def add_train_args(parser):
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-adapter-dir",
        default=None,
        help="Frozen adapter merged into base before adding the trainable LoRA. Round 2 uses this for M1 anchoring.",
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
        help="Optional KL(P_base || P_adapter) anchor on replay output tokens. Default 0 keeps the clean-grid SFT recipe unchanged.",
    )
    parser.add_argument("--kl-anchor-temperature", type=float, default=1.0)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
