#!/usr/bin/env python3
"""A11 placebo-patch control: token-scrambled patches, forward-only.

Spec (PROJECT_MASTER_PLAN Step 1.6 / metric dictionary `placebo_repair`,
operationalized by the v1.27 ruling):
- Each accepted H1 patch is scrambled at the token level: the patch's own
  tokens are shuffled, so length and vocabulary distribution are preserved
  while information is destroyed.
- THREE archived scramble seeds (configs/launch.json phase1.a11.scramble_seeds).
  An episode counts as placebo-rescuable if ANY seed's scramble passes V —
  the union is deliberately generous to the placebo, i.e. conservative for us.
- Injection uses the EXACT scaffold-on path: the same
  relevant_patches_for_episode selection and the same build_prompt template —
  only patch_text is scrambled.
- Pure forward evaluation (no training) on three surfaces:
  train-share failures, validation failures (156), and D_heldout (158).
  MAIN adjudication surface = heldout 158 (same exam as every arm);
  train/val are reference surfaces.
- Mechanical rule (v1.27): Wilson 95% lower bound of the union rate > 0 →
  "significantly above zero" → deduction fires (net teaching gain =
  repair(A2) − placebo_repair, point estimate, CI attached). Lower bound ≤ 0
  → "≈0" holds and the perturbation explanation is excluded.

Subcommands:
  scramble   CPU. Writes the frozen scrambled patch sets (all seeds) + hashes.
  eval       GPU. Forward evaluation: three surfaces x three scramble seeds.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from phase1.common import (
    add_common_args,
    load_config,
    load_inputs,
    load_probe_module,
    read_json,
    resolve_model_id,
    sha256_text,
    stable_int,
    write_json,
    REPRO_ROOT,
)


def scramble_text(tokenizer, text, rng):
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    shuffled = list(token_ids)
    rng.shuffle(shuffled)
    return tokenizer.decode(shuffled, skip_special_tokens=True)


def scramble(args, config):
    baseline, exp2_result, split, patches, evo = load_inputs(config["split_seed"])
    seeds = config["phase1"]["a11"]["scramble_seeds"]

    if args.dry_run:
        write_json(
            Path(args.output_dir) / "a11" / "a11_scramble_plan.json",
            {"probe": "a11_scramble_dry_run", "n_patches": len(patches), "scramble_seeds": seeds},
        )
        print(f"[a11-scramble:dry-run] patches={len(patches)} seeds={seeds}")
        return

    from scripts.lora_phase0 import load_tokenizer

    tokenizer = load_tokenizer(resolve_model_id(config))
    scrambled = []
    for patch in patches:
        variants = {}
        for seed in seeds:
            rng = random.Random(stable_int(seed, patch.patch_id, "a11_scramble"))
            text = scramble_text(tokenizer, patch.patch_text, rng)
            variants[str(seed)] = {"text": text, "sha256": sha256_text(text)}
        scrambled.append(
            {
                "patch_id": patch.patch_id,
                "original_sha256": sha256_text(patch.patch_text),
                "original_token_count": patch.token_count,
                "variants": variants,
            }
        )
    payload = {
        "probe": "a11_scrambled_patches",
        "scramble_seeds": seeds,
        "tokenizer_model": resolve_model_id(config),
        "n_patches": len(scrambled),
        "patches": scrambled,
    }
    output = Path(args.output_dir) / "a11" / "a11_scrambled_patches.json"
    write_json(output, payload)
    print(f"[a11-scramble] patches={len(scrambled)} seeds={seeds} -> {output}")


def load_surfaces(config, baseline, split):
    baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
    heldout_partition = read_json(REPRO_ROOT / config["eval"]["heldout_partition"])
    heldout = [baseline_by_id[row["episode_id"]] for row in heldout_partition["episodes"]]
    return {
        "train_failures": split["train"],
        "val_failures": split["validation"],
        "heldout": heldout,
    }


def evaluate(args, config):
    baseline, exp2_result, split, patches, evo = load_inputs(config["split_seed"])
    probe = load_probe_module()
    frozen = read_json(Path(args.output_dir) / "a11" / "a11_scrambled_patches.json")
    seeds = [str(seed) for seed in frozen["scramble_seeds"]]
    variants_by_id = {row["patch_id"]: row["variants"] for row in frozen["patches"]}

    # Same selection, scrambled text: rebuild PatchCandidate objects per seed
    # with the scrambled patch_text so relevance matching stays identical
    # (it uses metadata fields, never the text).
    import dataclasses

    placebo_sets = {
        seed: [dataclasses.replace(patch, patch_text=variants_by_id[patch.patch_id][seed]["text"]) for patch in patches]
        for seed in seeds
    }

    surfaces = load_surfaces(config, baseline, split)
    if args.limit:
        surfaces = {name: rows[: args.limit] for name, rows in surfaces.items()}

    if args.dry_run:
        total = sum(len(rows) for rows in surfaces.values()) * len(seeds)
        write_json(
            Path(args.output_dir) / "a11" / "a11_eval_plan.json",
            {
                "probe": "a11_eval_dry_run",
                "surfaces": {name: len(rows) for name, rows in surfaces.items()},
                "scramble_seeds": seeds,
                "forwards_total": total,
            },
        )
        print(f"[a11-eval:dry-run] forwards={total} (3 seeds x 3 surfaces)")
        return

    from scripts.lora_phase0 import generate_one, load_model_for_eval, load_tokenizer

    model_id = resolve_model_id(config)
    tokenizer = load_tokenizer(model_id)
    model = load_model_for_eval(model_id, adapter_dir=None, qlora=not args.no_qlora)

    max_new_tokens = config["eval"]["max_new_tokens"]
    results = {}
    for name, episodes in surfaces.items():
        records = []
        for index, episode in enumerate(episodes, start=1):
            if index == 1 or index % 25 == 0 or index == len(episodes):
                print(f"[a11-eval {name}] {index}/{len(episodes)}", flush=True)
            per_seed = {}
            n_patches = 0
            for seed in seeds:
                relevant = evo.relevant_patches_for_episode(placebo_sets[seed], episode)
                n_patches = len(relevant)
                prompt = evo.build_prompt(episode, relevant)
                raw = generate_one(model, tokenizer, prompt, max_new_tokens)
                prediction, parse_error = probe.parse_prediction(evo, raw)
                success = probe.success_for_prediction(evo, prediction, episode)
                per_seed[seed] = {
                    "success": success,
                    "prediction": prediction,
                    "raw_model_output": raw,
                    "parse_error": parse_error,
                }
            records.append(
                {
                    "episode_id": episode["episode_id"],
                    "n_patches": n_patches,
                    "union_success": any(item["success"] for item in per_seed.values()),
                    "per_seed": per_seed,
                }
            )
        union = sum(1 for r in records if r["union_success"])
        with_patch = [r for r in records if r["n_patches"] > 0]
        results[name] = {
            "n": len(records),
            "placebo_repair": union,
            "placebo_repair_rate": union / len(records) if records else 0.0,
            "aggregation": "union over scramble seeds (v1.27: generous to placebo)",
            "per_seed_repair": {seed: sum(1 for r in records if r["per_seed"][seed]["success"]) for seed in seeds},
            "episodes_with_relevant_patch": len(with_patch),
            "placebo_repair_with_patch": sum(1 for r in with_patch if r["union_success"]),
            "records": records,
        }
        write_json(Path(args.output_dir) / "a11" / f"a11_eval_{name}.json", results[name])

    summary = {
        "probe": "a11_placebo_eval",
        "model_id": model_id,
        "scramble_seeds": frozen["scramble_seeds"],
        "main_surface": "heldout",
        "surfaces": {
            name: {key: value for key, value in item.items() if key != "records"}
            for name, item in results.items()
        },
    }
    write_json(Path(args.output_dir) / "a11" / "a11_summary.json", summary)
    for name, item in results.items():
        print(f"[a11-eval] {name}: union placebo_repair={item['placebo_repair']}/{item['n']} per_seed={item['per_seed_repair']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scramble_parser = subparsers.add_parser("scramble")
    add_common_args(scramble_parser)

    eval_parser = subparsers.add_parser("eval")
    add_common_args(eval_parser)
    eval_parser.add_argument("--no-qlora", action="store_true")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "scramble":
        scramble(args, config)
    else:
        evaluate(args, config)


if __name__ == "__main__":
    main()
