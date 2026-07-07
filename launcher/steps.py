#!/usr/bin/env python3
"""Step DAG definitions for phase 1 and phase 2.

Every step is a plain subprocess with an explicit working directory,
explicit inputs, and explicit `produces` files (the resume markers).
Recipes are read from configs/launch.json and are preregistered — the
launcher never invents hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Step:
    id: str
    argv: list
    cwd: str = "."
    gpu: bool = False
    needs: list = field(default_factory=list)
    produces: list = field(default_factory=list)
    note: str = ""


def _recipe(config):
    r = config["recipe_locked"]
    return r["rank"], r["lr"], r["epochs"], r["kl_anchor_lambda"]


def _train_step(step_id, config, dataset_rel_to_repro, adapter_dir_rel_to_repro, seed, kl_lambda, needs):
    rank, lr, epochs, _ = _recipe(config)
    return Step(
        id=step_id,
        cwd="repro_rep2",
        gpu=True,
        needs=list(needs),
        argv=[
            "python3",
            "scripts/lora_phase0.py",
            "--model-id",
            config["model_id"],
            "train",
            "--dataset",
            dataset_rel_to_repro,
            "--output-dir",
            adapter_dir_rel_to_repro,
            "--rank",
            str(rank),
            "--lr",
            str(lr),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--kl-anchor-lambda",
            str(kl_lambda),
        ],
        produces=[_repro_rel(adapter_dir_rel_to_repro) + "/train_metadata.json"],
    )


def _heldout_step(step_id, config, arm_label, adapter_dir_rel_to_repro, dataset_rel_to_repro, seed, output_rel_to_repro, needs):
    return Step(
        id=step_id,
        cwd="repro_rep2",
        gpu=True,
        needs=list(needs),
        argv=[
            "python3",
            "scripts/s07_heldout_eval.py",
            "--root",
            ".",
            "--model-id",
            config["model_id"],
            "--adapter-dir",
            adapter_dir_rel_to_repro,
            "--arm",
            "main",
            "--config",
            arm_label,
            "--seed",
            str(seed),
            "--run-id",
            f"{arm_label}_seed{seed}",
            "--train-dataset",
            dataset_rel_to_repro,
            "--heldout-partition",
            config["eval"]["heldout_partition"],
            "--failures",
            "../EDG-EXP1/results/a2_failures.json",
            "--exp2-root",
            "../EDG-EXP2-struct",
            "--output",
            output_rel_to_repro,
        ],
        produces=[_repro_rel(output_rel_to_repro)],
    )


def _sibling_step(step_id, config, arm_label, adapter_dir_rel_to_repro, seed, output_rel_to_repro, needs):
    return Step(
        id=step_id,
        cwd="repro_rep2",
        gpu=True,
        needs=list(needs),
        argv=[
            "python3",
            "scripts/s07_sibling_arena_eval.py",
            "--root",
            ".",
            "--model-id",
            config["model_id"],
            "--adapter-dir",
            adapter_dir_rel_to_repro,
            "--arm",
            "main",
            "--config",
            arm_label,
            "--seed",
            str(seed),
            "--run-id",
            f"{arm_label}_seed{seed}",
            "--arena",
            config["eval"]["sibling_arena"],
            "--failures",
            "../EDG-EXP1/results/a2_failures.json",
            "--output",
            output_rel_to_repro,
        ],
        produces=[_repro_rel(output_rel_to_repro)],
    )


def _repro_rel(path_rel_to_repro):
    """Map a path expressed relative to repro_rep2 to a repo-root-relative path."""

    if path_rel_to_repro.startswith("../"):
        return path_rel_to_repro[3:]
    return "repro_rep2/" + path_rel_to_repro


def _train_eval_block(steps, config, arm, dataset_repo_rel, kl_lambda, seeds, needs_build):
    """train + heldout + sibling for each seed of one training arm."""

    label = f"{arm}_rep2lock"
    dataset_repro_rel = "../" + dataset_repo_rel
    for seed in seeds:
        adapter = f"../phase1_outputs/{arm.split('_')[0]}/adapters/{arm}_seed{seed}"
        train_id = f"p1.{arm}.train.s{seed}"
        steps.append(_train_step(train_id, config, dataset_repro_rel, adapter, seed, kl_lambda, needs_build))
        base = arm.split("_")[0]
        heldout_out = f"../phase1_outputs/{base}/eval/{arm}_seed{seed}_heldout.json"
        sibling_out = f"../phase1_outputs/{base}/eval/{arm}_seed{seed}_sibling.json"
        steps.append(_heldout_step(f"p1.{arm}.heldout.s{seed}", config, label, adapter, dataset_repro_rel, seed, heldout_out, [train_id]))
        steps.append(_sibling_step(f"p1.{arm}.sibling.s{seed}", config, label, adapter, seed, sibling_out, [train_id]))


def phase1_steps(config):
    steps = []
    seeds = config["phase1"]["seeds"]
    a5 = config["phase1"]["a5"]

    # --- A5: resample -> build -> train/eval x seeds ---------------------
    shard_ids = []
    for shard in range(a5["shard_count"]):
        step_id = f"p1.a5.resample.shard{shard}"
        shard_ids.append(step_id)
        steps.append(
            Step(
                id=step_id,
                gpu=True,
                argv=[
                    "python3",
                    "-m",
                    "phase1.a5_resample_teacher",
                    "--shard-index",
                    str(shard),
                    "--shard-count",
                    str(a5["shard_count"]),
                ],
                produces=[f"phase1_outputs/a5/a5_teacher_all_shard_{shard}.json"],
            )
        )
    steps.append(
        Step(
            id="p1.a5.build",
            argv=["python3", "-m", "phase1.a5_build_unverified"],
            needs=shard_ids,
            produces=["phase1_outputs/a5/distill_a5_unverified.jsonl", "phase1_outputs/a5/a5_dataset_summary.json"],
        )
    )
    _train_eval_block(steps, config, "a5", "phase1_outputs/a5/distill_a5_unverified.jsonl", config["recipe_locked"]["kl_anchor_lambda"], seeds, ["p1.a5.build"])

    # --- A7 / A8 datasets -------------------------------------------------
    steps.append(
        Step(
            id="p1.a7a8.build",
            argv=["python3", "-m", "phase1.a7_a8_build"],
            produces=["phase1_outputs/a7/distill_a7_noreplay.jsonl"]
            + [f"phase1_outputs/a8/distill_a8_{int(f * 100)}pct.jsonl" for f in config["phase1"]["a8"]["fractions"]],
        )
    )
    # A7 = rep2 minus replay minus KL (v1.26): kl lambda forced to 0.
    _train_eval_block(steps, config, "a7", "phase1_outputs/a7/distill_a7_noreplay.jsonl", 0.0, seeds, ["p1.a7a8.build"])
    for fraction in config["phase1"]["a8"]["fractions"]:
        tag = f"{int(fraction * 100)}pct"
        _train_eval_block(
            steps,
            config,
            f"a8_{tag}",
            f"phase1_outputs/a8/distill_a8_{tag}.jsonl",
            config["recipe_locked"]["kl_anchor_lambda"],
            seeds,
            ["p1.a7a8.build"],
        )

    # --- A6: frozen retrieval -> forward eval -----------------------------
    steps.append(
        Step(
            id="p1.a6.index",
            gpu=True,  # dense encoding benefits from a GPU; BM25 alone is CPU
            argv=["python3", "-m", "phase1.a6_retrieval", "build-index"],
            produces=["phase1_outputs/a6/a6_frozen_retrieval.json"],
        )
    )
    steps.append(
        Step(
            id="p1.a6.eval",
            gpu=True,
            needs=["p1.a6.index"],
            argv=["python3", "-m", "phase1.a6_retrieval", "eval"],
            produces=["phase1_outputs/a6/a6_summary.json"],
        )
    )

    # --- A11: scramble -> forward eval ------------------------------------
    steps.append(
        Step(
            id="p1.a11.scramble",
            argv=["python3", "-m", "phase1.a11_placebo", "scramble"],
            produces=["phase1_outputs/a11/a11_scrambled_patches.json"],
        )
    )
    steps.append(
        Step(
            id="p1.a11.eval",
            gpu=True,
            needs=["p1.a11.scramble"],
            argv=["python3", "-m", "phase1.a11_placebo", "eval"],
            produces=["phase1_outputs/a11/a11_summary.json"],
        )
    )

    # --- Step 1.5 material + Gate 1 tally ---------------------------------
    steps.append(
        Step(
            id="p1.step15",
            argv=["python3", "-m", "phase1.step15_sample_nn"],
            produces=["phase1_outputs/step15/step15_nn_package.json"],
        )
    )
    eval_ids = [step.id for step in steps if ".heldout." in step.id or ".sibling." in step.id or step.id in ("p1.a6.eval", "p1.a11.eval")]
    steps.append(
        Step(
            id="p1.gate1",
            argv=["python3", "-m", "phase1.gate1_report"],
            needs=eval_ids,
            produces=["phase1_outputs/gate1_report.md", "phase1_outputs/gate1_report.json"],
            note="Mechanical tally only; human sign-off required.",
        )
    )
    return steps


def phase2_steps(config):
    model_id = config["model_id"]
    seeds = config["phase2"]["seeds"]
    manifest = config["phase2"]["m1_manifest"]
    import json

    m1 = json.loads((REPO_ROOT / manifest).read_text(encoding="utf-8"))["m1_designated"]
    m1_adapter = str(Path(m1["adapter"]["path"]).parent)

    steps = [
        Step(
            id="p2.package_check",
            argv=["python3", "round2/reconcile.py", "--skip-model-check"],
            produces=[],
            note="pytest + static package validation (prints PACKAGE_ONLY)",
        ),
        Step(
            id="p2.reconcile_m1",
            gpu=True,
            needs=["p2.package_check"],
            argv=[
                "python3",
                "round2/reconcile.py",
                "--skip-pytest",
                "--model-id",
                model_id,
            ],
            produces=["round2_outputs/reconcile/m1.heldout.json", "round2_outputs/reconcile/m1.sibling.json"],
            note="Company acceptance: must print ACCEPTED; also provides the Gate-2 M1 heldout reference.",
        ),
        Step(
            id="p2.collect",
            gpu=True,
            needs=["p2.package_check"],
            argv=[
                "python3",
                "round2/collect_failures.py",
                "--m1-adapter",
                m1_adapter,
                "--model-id",
                model_id,
                "--failures",
                "EDG-EXP1/results/a2_failures.json",
                "--exp2-root",
                "EDG-EXP2-struct",
            ],
            produces=["round2_outputs/f2_failures.json"],
        ),
        Step(
            id="p2.smoke",
            gpu=True,
            needs=["p2.collect"],
            argv=[
                "python3",
                "round2/loop/no_patch_equivalence_smoke.py",
                "--collect-json",
                "round2_outputs/f2_failures.json",
                "--model-id",
                model_id,
                "--limit",
                "10",
            ],
            produces=["round2_outputs/loop/no_patch_equivalence.json"],
            note="Must pass before the loop result is accepted (v1.24).",
        ),
        Step(
            id="p2.loop",
            gpu=True,
            needs=["p2.smoke"],
            argv=[
                "python3",
                "round2/loop/evolution_loop_round2.py",
                "--m1-adapter",
                m1_adapter,
                "--f2",
                "round2_outputs/f2_failures.json",
                "--model-id",
                model_id,
                "--output-dir",
                "round2_outputs/loop",
            ],
            produces=["round2_outputs/loop/evolution_loop.json"],
        ),
        Step(
            id="p2.t2",
            gpu=True,
            needs=["p2.loop"],
            argv=[
                "python3",
                "round2/build_t2.py",
                "--f2-json",
                "round2_outputs/f2_failures.json",
                "--loop-output",
                "round2_outputs/loop/evolution_loop.json",
                "--m1-adapter",
                m1_adapter,
                "--model-id",
                model_id,
            ],
            produces=["round2_outputs/t2.jsonl", "round2_outputs/t1_t2.jsonl", "round2_outputs/t2_summary.json"],
        ),
        Step(
            id="p2.m1success",
            gpu=True,
            needs=["p2.collect"],
            argv=[
                "python3",
                "round2/build_m1_success.py",
                "--m1-adapter",
                m1_adapter,
                "--collect-json",
                "round2_outputs/f2_failures.json",
                "--model-id",
                model_id,
                "--failures",
                "EDG-EXP1/results/a2_failures.json",
                "--exp2-root",
                "EDG-EXP2-struct",
            ],
            produces=["round2_outputs/m1_train_success.jsonl"],
        ),
        Step(
            id="p2.replay",
            needs=["p2.t2", "p2.m1success"],
            argv=[
                "python3",
                "round2/build_replay2.py",
                "--m1-success-jsonl",
                "round2_outputs/m1_train_success.jsonl",
                "--t2-jsonl",
                "round2_outputs/t2.jsonl",
            ],
            produces=["round2_outputs/replay2.jsonl"],
        ),
    ]
    for index, seed in enumerate(seeds):
        train_id = f"p2.train.s{seed}"
        steps.append(
            Step(
                id=train_id,
                gpu=True,
                needs=["p2.replay"],
                argv=[
                    "python3",
                    "round2/train_round2.py",
                    "--m1-adapter",
                    m1_adapter,
                    "--train-jsonl",
                    "round2_outputs/t2.jsonl",
                    "--replay-jsonl",
                    "round2_outputs/replay2.jsonl",
                    "--seed",
                    str(seed),
                    "--output-adapter-dir",
                    f"round2_outputs/a2_seed{seed}",
                    "--output-json",
                    f"round2_outputs/a2_seed{seed}.json",
                ],
                produces=[f"round2_outputs/a2_seed{seed}.json"],
            )
        )
        eval_argv = [
            "python3",
            "round2/eval_round2.py",
            "--m1-adapter",
            m1_adapter,
            "--a2-adapter",
            f"round2_outputs/a2_seed{seed}",
            "--train-signature-jsonl",
            "round2_outputs/t1_t2.jsonl",
            "--model-id",
            model_id,
            "--output-prefix",
            f"round2_outputs/eval/m2_seed{seed}",
        ]
        produces = [
            f"round2_outputs/eval/m2_seed{seed}.heldout.json",
            f"round2_outputs/eval/m2_seed{seed}.sibling.json",
        ]
        if index == 0:
            # teacher-2 = M1+H2 is seed-independent: forward it once, on the
            # first eval; gate2_glue computes retention_2 for the other seeds.
            eval_argv.extend(["--teacher2-loop-output", "round2_outputs/loop/evolution_loop.json"])
            produces.append(f"round2_outputs/eval/m2_seed{seed}.teacher2_heldout.json")
        steps.append(
            Step(
                id=f"p2.eval.s{seed}",
                gpu=True,
                needs=[train_id],
                argv=eval_argv,
                produces=produces,
            )
        )
    steps.append(
        Step(
            id="p2.gate2",
            needs=["p2.reconcile_m1"] + [f"p2.eval.s{seed}" for seed in seeds],
            argv=["python3", "-m", "launcher.gate2_glue"],
            produces=["round2_outputs/eval/gate2.json", "round2_outputs/gate2_report.md"],
            note="Mechanical Gate-2 dynamics classification; human sign-off required.",
        )
    )
    return steps
