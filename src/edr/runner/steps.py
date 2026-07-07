"""Step DAG definitions for the ablation stage and round 2.

Every step is a subprocess `python -m edr.<module> ...` run from the repo
root with an explicit `produces` list (the resume markers). Recipes come from
configs/ and are preregistered — the runner never invents hyperparameters.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from edr.config import load_config, resolve_model_id
from edr.paths import REPO_ROOT


@dataclass
class Step:
    id: str
    argv: list
    gpu: bool = False
    needs: list = field(default_factory=list)
    produces: list = field(default_factory=list)
    note: str = ""


PY = sys.executable or "python3"


def _module(name, *args):
    return [PY, "-m", name, *[str(a) for a in args]]


def _train_step(step_id, config, dataset, adapter_dir, seed, kl_lambda, needs):
    recipe = config["recipe_locked"]
    return Step(
        id=step_id,
        gpu=True,
        needs=list(needs),
        argv=_module(
            "edr.training.lora",
            "--model-id", resolve_model_id(config),
            "--dataset", dataset,
            "--output-dir", adapter_dir,
            "--rank", recipe["rank"],
            "--lr", recipe["lr"],
            "--epochs", recipe["epochs"],
            "--seed", seed,
            "--kl-anchor-lambda", kl_lambda,
        ),
        produces=[f"{adapter_dir}/train_metadata.json"],
    )


def _heldout_step(step_id, config, arm, adapter_dir, dataset, seed, output, needs):
    return Step(
        id=step_id,
        gpu=True,
        needs=list(needs),
        argv=_module(
            "edr.evaluation.heldout",
            "--model-id", resolve_model_id(config),
            "--adapter-dir", adapter_dir,
            "--arm", arm,
            "--config", f"{arm}_recipe_locked",
            "--seed", seed,
            "--run-id", f"{arm}_seed{seed}",
            "--train-dataset", dataset,
            "--output", output,
        ),
        produces=[output],
    )


def _sibling_step(step_id, config, arm, adapter_dir, seed, output, needs):
    return Step(
        id=step_id,
        gpu=True,
        needs=list(needs),
        argv=_module(
            "edr.evaluation.sibling",
            "--model-id", resolve_model_id(config),
            "--adapter-dir", adapter_dir,
            "--arm", arm,
            "--config", f"{arm}_recipe_locked",
            "--seed", seed,
            "--run-id", f"{arm}_seed{seed}",
            "--output", output,
        ),
        produces=[output],
    )


def _train_eval_block(steps, config, arm, dataset, kl_lambda, seeds, needs_build):
    base = arm.split("_")[0]
    for seed in seeds:
        adapter = f"outputs/ablations/{base}/adapters/{arm}_seed{seed}"
        train_id = f"ab.{arm}.train.s{seed}"
        steps.append(_train_step(train_id, config, dataset, adapter, seed, kl_lambda, needs_build))
        heldout_out = f"outputs/ablations/{base}/eval/{arm}_seed{seed}_heldout.json"
        sibling_out = f"outputs/ablations/{base}/eval/{arm}_seed{seed}_sibling.json"
        steps.append(_heldout_step(f"ab.{arm}.heldout.s{seed}", config, arm, adapter, dataset, seed, heldout_out, [train_id]))
        steps.append(_sibling_step(f"ab.{arm}.sibling.s{seed}", config, arm, adapter, seed, sibling_out, [train_id]))


def ablation_steps():
    config = load_config("ablations")
    steps = []
    seeds = config["seeds"]
    a5 = config["a5"]
    kl = config["recipe_locked"]["kl_anchor_lambda"]

    shard_ids = []
    for shard in range(a5["shard_count"]):
        step_id = f"ab.a5.resample.shard{shard}"
        shard_ids.append(step_id)
        steps.append(
            Step(
                id=step_id,
                gpu=True,
                argv=_module("edr.data.ablations", "resample-teacher", "--shard-index", shard, "--shard-count", a5["shard_count"]),
                produces=[f"outputs/ablations/a5/a5_teacher_all_shard_{shard}.json"],
            )
        )
    steps.append(
        Step(
            id="ab.a5.build",
            argv=_module("edr.data.ablations", "build-a5"),
            needs=shard_ids,
            produces=["outputs/ablations/a5/distill_a5_unverified.jsonl", "outputs/ablations/a5/a5_dataset_summary.json"],
        )
    )
    _train_eval_block(steps, config, "a5", "outputs/ablations/a5/distill_a5_unverified.jsonl", kl, seeds, ["ab.a5.build"])

    steps.append(
        Step(
            id="ab.a7a8.build",
            argv=_module("edr.data.ablations", "build-a7a8"),
            produces=["outputs/ablations/a7/distill_a7_noreplay.jsonl"]
            + [f"outputs/ablations/a8/distill_a8_{int(f * 100)}pct.jsonl" for f in config["a8"]["fractions"]],
        )
    )
    # A7 = recipe minus replay minus KL: kl lambda forced to 0.
    _train_eval_block(steps, config, "a7", "outputs/ablations/a7/distill_a7_noreplay.jsonl", 0.0, seeds, ["ab.a7a8.build"])
    for fraction in config["a8"]["fractions"]:
        tag = f"{int(fraction * 100)}pct"
        _train_eval_block(steps, config, f"a8_{tag}", f"outputs/ablations/a8/distill_a8_{tag}.jsonl", kl, seeds, ["ab.a7a8.build"])

    steps.append(
        Step(
            id="ab.a6.index",
            gpu=True,  # dense encoding benefits from a GPU; BM25 alone is CPU
            argv=_module("edr.retrieval.patch_retrieval", "build-index"),
            produces=["outputs/ablations/a6/a6_frozen_retrieval.json"],
        )
    )
    steps.append(
        Step(
            id="ab.a6.eval",
            gpu=True,
            needs=["ab.a6.index"],
            argv=_module("edr.retrieval.patch_retrieval", "eval"),
            produces=["outputs/ablations/a6/a6_summary.json"],
        )
    )
    steps.append(
        Step(
            id="ab.a11.scramble",
            argv=_module("edr.evaluation.placebo", "scramble"),
            produces=["outputs/ablations/a11/a11_scrambled_patches.json"],
        )
    )
    steps.append(
        Step(
            id="ab.a11.eval",
            gpu=True,
            needs=["ab.a11.scramble"],
            argv=_module("edr.evaluation.placebo", "eval"),
            produces=["outputs/ablations/a11/a11_summary.json"],
        )
    )
    steps.append(
        Step(
            id="ab.qualitative",
            argv=_module("edr.analysis.qualitative"),
            produces=["outputs/ablations/qualitative/case_package.json"],
        )
    )
    eval_ids = [step.id for step in steps if ".heldout." in step.id or ".sibling." in step.id or step.id in ("ab.a6.eval", "ab.a11.eval")]
    steps.append(
        Step(
            id="ab.gate1",
            argv=_module("edr.analysis.gate1"),
            needs=eval_ids,
            produces=["outputs/ablations/gate1_report.md", "outputs/ablations/gate1_report.json"],
            note="Mechanical tally only; human sign-off required.",
        )
    )
    return steps


def round2_steps():
    config = load_config("round2")
    seeds = config["seeds"]

    steps = [
        Step(
            id="r2.package_check",
            argv=[PY, "scripts/reconcile.py", "--skip-model-check"],
            produces=[],
            note="pytest + static package validation (prints PACKAGE_ONLY)",
        ),
        Step(
            id="r2.reconcile_m1",
            gpu=True,
            needs=["r2.package_check"],
            argv=[PY, "scripts/reconcile.py", "--skip-pytest"],
            produces=["outputs/reconcile/m1.heldout.json", "outputs/reconcile/m1.sibling.json"],
            note="Acceptance: must print ACCEPTED; also the Gate-2 M1 heldout reference.",
        ),
        Step(
            id="r2.collect",
            gpu=True,
            needs=["r2.package_check"],
            argv=_module("edr.round2.collect_failures"),
            produces=["outputs/round2/f2_failures.json"],
        ),
        Step(
            id="r2.smoke",
            gpu=True,
            needs=["r2.collect"],
            argv=_module("edr.round2.equivalence_smoke", "--collect-json", "outputs/round2/f2_failures.json", "--limit", 10),
            produces=["outputs/round2/loop/no_patch_equivalence.json"],
            note="Must pass before the loop result is accepted.",
        ),
        Step(
            id="r2.loop",
            gpu=True,
            needs=["r2.smoke"],
            argv=_module(
                "edr.scaffold.loop",
                # Round-2 loop input = F2 (M1's residual failures), split internally
                # at the frozen seed — same semantics as the reviewed round-2 wrapper.
                "--input", "outputs/round2/f2_failures.json",
                "--base-adapter-dir", config["m1_adapter"],
                "--output", "outputs/round2/loop/evolution_loop.json",
                "--seeds", config.get("loop_split_seed", 20260630),
                # NL family only: T2 consumes NL patches exclusively and family
                # loops are independent, so this is identical H2 at 1/3 the cost.
                "--families", "NL",
            ),
            produces=["outputs/round2/loop/evolution_loop.json"],
        ),
        Step(
            id="r2.t2",
            gpu=True,
            needs=["r2.loop"],
            argv=_module(
                "edr.round2.build_t2",
                "--f2-json", "outputs/round2/f2_failures.json",
                "--loop-output", "outputs/round2/loop/evolution_loop.json",
            ),
            produces=["outputs/round2/t2.jsonl", "outputs/round2/t1_t2.jsonl", "outputs/round2/t2_summary.json"],
        ),
        Step(
            id="r2.m1success",
            gpu=True,
            needs=["r2.collect"],
            argv=_module("edr.round2.build_m1_success", "--collect-json", "outputs/round2/f2_failures.json"),
            produces=["outputs/round2/m1_train_success.jsonl"],
        ),
        Step(
            id="r2.replay",
            needs=["r2.t2", "r2.m1success"],
            argv=_module(
                "edr.round2.build_replay2",
                "--m1-success-jsonl", "outputs/round2/m1_train_success.jsonl",
                "--t2-jsonl", "outputs/round2/t2.jsonl",
            ),
            produces=["outputs/round2/replay2.jsonl"],
        ),
    ]
    for index, seed in enumerate(seeds):
        train_id = f"r2.train.s{seed}"
        steps.append(
            Step(
                id=train_id,
                gpu=True,
                needs=["r2.replay"],
                argv=_module(
                    "edr.round2.train",
                    "--train-jsonl", "outputs/round2/t2.jsonl",
                    "--replay-jsonl", "outputs/round2/replay2.jsonl",
                    "--seed", seed,
                    "--output-adapter-dir", f"outputs/round2/a2_seed{seed}",
                ),
                produces=[f"outputs/round2/a2_seed{seed}/train_metadata.json"],
            )
        )
        eval_argv = _module(
            "edr.round2.evaluate",
            "--a2-adapter", f"outputs/round2/a2_seed{seed}",
            "--output-prefix", f"outputs/round2/eval/m2_seed{seed}",
        )
        produces = [
            f"outputs/round2/eval/m2_seed{seed}.heldout.json",
            f"outputs/round2/eval/m2_seed{seed}.sibling.json",
        ]
        if index == 0:
            # teacher-2 is seed-independent: forward once; gate2 report reuses it.
            eval_argv.extend(["--teacher2-loop-output", "outputs/round2/loop/evolution_loop.json"])
            produces.append(f"outputs/round2/eval/m2_seed{seed}.teacher2_heldout.json")
        steps.append(Step(id=f"r2.eval.s{seed}", gpu=True, needs=[train_id], argv=eval_argv, produces=produces))
    steps.append(
        Step(
            id="r2.gate2",
            needs=["r2.reconcile_m1"] + [f"r2.eval.s{seed}" for seed in seeds],
            argv=_module("edr.analysis.gate2", "report"),
            produces=["outputs/round2/eval/gate2.json", "outputs/round2/gate2_report.md"],
            note="Mechanical Gate-2 classification; human sign-off required.",
        )
    )
    return steps
