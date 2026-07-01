import json
from pathlib import Path

from gate_scaffold.arms_skeleton import (
    alloc_greedy_by_diagnosis_confidence,
    alloc_oracle_empirical,
    alloc_ours_semantic,
    alloc_random,
    alloc_uniform,
    estimate_value,
)
from gate_scaffold.budget_meter import load_budget_config
from gate_scaffold.diagnosis_skeleton import DiagnosisRecord, diagnose
from gate_scaffold.metrics_schema import ExperimentMetrics, PatchRecord
from gate_scaffold.patch_skeleton import generate_patches
from scripts.bfcl_common import write_json


def _raises_not_implemented(func, *args):
    try:
        func(*args)
    except NotImplementedError:
        return True
    return False


def build_scaffold_results():
    diagnosis = DiagnosisRecord(layer="tool_interface", confidence=0.0, notes="", candidate_patch_ids=[])
    budgets = load_budget_config()
    metrics = ExperimentMetrics(
        arm="ours_semantic",
        split="multiple",
        version="original",
        budget="tight",
        seed=0,
        repaired_failures=0,
        validation_calls=0,
        net_pass_rate_gain=0.0,
        regression_count=0,
        oracle_80pct_calls=None,
        oracle_90pct_calls=None,
    )
    patch = PatchRecord(
        patch_id="p0",
        component="router",
        patch_type="function_alias",
        validation_cost={"calls": 0, "latency_ms": 0, "context_tokens": 0},
        rollbackable=True,
    )
    return {
        "a4": {
            "diagnose_raises_not_implemented": _raises_not_implemented(diagnose, {}, {}, []),
            "generate_patches_raises_not_implemented": _raises_not_implemented(generate_patches, {}, diagnosis),
        },
        "a5": {
            "alloc_random_raises_not_implemented": _raises_not_implemented(alloc_random, None),
            "alloc_uniform_raises_not_implemented": _raises_not_implemented(alloc_uniform, None),
            "alloc_greedy_by_diagnosis_confidence_raises_not_implemented": _raises_not_implemented(alloc_greedy_by_diagnosis_confidence, None),
            "alloc_ours_semantic_raises_not_implemented": _raises_not_implemented(alloc_ours_semantic, None),
            "alloc_oracle_empirical_raises_not_implemented": _raises_not_implemented(alloc_oracle_empirical, None),
            "estimate_value_raises_not_implemented": _raises_not_implemented(estimate_value, None),
        },
        "a6": {
            "budget_tiers": list(budgets.keys()),
            "budget_caps": {name: caps.__dict__ for name, caps in budgets.items()},
            "metrics_example": metrics.to_json_dict(),
            "patch_record_example": patch.to_json_dict(),
        },
        "locked_phase_b_cores": ["diagnosis", "patch generation", "semantic value estimation", "empirical oracle"],
    }


def main():
    result = build_scaffold_results()
    write_json("results/a4_a5_a6_scaffold.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
