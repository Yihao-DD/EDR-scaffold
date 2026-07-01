import json

import pytest

from gate_scaffold.arms_skeleton import (
    alloc_greedy_by_diagnosis_confidence,
    alloc_oracle_empirical,
    alloc_ours_semantic,
    alloc_random,
    alloc_uniform,
    estimate_value,
)
from gate_scaffold.budget_meter import BudgetCaps, BudgetMeter, load_budget_config
from gate_scaffold.diagnosis_skeleton import DiagnosisRecord, diagnose
from gate_scaffold.metrics_schema import ExperimentMetrics, PatchRecord
from gate_scaffold.patch_skeleton import Patch, generate_patches
from scripts.write_scaffold_results import build_scaffold_results


def test_diagnosis_and_patch_skeletons_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        diagnose({}, {}, [])
    with pytest.raises(NotImplementedError):
        generate_patches({}, DiagnosisRecord(layer="tool_interface", confidence=0.0, notes="", candidate_patch_ids=[]))


def test_allocator_and_value_skeletons_raise_not_implemented():
    for func in [
        alloc_random,
        alloc_uniform,
        alloc_greedy_by_diagnosis_confidence,
        alloc_ours_semantic,
        alloc_oracle_empirical,
        estimate_value,
    ]:
        with pytest.raises(NotImplementedError):
            func(None)


def test_budget_meter_counts_validation_latency_and_tokens():
    meter = BudgetMeter(BudgetCaps(validation_calls=2, latency_ms=100, context_tokens=50))

    assert meter.add(validation_calls=1, latency_ms=25, context_tokens=20)["over_limit"] is False
    assert meter.add(validation_calls=2, latency_ms=80, context_tokens=40)["over_limit"] is True
    assert meter.totals.validation_calls == 3
    assert set(load_budget_config()) == {"loose", "medium", "tight"}


def test_metrics_and_patch_records_serialize():
    metrics = ExperimentMetrics(
        arm="ours_semantic",
        split="multiple",
        version="decoupled",
        budget="tight",
        seed=7,
        repaired_failures=3,
        validation_calls=10,
        net_pass_rate_gain=0.1,
        regression_count=1,
        oracle_80pct_calls=20,
        oracle_90pct_calls=30,
    )
    patch = PatchRecord(
        patch_id="p1",
        component="router",
        patch_type="function_alias",
        validation_cost={"calls": 1, "latency_ms": 2, "context_tokens": 3},
        rollbackable=True,
    )
    generated = Patch(patch_id="p2", component="validator", patch_type="required_check", payload={})

    assert metrics.to_json_dict()["version"] == "decoupled"
    assert patch.to_json_dict()["rollbackable"] is True
    assert generated.component == "validator"
    json.dumps({"metrics": metrics.to_json_dict(), "patch": patch.to_json_dict()})


def test_scaffold_results_record_locked_phase_b_cores():
    result = build_scaffold_results()

    assert result["a4"]["diagnose_raises_not_implemented"] is True
    assert result["a4"]["generate_patches_raises_not_implemented"] is True
    assert result["a5"]["estimate_value_raises_not_implemented"] is True
    assert set(result["a6"]["budget_tiers"]) == {"loose", "medium", "tight"}
    assert "diagnosis" in result["locked_phase_b_cores"]
    json.dumps(result)
