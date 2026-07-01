import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate_scaffold.arms_skeleton import (
    arm_no_update,
    arm_oracle_allocator,
    arm_ours_history_based,
    arm_random_allocator,
    arm_router_only,
    arm_uniform_split,
    arm_update_both,
    arm_validator_only,
)
from gate_scaffold.budget_meter import BudgetMeter, load_budget_config
from gate_scaffold.metrics_schema import AcceptedUpdateLog, ArmBudgetSeedMetrics
from gate_scaffold.patch_operators_skeleton import (
    apply_patch,
    estimate_utility,
    generate_router_patches,
    generate_validator_patches,
    rollback_patch,
)


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _raises_not_implemented(func):
    try:
        func(None)
    except NotImplementedError:
        return True
    return False


def main():
    budget_config = load_budget_config()
    budget_checks = {}
    for name, caps in budget_config.items():
        meter = BudgetMeter(caps)
        budget_checks[name] = {
            "caps": caps.__dict__,
            "first_add": meter.add(validation_calls=1, latency_ms=1, context_tokens=1)["over_limit"],
        }
    _write(
        "results/b1_budget_framework.json",
        {
            "budget_unit": "1 validation call = 1 patch replayed on 1 validation episode",
            "tiers": budget_checks,
            "placeholder_caps": True,
            "asserts": {"three_tiers_present": set(budget_checks) == {"loose", "medium", "tight"}},
        },
    )

    arms = [
        arm_no_update,
        arm_router_only,
        arm_validator_only,
        arm_uniform_split,
        arm_random_allocator,
        arm_update_both,
        arm_oracle_allocator,
        arm_ours_history_based,
    ]
    _write(
        "results/b2_arms_skeleton.json",
        {"n_arms": len(arms), "raises_not_implemented": {func.__name__: _raises_not_implemented(func) for func in arms}},
    )

    operators = [generate_router_patches, generate_validator_patches, apply_patch, rollback_patch, estimate_utility]
    _write(
        "results/b3_patch_operators_skeleton.json",
        {
            "n_operators": len(operators),
            "raises_not_implemented": {func.__name__: _raises_not_implemented(func) for func in operators},
        },
    )

    metrics = ArmBudgetSeedMetrics("ours", "tight", 7, 0.0, 1, 2, 3, 0, 0, 0.0)
    update = AcceptedUpdateLog(
        "u1",
        "router",
        ["e1"],
        "alias",
        {"from": "x", "to": "y"},
        {"calls": 1, "latency_ms": 2, "context_tokens": 3},
        {"fixed": 0, "regressed": 0, "net_gain": 0},
        False,
    )
    _write(
        "results/b4_metrics_schema.json",
        {"metrics_example": metrics.to_json_dict(), "accepted_update_example": update.to_json_dict()},
    )

    _write(
        "results/b5_anti_shortcut_rules.json",
        {
            "rules_document": "logs/b5_anti_shortcut_rules.md",
            "locked_cores": ["patch generation", "utility estimation", "oracle", "repair experiment"],
        },
    )


if __name__ == "__main__":
    main()
