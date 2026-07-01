from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ArmBudgetSeedMetrics:
    arm: str
    budget: str
    seed: int
    repair_success: float
    validation_calls: int
    latency_ms: int
    context_tokens: int
    regression_count: int
    rollback_count: int
    net_utility: float

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AcceptedUpdateLog:
    update_id: str
    component: str
    trigger_failure_ids: List[str]
    patch_type: str
    patch: Dict[str, Any]
    validation_cost: Dict[str, int]
    validation_result: Dict[str, int]
    accepted: bool

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)
