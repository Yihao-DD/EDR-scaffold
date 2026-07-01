from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExperimentMetrics:
    arm: str
    split: str
    version: str
    budget: str
    seed: int
    repaired_failures: int
    validation_calls: int
    net_pass_rate_gain: float
    regression_count: int
    oracle_80pct_calls: int | None
    oracle_90pct_calls: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchRecord:
    patch_id: str
    component: str
    patch_type: str
    validation_cost: dict[str, int]
    rollbackable: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
