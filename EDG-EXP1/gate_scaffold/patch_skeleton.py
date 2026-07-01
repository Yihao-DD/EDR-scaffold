from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Patch:
    patch_id: str
    component: str
    patch_type: str
    payload: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_patches(failure_episode, diagnosis):
    """Generate router and validator candidate patches for every failure.

    Hard constraint: every failure receives both router-class and validator-class
    candidates. Candidate generation must not prefilter by failure_class.
    """
    raise NotImplementedError("Phase B, human present")
