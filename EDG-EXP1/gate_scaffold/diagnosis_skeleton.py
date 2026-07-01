from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DiagnosisRecord:
    layer: str
    confidence: float
    notes: str
    candidate_patch_ids: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose(failure_episode, trace, harness_layers) -> DiagnosisRecord:
    """Phase B lightweight HarnessFix-style diagnostic.

    Reads failure trace and attributes the failure to a harness layer such as
    Tool Interface, Context, or Verification. This is the semantic core of
    Experiment 1 and remains locked for human-reviewed Phase B implementation.
    """
    raise NotImplementedError("Phase B, human present")
