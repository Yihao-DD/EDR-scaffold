from dataclasses import dataclass


@dataclass
class BudgetCaps:
    validation_calls: int
    latency_ms: int
    context_tokens: int


@dataclass
class BudgetTotals:
    validation_calls: int = 0
    latency_ms: int = 0
    context_tokens: int = 0


class BudgetMeter:
    """Pure budget accounting. It does not decide how any budget is spent."""

    def __init__(self, caps: BudgetCaps):
        self.caps = caps
        self.totals = BudgetTotals()

    def add(self, validation_calls=0, latency_ms=0, context_tokens=0):
        self.totals.validation_calls += validation_calls
        self.totals.latency_ms += latency_ms
        self.totals.context_tokens += context_tokens
        return {"totals": self.totals, "over_limit": self.over_limit()}

    def over_limit(self):
        return (
            self.totals.validation_calls > self.caps.validation_calls
            or self.totals.latency_ms > self.caps.latency_ms
            or self.totals.context_tokens > self.caps.context_tokens
        )


def load_budget_config():
    # Placeholder caps for morning review. These are not gate thresholds.
    return {
        "loose": BudgetCaps(validation_calls=1000, latency_ms=3_600_000, context_tokens=2_000_000),
        "medium": BudgetCaps(validation_calls=300, latency_ms=1_200_000, context_tokens=600_000),
        "tight": BudgetCaps(validation_calls=100, latency_ms=600_000, context_tokens=200_000),
    }
