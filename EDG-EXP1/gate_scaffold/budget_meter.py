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
    def __init__(self, caps: BudgetCaps):
        self.caps = caps
        self.totals = BudgetTotals()

    def add(self, validation_calls=0, latency_ms=0, context_tokens=0):
        self.totals.validation_calls += validation_calls
        self.totals.latency_ms += latency_ms
        self.totals.context_tokens += context_tokens
        return {
            "totals": {
                "validation_calls": self.totals.validation_calls,
                "latency_ms": self.totals.latency_ms,
                "context_tokens": self.totals.context_tokens,
            },
            "over_limit": self.over_limit(),
        }

    def over_limit(self):
        return (
            self.totals.validation_calls > self.caps.validation_calls
            or self.totals.latency_ms > self.caps.latency_ms
            or self.totals.context_tokens > self.caps.context_tokens
        )


def load_budget_config():
    return {
        "loose": BudgetCaps(validation_calls=1000, latency_ms=3600000, context_tokens=2000000),
        "medium": BudgetCaps(validation_calls=300, latency_ms=1200000, context_tokens=600000),
        "tight": BudgetCaps(validation_calls=100, latency_ms=600000, context_tokens=200000),
    }
