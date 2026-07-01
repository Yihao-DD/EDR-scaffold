def alloc_random(*args, **kwargs):
    """Random allocation baseline. Phase B implementation only."""
    raise NotImplementedError("Phase B, human present")


def alloc_uniform(*args, **kwargs):
    """Uniform router/validator budget split baseline. Phase B implementation only."""
    raise NotImplementedError("Phase B, human present")


def alloc_greedy_by_diagnosis_confidence(*args, **kwargs):
    """Greedy baseline keyed by diagnostic confidence. Phase B implementation only."""
    raise NotImplementedError("Phase B, human present")


def alloc_ours_semantic(*args, **kwargs):
    """Semantic allocator that reads diagnosis content.

    This is the Experiment 1 core being tested by original-vs-decoupled quality
    control, so it remains locked for human-reviewed Phase B implementation.
    """
    raise NotImplementedError("Phase B, human present")


def alloc_oracle_empirical(*args, **kwargs):
    """Empirical oracle that enumerates candidates and measures validation net gain."""
    raise NotImplementedError("Phase B, human present")


def estimate_value(*args, **kwargs):
    """Estimate patch value from diagnosis and history.

    This is a judgment-heavy semantic value estimator and remains locked for
    human-reviewed Phase B implementation.
    """
    raise NotImplementedError("Phase B, human present")
