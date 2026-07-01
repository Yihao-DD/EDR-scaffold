def _morning():
    raise NotImplementedError("implement in the morning, human present")


def arm_no_update(*args, **kwargs):
    """No external scaffold update is applied. Must not run patch generation or validation."""
    _morning()


def arm_router_only(*args, **kwargs):
    """Only router patches may be considered. Candidate generation rules still need morning review."""
    _morning()


def arm_validator_only(*args, **kwargs):
    """Only validator/parser patches may be considered. Candidate generation rules still need morning review."""
    _morning()


def arm_uniform_split(*args, **kwargs):
    """Validation budget is split uniformly between components without reading failure semantics."""
    _morning()


def arm_random_allocator(*args, **kwargs):
    """Validation budget is allocated randomly; random seed and candidate order controls need morning review."""
    _morning()


def arm_update_both(*args, **kwargs):
    """Both component candidates are validated when budget permits; exact consumption logic is locked."""
    _morning()


def arm_oracle_allocator(*args, **kwargs):
    """Empirical oracle: enumerate candidates and choose measured net gain. No human target labels allowed."""
    _morning()


def arm_ours_history_based(*args, **kwargs):
    """History-only allocator: use measured component net gain/cost; do not read failure text or function names."""
    _morning()
