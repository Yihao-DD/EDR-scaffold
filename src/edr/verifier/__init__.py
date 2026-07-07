"""BFCL AST verifier — the ONLY judge in this project.

It is a task-level oracle against human ground truth (function choice ∧
required arguments ∧ argument values within the accepted answer set), not a
format check. No LLM judge exists anywhere in the pipeline.
"""

from edr.verifier.ast_matcher import (  # noqa: F401
    call_success,
    parse_bfcl_call,
    parse_model_output,
    router_success,
    validator_success,
)
