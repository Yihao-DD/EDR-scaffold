"""Scaffold: patch structures, prompt injection, and the evolve loop."""

from edr.scaffold.patches import (  # noqa: F401
    PROMPT_TEMPLATE,
    PatchCandidate,
    build_prompt,
    materialize_candidate,
    patch_relevant_to_episode,
    relevant_patches_for_episode,
)
