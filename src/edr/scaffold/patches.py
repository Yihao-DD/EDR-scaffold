"""Patch data structures, relevance matching, and scaffold-on prompt building.

Ported verbatim from the frozen round-1 evolution loop; the prompt template
and relevance rules are part of the preregistered protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

FAMILIES = ["NL", "NL-padded", "STRUCT"]

PROMPT_TEMPLATE = """You are a function-calling model. Choose exactly one function from the function document and produce exactly one JSON object.

Output format:
{{"name": "<function_name>", "arguments": {{"<argument_name>": <argument_value>}}}}

Harness patch guidance:
{patch_context}

User query:
{query}

Function document:
{function_doc}

Return only the JSON object. Do not explain."""


@dataclass
class PatchCandidate:
    patch_id: str
    family: str
    component: str
    trigger_episode_id: str
    target_function: str
    selected_function: str
    parameter_names: list[str]
    patch_text: str
    token_count: int
    iu_ids: list[str]

    def to_json_dict(self):
        return asdict(self)


def materialize_candidate(failure, family, tokenizer=None):
    from edr.scaffold.patch_generation import build_pair

    pair = build_pair(failure, tokenizer=tokenizer)
    if family == "STRUCT":
        patch_text = pair["struct_patch"]
        token_count = pair["struct_token_count"]
        iu_ids = pair["struct_iu_ids"]
    elif family == "NL":
        patch_text = pair["nl_patch"]
        token_count = pair["nl_token_count"]
        iu_ids = pair["nl_iu_ids"]
    elif family == "NL-padded":
        patch_text = pair["nl_padded_patch"]
        token_count = pair["nl_padded_token_count"]
        iu_ids = pair["nl_padded_iu_ids"]
    else:
        raise ValueError(f"Unknown patch family: {family}")
    component = "router" if failure.get("failure_class") == "router_fail" else "validator"
    return PatchCandidate(
        patch_id=f"{family}:{failure['episode_id']}",
        family=family,
        component=component,
        trigger_episode_id=failure["episode_id"],
        target_function=failure.get("ground_truth_call", {}).get("name", ""),
        selected_function=(failure.get("predicted_call") or {}).get("name", ""),
        parameter_names=list((failure.get("ground_truth_call", {}).get("accepted_arguments") or {}).keys()),
        patch_text=patch_text,
        token_count=token_count,
        iu_ids=iu_ids,
    )


def patch_relevant_to_episode(patch, episode):
    gt_name = episode.get("ground_truth_call", {}).get("name", "")
    if patch.component == "router":
        function_names = {function.get("name") for function in episode.get("function_pool", []) if isinstance(function, dict)}
        selected = (episode.get("predicted_call") or {}).get("name", "")
        return patch.target_function == gt_name and (patch.selected_function == selected or patch.selected_function in function_names)
    episode_params = set((episode.get("ground_truth_call", {}).get("accepted_arguments") or {}).keys())
    return patch.target_function == gt_name and bool(episode_params.intersection(patch.parameter_names))


def relevant_patches_for_episode(patches, episode):
    return [patch for patch in patches if patch_relevant_to_episode(patch, episode)]


def build_prompt(episode, patches):
    patch_context = "\n\n".join(f"[PATCH {index}] {patch.patch_text}" for index, patch in enumerate(patches, start=1))
    if not patch_context:
        patch_context = "None."
    return PROMPT_TEMPLATE.format(
        patch_context=patch_context,
        query=episode.get("query", ""),
        function_doc=json.dumps(episode.get("function_pool", []), ensure_ascii=False),
    )
