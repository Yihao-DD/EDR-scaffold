"""Parallel-category prompting and judging for the sibling regression arena.

Ported verbatim from the frozen round-1 arena module. The arena itself
(data/round1/sibling_arena.json) is a frozen artifact; this module only
renders prompts and judges outputs. Multi-call matching is a backtracking
assignment against ground-truth calls — still pure AST matching, no LLM.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter

PARALLEL_CATEGORIES = ["parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"]

PARALLEL_PROMPT_TEMPLATE = """You are a function-calling model. Choose every function call required by the user from the function document.

Output format:
[{{"name": "<function_name>", "arguments": {{"<argument_name>": <argument_value>}}}}, ...]

Rules:
- Return exactly one JSON array.
- Each array item is one function call object.
- Do not explain.

User query:
{query}

Function document:
{function_doc}

Return only the JSON array."""


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_call(call):
    if isinstance(call, dict) and "name" in call and "arguments" in call:
        return {"name": call["name"], "arguments": dict(call.get("arguments") or {})}
    if isinstance(call, dict) and len(call) == 1:
        name, arguments = next(iter(call.items()))
        return {"name": name, "arguments": dict(arguments or {})}
    if not isinstance(call, str):
        raise ValueError(f"Unsupported call shape: {call!r}")
    expression = ast.parse(call, mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Expected a function call expression: {call!r}")
    arguments = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("Variadic keyword arguments are not supported")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": call_name(expression.func), "arguments": arguments}


def parse_model_output_multi(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [parse_call(item) for item in obj]

    calls = []
    consumed = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if any(start <= index < stop for start, stop in consumed):
            continue
        try:
            calls.append(parse_call(obj))
            consumed.append((index, index + end))
        except Exception:
            continue
    if calls:
        return calls

    line_calls = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)", line, re.S):
            line_calls.append(parse_call(line))
    if line_calls:
        return line_calls
    raise ValueError("No parseable parallel function calls found")


def has_concrete(value):
    if value == "":
        return False
    if isinstance(value, dict):
        return any(has_concrete(nested) for nested in value.values())
    return True


def value_matches(value, accepted):
    if isinstance(accepted, dict):
        if not isinstance(value, dict):
            return False
        required = {
            key
            for key, nested in accepted.items()
            if any(has_concrete(candidate) for candidate in (nested if isinstance(nested, list) else [nested]))
        }
        if set(value) != required:
            return False
        return all(matches_accepted(value[key], accepted[key]) for key in required)
    return value == accepted


def matches_accepted(value, accepted):
    values = accepted if isinstance(accepted, list) else [accepted]
    non_empty = [candidate for candidate in values if candidate != ""]
    if not non_empty:
        return value in ("", None)
    return any(value_matches(value, candidate) for candidate in non_empty)


def single_call_success(prediction, ground_truth):
    if prediction.get("name") != ground_truth["name"]:
        return False
    required = {
        key
        for key, accepted in ground_truth["accepted_arguments"].items()
        if any(has_concrete(candidate) for candidate in (accepted if isinstance(accepted, list) else [accepted]))
    }
    arguments = prediction.get("arguments") or {}
    if set(arguments) != required:
        return False
    return all(matches_accepted(arguments[key], ground_truth["accepted_arguments"][key]) for key in required)


def multiset_equal(left, right):
    return Counter(left) == Counter(right)


def parallel_call_success(predictions, ground_truth_calls):
    if len(predictions) != len(ground_truth_calls):
        return False
    used = [False] * len(predictions)

    def backtrack(gt_index):
        if gt_index == len(ground_truth_calls):
            return True
        gt = ground_truth_calls[gt_index]
        for pred_index, prediction in enumerate(predictions):
            if used[pred_index]:
                continue
            if single_call_success(prediction, gt):
                used[pred_index] = True
                if backtrack(gt_index + 1):
                    return True
                used[pred_index] = False
        return False

    return backtrack(0)


def parallel_router_success(predictions, ground_truth_calls):
    return len(predictions) == len(ground_truth_calls) and multiset_equal(
        [prediction.get("name") for prediction in predictions],
        [gt["name"] for gt in ground_truth_calls],
    )


def render_parallel_prompt(row):
    return PARALLEL_PROMPT_TEMPLATE.format(
        query=row.get("query", ""),
        function_doc=json.dumps(row.get("function_pool", []), ensure_ascii=False),
    )
