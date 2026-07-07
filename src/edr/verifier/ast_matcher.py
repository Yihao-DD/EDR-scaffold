"""BFCL AST matching: parse model output, judge against ground truth.

Ported verbatim from the frozen round-1 evolution loop. The matching
semantics are part of the preregistered protocol — do not modify.
"""

from __future__ import annotations

import ast
import json
import re


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_bfcl_call(call):
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
    return {"name": _call_name(expression.func), "arguments": arguments}


def parse_model_output(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "name" in obj and "arguments" in obj:
                return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
            if len(obj) == 1:
                name, arguments = next(iter(obj.items()))
                if isinstance(arguments, dict):
                    return {"name": name, "arguments": arguments}
    match = re.search(r"[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)", text, re.S)
    if match:
        return parse_bfcl_call(match.group(0))
    raise ValueError("No parseable function call found")


def _has_concrete(value):
    if value == "":
        return False
    if isinstance(value, dict):
        return any(_has_concrete(nested) for nested in value.values())
    return True


def _matches_accepted(value, accepted):
    values = accepted if isinstance(accepted, list) else [accepted]
    non_empty = [candidate for candidate in values if candidate != ""]
    if not non_empty:
        return value in ("", None)
    return any(_value_matches(value, candidate) for candidate in non_empty)


def _value_matches(value, accepted):
    if isinstance(accepted, dict):
        if not isinstance(value, dict):
            return False
        required = {
            key
            for key, nested in accepted.items()
            if any(_has_concrete(v) for v in (nested if isinstance(nested, list) else [nested]))
        }
        if set(value) != required:
            return False
        return all(_matches_accepted(value[key], accepted[key]) for key in required)
    return value == accepted


def router_success(prediction, ground_truth):
    try:
        return parse_bfcl_call(prediction)["name"] == ground_truth["name"]
    except Exception:
        return False


def validator_success(prediction, ground_truth):
    try:
        parsed = parse_bfcl_call(prediction)
        if parsed["name"] != ground_truth["name"]:
            return False
        required = {
            key
            for key, accepted in ground_truth["accepted_arguments"].items()
            if any(_has_concrete(value) for value in (accepted if isinstance(accepted, list) else [accepted]))
        }
        if set(parsed["arguments"]) != required:
            return False
        return all(_matches_accepted(parsed["arguments"][key], ground_truth["accepted_arguments"][key]) for key in required)
    except Exception:
        return False


def call_success(prediction, ground_truth):
    return router_success(prediction, ground_truth) and validator_success(prediction, ground_truth)
