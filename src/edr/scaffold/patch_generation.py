"""Patch text generation (NL/STRUCT/NL-padded information-unit machinery).

Verbatim port of the frozen round-1 patch_pair_probe module."""

import argparse
import json
import re
import statistics
from pathlib import Path


PADDING_UNITS = [
    "Please pause briefly before continuing.",
    "Maintain a neutral posture.",
    "Proceed with careful neutral consideration.",
    "Keep the wording concise and orderly.",
    "Pause.",
    ".",
]


def _accepted_to_display(value):
    if isinstance(value, list):
        non_empty = [item for item in value if item != ""]
        return non_empty if non_empty else value
    return value


def _parameter_type(schema):
    if not isinstance(schema, dict):
        return "unknown"
    return schema.get("type", "unknown")


def _parameter_description(schema):
    if not isinstance(schema, dict):
        return ""
    return schema.get("description", "")


def _function_schema(function_pool, function_name):
    for function in function_pool:
        if isinstance(function, dict) and function.get("name") == function_name:
            return function
    return function_pool[0] if function_pool else {}


def _candidate_functions(function_pool):
    candidates = []
    for function in function_pool:
        if not isinstance(function, dict):
            continue
        candidates.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
            }
        )
    return candidates


def _safe_iu_suffix(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _has_format_requirement(param_name, schema):
    text = f"{param_name} {schema.get('type', '') if isinstance(schema, dict) else ''} {schema.get('description', '') if isinstance(schema, dict) else ''}"
    has_textual_format = bool(re.search(r"\b(format|iso|yyyy|hh:mm|date|time|enum)\b", text, re.I))
    has_enum_schema = isinstance(schema, dict) and bool(schema.get("enum"))
    return has_textual_format or has_enum_schema


def _format_requirement(param_name, schema):
    if not isinstance(schema, dict):
        return f"Use the required format for {param_name}."
    description = schema.get("description", "")
    param_type = schema.get("type", "unknown")
    return f"{param_name} must follow type {param_type}; schema note: {description}".strip()


def _validator_parameters(failure):
    gt = failure.get("ground_truth_call", {})
    function = _function_schema(failure.get("function_pool", []), gt.get("name"))
    properties = (((function.get("parameters") or {}).get("properties")) or {}) if isinstance(function, dict) else {}
    accepted = gt.get("accepted_arguments", {})
    parameters = []
    for name, values in accepted.items():
        schema = properties.get(name, {})
        parameters.append(
            {
                "name": name,
                "type": _parameter_type(schema),
                "description": _parameter_description(schema),
                "accepted_values": _accepted_to_display(values),
                "predicted_value": (failure.get("predicted_call") or {}).get("arguments", {}).get(name, "<missing>"),
            }
        )
    return parameters


def generate_iu_set(failure, diagnosis=None):
    """Extract auditable Information Units shared by STRUCT and NL renderers."""
    failure_class = failure.get("failure_class")
    if failure_class == "router_fail":
        candidate_functions = _candidate_functions(failure.get("function_pool", []))
        gt_function = failure.get("ground_truth_call", {}).get("name", "")
        selected = (failure.get("predicted_call") or {}).get("name", "<parse_error>")
        units = [
            {
                "id": "IU-intent",
                "kind": "intent",
                "content": {
                    "user_query": failure.get("query", ""),
                    "intent_description": failure.get("query", ""),
                },
            }
        ]
        for function in candidate_functions:
            units.append(
                {
                    "id": f"IU-fn-{_safe_iu_suffix(function['name'])}",
                    "kind": "candidate_function",
                    "content": {
                        "name": function["name"],
                        "description": function["description"],
                    },
                }
            )
        units.append(
            {
                "id": "IU-match",
                "kind": "match",
                "content": {
                    "model_selected": selected,
                    "gt_function": gt_function,
                    "rationale": "The best function is the one whose description matches the user intent.",
                },
            }
        )
        return units

    gt = failure.get("ground_truth_call", {})
    function = _function_schema(failure.get("function_pool", []), gt.get("name"))
    properties = (((function.get("parameters") or {}).get("properties")) or {}) if isinstance(function, dict) else {}
    units = []
    for param in _validator_parameters(failure):
        schema = properties.get(param["name"], {})
        units.append(
            {
                "id": f"IU-param-{_safe_iu_suffix(param['name'])}",
                "kind": "parameter",
                "content": {
                    "function_name": gt.get("name", ""),
                    "user_query": failure.get("query", ""),
                    "name": param["name"],
                    "type": param["type"],
                    "description": param["description"],
                    "accepted_values": param["accepted_values"],
                    "predicted_value": param["predicted_value"],
                    "extraction_rule": (
                        f"Extract {param['name']} as type {param['type']} using the query and schema; "
                        f"accepted value(s): {param['accepted_values']}; predicted value: {param['predicted_value']}."
                    ),
                },
            }
        )
        if _has_format_requirement(param["name"], schema):
            units.append(
                {
                    "id": f"IU-format-{_safe_iu_suffix(param['name'])}",
                    "kind": "format",
                    "content": {
                        "parameter": param["name"],
                        "requirement": _format_requirement(param["name"], schema),
                    },
                }
            )
    return units


def _iu_by_kind(iu_set, kind):
    return [unit for unit in iu_set if unit["kind"] == kind]


def _iu_ids(iu_set):
    return [unit["id"] for unit in iu_set]


def build_information_bundle(failure):
    iu_set = generate_iu_set(failure)
    base = {
        "episode_id": failure.get("episode_id"),
        "split": failure.get("split"),
        "failure_class": failure.get("failure_class"),
        "user_query": failure.get("query", ""),
        "iu_set": iu_set,
        "iu_ids": _iu_ids(iu_set),
    }
    if failure.get("failure_class") == "router_fail":
        base.update(
            {
                "candidate_functions": _candidate_functions(failure.get("function_pool", [])),
                "model_selected": (failure.get("predicted_call") or {}).get("name", "<parse_error>"),
                "gt_function": failure.get("ground_truth_call", {}).get("name", ""),
                "repair_rule": "Choose the candidate function whose description best matches the user intent.",
            }
        )
    else:
        base.update(
            {
                "function_name": failure.get("ground_truth_call", {}).get("name", ""),
                "parameters": _validator_parameters(failure),
                "repair_rule": "Fill every required parameter with the value and type indicated by the query and schema.",
                "gt_arguments": failure.get("ground_truth_call", {}).get("accepted_arguments", {}),
                "predicted_arguments": (failure.get("predicted_call") or {}).get("arguments", {}),
            }
        )
    return base


def render_struct_patch(iu_set, failure_class):
    if failure_class == "router_fail":
        intent = _iu_by_kind(iu_set, "intent")[0]["content"]
        functions = _iu_by_kind(iu_set, "candidate_function")
        match = _iu_by_kind(iu_set, "match")[0]["content"]
        lines = [
            "[STEP 1] User wants to:",
            f"  {intent['intent_description']}",
            "[STEP 2] Candidate functions and what each does:",
        ]
        for unit in functions:
            function = unit["content"]
            lines.append(f"  - name: {function['name']} | does: {function['description']}")
        lines.extend(
            [
                f"[STEP 3] Model selected: {match['model_selected']}",
                f"[STEP 4] Best match: {match['gt_function']}",
                f"[WHY] {match['rationale']}",
            ]
        )
        return {"family": "STRUCT", "text": "\n".join(lines), "covered_iu_ids": _iu_ids(iu_set)}
    params = _iu_by_kind(iu_set, "parameter")
    formats = _iu_by_kind(iu_set, "format")
    function_name = params[0]["content"]["function_name"] if params else ""
    query = params[0]["content"]["user_query"] if params else ""
    lines = [
        f"[FUNCTION] {function_name}",
        f"[QUERY] {query}",
        "[PARAMETERS]",
    ]
    for index, unit in enumerate(params, start=1):
        param = unit["content"]
        lines.append(
            f"[PARAM {index}] name: {param['name']} | type: {param['type']} | description: {param['description']} | "
            f"accepted: {param['accepted_values']} | predicted: {param['predicted_value']} | rule: {param['extraction_rule']}"
        )
    if formats:
        lines.append("[FORMAT REQUIREMENTS]")
        for unit in formats:
            content = unit["content"]
            lines.append(f"[FORMAT] parameter: {content['parameter']} | requirement: {content['requirement']}")
    return {"family": "STRUCT", "text": "\n".join(lines), "covered_iu_ids": _iu_ids(iu_set)}


def render_nl_patch(iu_set, failure_class):
    if failure_class == "router_fail":
        intent = _iu_by_kind(iu_set, "intent")[0]["content"]
        functions = _iu_by_kind(iu_set, "candidate_function")
        match = _iu_by_kind(iu_set, "match")[0]["content"]
        candidate_text = "; ".join(
            f"{unit['content']['name']} does {unit['content']['description']}" for unit in functions
        )
        text = (
            f"For the user query {intent['intent_description']!r}, compare these candidate functions: {candidate_text}. "
            f"The model selected {match['model_selected']}, while the best matching function is {match['gt_function']} because "
            f"{match['rationale']}"
        )
        return {"family": "NL", "text": text, "covered_iu_ids": _iu_ids(iu_set)}
    params = _iu_by_kind(iu_set, "parameter")
    formats = _iu_by_kind(iu_set, "format")
    param_text = "; ".join(
        f"{unit['content']['name']} has type {unit['content']['type']}, description {unit['content']['description']!r}, "
        f"accepted value {unit['content']['accepted_values']}, predicted value {unit['content']['predicted_value']}, and rule "
        f"{unit['content']['extraction_rule']}"
        for unit in params
    )
    format_text = " ".join(
        f"Format requirement for {unit['content']['parameter']}: {unit['content']['requirement']}." for unit in formats
    )
    function_name = params[0]["content"]["function_name"] if params else ""
    query = params[0]["content"]["user_query"] if params else ""
    text = f"For function {function_name} and query {query!r}, extract the parameters as follows: {param_text}."
    if format_text:
        text = f"{text} {format_text}"
    return {"family": "NL", "text": text, "covered_iu_ids": _iu_ids(iu_set)}


def render_struct(bundle):
    return render_struct_patch(bundle["iu_set"], bundle["failure_class"])["text"]


def render_nl(bundle):
    return render_nl_patch(bundle["iu_set"], bundle["failure_class"])["text"]


def count_tokens(text, tokenizer=None):
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _meaningful_terms(text):
    return {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_./:-]*", str(text)) if len(term) >= 3}


def _split_identifier_terms(text):
    terms = set()
    for term in _meaningful_terms(text):
        terms.add(term)
        for piece in re.split(r"[^A-Za-z0-9]+|_", term):
            if len(piece) >= 3:
                terms.add(piece.lower())
    return terms


def _collect_forbidden_padding_terms(iu_set):
    forbidden = set()
    type_words = {"string", "integer", "float", "number", "boolean", "dict", "array", "object", "list"}
    for unit in iu_set:
        content = unit.get("content", {})
        if unit["kind"] == "intent":
            forbidden.update(_meaningful_terms(content.get("intent_description", "")))
        elif unit["kind"] == "candidate_function":
            forbidden.update(_split_identifier_terms(content.get("name", "")))
            forbidden.update(_meaningful_terms(content.get("description", "")))
        elif unit["kind"] == "match":
            forbidden.update(_split_identifier_terms(content.get("model_selected", "")))
            forbidden.update(_split_identifier_terms(content.get("gt_function", "")))
            forbidden.update({"function", "intent", "match", "selected", "best"})
        elif unit["kind"] == "parameter":
            forbidden.update(_split_identifier_terms(content.get("function_name", "")))
            forbidden.update(_split_identifier_terms(content.get("name", "")))
            forbidden.update(_meaningful_terms(content.get("type", "")))
            forbidden.update(_meaningful_terms(content.get("description", "")))
            forbidden.update(_meaningful_terms(content.get("accepted_values", "")))
            forbidden.update(_meaningful_terms(content.get("predicted_value", "")))
            forbidden.update(type_words)
        elif unit["kind"] == "format":
            forbidden.update(_split_identifier_terms(content.get("parameter", "")))
            forbidden.update(_meaningful_terms(content.get("requirement", "")))
            forbidden.update(type_words)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "your",
        "you",
        "are",
        "can",
        "has",
        "have",
        "value",
        "values",
        "query",
        "user",
        "schema",
        "description",
        "required",
        "optional",
    }
    return forbidden - stopwords


def assert_no_info_padding(padding_text, iu_set):
    padding_lower = padding_text.lower()
    hits = []
    for term in sorted(_collect_forbidden_padding_terms(iu_set), key=len, reverse=True):
        if not term:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", padding_lower):
            hits.append(term)
    assert not hits, {"padding_text": padding_text, "forbidden_hits": hits[:20]}
    return True


def _padding_unit_is_safe(unit, forbidden_terms):
    unit_lower = unit.lower()
    for term in forbidden_terms:
        if not term:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", unit_lower):
            return False
    return True


def extract_patch_iu_ids(patch):
    return list(patch.get("covered_iu_ids", []))


def assert_iu_parity(struct_patch, nl_patch):
    struct_ids = extract_patch_iu_ids(struct_patch)
    nl_ids = extract_patch_iu_ids(nl_patch)
    assert struct_ids == nl_ids, {"struct": struct_ids, "nl": nl_ids}
    return True


def render_nl_padded_patch(iu_set, failure_class, target_token_count, tokenizer=None, tolerance=0.05):
    nl_patch = render_nl_patch(iu_set, failure_class)
    base_text = nl_patch["text"]
    base_count = count_tokens(base_text, tokenizer)
    lower = int(target_token_count * (1 - tolerance) + 0.999999)
    upper = int(target_token_count * (1 + tolerance))
    padding_parts = []
    current = base_count
    forbidden_terms = _collect_forbidden_padding_terms(iu_set)
    safe_units = [unit for unit in PADDING_UNITS if _padding_unit_is_safe(unit, forbidden_terms)]
    if "." not in safe_units:
        safe_units.append(".")
    unit_counts = [(unit, count_tokens(unit, tokenizer)) for unit in safe_units]
    unit_counts = [(unit, count) for unit, count in unit_counts if count > 0]
    guard = 0
    while current < lower and guard < 1000:
        guard += 1
        remaining = upper - current
        candidates = [(unit, count) for unit, count in unit_counts if count <= remaining]
        if not candidates:
            break
        unit, count = max(candidates, key=lambda item: item[1])
        padding_parts.append(unit)
        current += count
    padding_text = " ".join(padding_parts)
    if padding_text:
        assert_no_info_padding(padding_text, iu_set)
    full_text = f"{base_text}\n\n{padding_text}" if padding_text else base_text
    return {
        "family": "NL-padded",
        "text": full_text,
        "base_text": base_text,
        "padding_text": padding_text,
        "covered_iu_ids": _iu_ids(iu_set),
        "target_token_count": target_token_count,
        "base_token_count": base_count,
        "padding_token_count": count_tokens(padding_text, tokenizer) if padding_text else 0,
        "token_count": count_tokens(full_text, tokenizer),
    }


def assert_three_way_parity(struct_patch, nl_patch, nl_padded_patch, target_token_count, tokenizer=None, tolerance=0.05):
    struct_ids = extract_patch_iu_ids(struct_patch)
    nl_ids = extract_patch_iu_ids(nl_patch)
    padded_ids = extract_patch_iu_ids(nl_padded_patch)
    assert struct_ids == nl_ids == padded_ids, {"struct": struct_ids, "nl": nl_ids, "nl_padded": padded_ids}
    padded_count = count_tokens(nl_padded_patch["text"], tokenizer)
    lower = target_token_count * (1 - tolerance)
    upper = target_token_count * (1 + tolerance)
    assert lower <= padded_count <= upper, {
        "target": target_token_count,
        "padded": padded_count,
        "lower": lower,
        "upper": upper,
    }
    assert_no_info_padding(nl_padded_patch.get("padding_text", ""), nl_padded_patch.get("iu_set", []))
    return True


def build_pair(failure, tokenizer=None):
    bundle = build_information_bundle(failure)
    struct_patch_record = render_struct_patch(bundle["iu_set"], bundle["failure_class"])
    nl_patch_record = render_nl_patch(bundle["iu_set"], bundle["failure_class"])
    iu_parity = assert_iu_parity(struct_patch_record, nl_patch_record)
    struct_patch = struct_patch_record["text"]
    nl_patch = nl_patch_record["text"]
    struct_count = count_tokens(struct_patch, tokenizer)
    nl_count = count_tokens(nl_patch, tokenizer)
    nl_padded_patch_record = render_nl_padded_patch(
        bundle["iu_set"],
        bundle["failure_class"],
        target_token_count=struct_count,
        tokenizer=tokenizer,
    )
    nl_padded_patch_record["iu_set"] = bundle["iu_set"]
    three_way_parity = assert_three_way_parity(
        struct_patch_record,
        nl_patch_record,
        nl_padded_patch_record,
        target_token_count=struct_count,
        tokenizer=tokenizer,
    )
    nl_padded_count = count_tokens(nl_padded_patch_record["text"], tokenizer)
    ratio = struct_count / nl_count if nl_count else None
    padded_ratio = nl_padded_count / struct_count if struct_count else None
    return {
        "episode_id": failure.get("episode_id"),
        "failure_class": failure.get("failure_class"),
        "information_items": bundle,
        "iu_set": bundle["iu_set"],
        "struct_patch_record": struct_patch_record,
        "nl_patch_record": nl_patch_record,
        "nl_padded_patch_record": nl_padded_patch_record,
        "struct_iu_ids": extract_patch_iu_ids(struct_patch_record),
        "nl_iu_ids": extract_patch_iu_ids(nl_patch_record),
        "nl_padded_iu_ids": extract_patch_iu_ids(nl_padded_patch_record),
        "iu_parity": iu_parity,
        "three_way_parity": three_way_parity,
        "struct_patch": struct_patch,
        "nl_patch": nl_patch,
        "nl_padded_patch": nl_padded_patch_record["text"],
        "nl_padded_padding_text": nl_padded_patch_record["padding_text"],
        "struct_token_count": struct_count,
        "nl_token_count": nl_count,
        "nl_padded_token_count": nl_padded_count,
        "nl_padded_padding_token_count": nl_padded_patch_record["padding_token_count"],
        "token_count_delta_abs": abs(struct_count - nl_count),
        "struct_to_nl_token_ratio": ratio,
        "nl_padded_to_struct_ratio": padded_ratio,
        "same_information_source": True,
        "padding_no_info": assert_no_info_padding(nl_padded_patch_record["padding_text"], bundle["iu_set"]),
    }


def select_probe_failures(failures, n=20, min_per_class=None):
    multiple = [item for item in failures if item.get("split") == "multiple"]
    router = [item for item in multiple if item.get("failure_class") == "router_fail"]
    validator = [item for item in multiple if item.get("failure_class") == "validator_fail"]
    router.sort(key=lambda item: (len(item.get("function_pool", [])), item.get("episode_id", "")))
    validator.sort(key=lambda item: (len(item.get("function_pool", [])), item.get("episode_id", "")))
    if min_per_class is None:
        min_per_class = min(10, n // 2) if n >= 20 else (max(1, min(2, n // 2)) if router and validator else 0)
    router_take = min(len(router), min_per_class)
    validator_take = min(len(validator), min_per_class)
    selected = router[:router_take]
    selected.extend(validator[:validator_take])
    remaining = n - len(selected)
    if remaining > 0:
        leftovers = router[router_take:] + validator[validator_take:]
        leftovers.sort(key=lambda item: (item.get("failure_class", ""), len(item.get("function_pool", [])), item.get("episode_id", "")))
        selected.extend(leftovers[:remaining])
    if len(selected) < n:
        selected.extend([item for item in multiple if item not in selected][: n - len(selected)])
    return selected[:n]


def _stat(values):
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def summarize_pairs(pairs):
    struct = [pair["struct_token_count"] for pair in pairs]
    nl = [pair["nl_token_count"] for pair in pairs]
    padded = [pair["nl_padded_token_count"] for pair in pairs]
    ratios = [pair["struct_to_nl_token_ratio"] for pair in pairs if pair["struct_to_nl_token_ratio"] is not None]
    padded_ratios = [pair["nl_padded_to_struct_ratio"] for pair in pairs if pair["nl_padded_to_struct_ratio"] is not None]
    deltas = [pair["token_count_delta_abs"] for pair in pairs]
    by_class = {}
    for pair in pairs:
        bucket = by_class.setdefault(
            pair["failure_class"],
            {"n": 0, "struct_tokens": [], "nl_tokens": [], "padded_tokens": [], "ratios": [], "padded_ratios": [], "deltas": []},
        )
        bucket["n"] += 1
        bucket["struct_tokens"].append(pair["struct_token_count"])
        bucket["nl_tokens"].append(pair["nl_token_count"])
        bucket["padded_tokens"].append(pair["nl_padded_token_count"])
        bucket["ratios"].append(pair["struct_to_nl_token_ratio"])
        bucket["padded_ratios"].append(pair["nl_padded_to_struct_ratio"])
        bucket["deltas"].append(pair["token_count_delta_abs"])
    parity_count = sum(1 for pair in pairs if pair.get("iu_parity"))
    three_way_count = sum(1 for pair in pairs if pair.get("three_way_parity"))
    padding_no_info_count = sum(1 for pair in pairs if pair.get("padding_no_info"))
    return {
        "n_pairs": len(pairs),
        "struct_token_count": _stat(struct),
        "nl_token_count": _stat(nl),
        "nl_padded_token_count": _stat(padded),
        "struct_to_nl_token_ratio": _stat(ratios),
        "nl_padded_to_struct_ratio": _stat(padded_ratios),
        "mean_abs_delta": sum(deltas) / len(deltas) if deltas else None,
        "iu_parity_pass_rate": parity_count / len(pairs) if pairs else 0.0,
        "three_way_parity_pass_rate": three_way_count / len(pairs) if pairs else 0.0,
        "padding_no_info_pass_rate": padding_no_info_count / len(pairs) if pairs else 0.0,
        "by_failure_class": {
            key: {
                "n": value["n"],
                "struct_token_count": _stat(value["struct_tokens"]),
                "nl_token_count": _stat(value["nl_tokens"]),
                "nl_padded_token_count": _stat(value["padded_tokens"]),
                "struct_to_nl_token_ratio": _stat(value["ratios"]),
                "nl_padded_to_struct_ratio": _stat(value["padded_ratios"]),
                "mean_abs_delta": sum(value["deltas"]) / len(value["deltas"]) if value["deltas"] else None,
            }
            for key, value in by_class.items()
        },
    }


def load_tokenizer(model_id, cache_dir):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    except Exception:
        return None


def build_probe_result(input_path, n=20, tokenizer=None, min_per_class=None):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    failures = data.get("failures", [])
    selected = select_probe_failures(failures, n=n, min_per_class=min_per_class)
    pairs = [build_pair(failure, tokenizer=tokenizer) for failure in selected]
    return {
        "source": str(input_path),
        "tokenizer": "qwen" if tokenizer is not None else "regex_fallback",
        "summary": summarize_pairs(pairs),
        "pairs": pairs,
        "asserts": {
            "n_pairs_requested": len(pairs) == n,
            "all_pairs_same_information_source": all(pair["same_information_source"] for pair in pairs),
            "all_iu_parity_asserts_passed": all(pair["iu_parity"] for pair in pairs),
            "all_three_way_parity_asserts_passed": all(pair["three_way_parity"] for pair in pairs),
            "all_padding_no_info_asserts_passed": all(pair["padding_no_info"] for pair in pairs),
            "contains_router_pair": any(pair["failure_class"] == "router_fail" for pair in pairs),
            "contains_validator_pair": any(pair["failure_class"] == "validator_fail" for pair in pairs),
            "router_pairs_ge_min": sum(pair["failure_class"] == "router_fail" for pair in pairs) >= (min_per_class or 0),
            "validator_pairs_ge_min": sum(pair["failure_class"] == "validator_fail" for pair in pairs) >= (min_per_class or 0),
        },
    }


def write_markdown(result, output_path):
    lines = [
        "# STRUCT vs NL Pairing Probe",
        "",
        f"PROBE signal (n={result['summary']['n_pairs']}). This step checks whether paired STRUCT/NL patch text can be generated from the same information bundle.",
        "",
        "## Token Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| STRUCT mean tokens | {result['summary']['struct_token_count']['mean']:.3f} |",
        f"| NL mean tokens | {result['summary']['nl_token_count']['mean']:.3f} |",
        f"| NL-padded mean tokens | {result['summary']['nl_padded_token_count']['mean']:.3f} |",
        f"| STRUCT/NL ratio mean | {result['summary']['struct_to_nl_token_ratio']['mean']:.3f} |",
        f"| STRUCT/NL ratio std | {result['summary']['struct_to_nl_token_ratio']['std']:.3f} |",
        f"| NL-padded/STRUCT ratio mean | {result['summary']['nl_padded_to_struct_ratio']['mean']:.3f} |",
        f"| mean absolute token delta | {result['summary']['mean_abs_delta']:.3f} |",
        f"| IU parity pass rate | {result['summary']['iu_parity_pass_rate']:.3f} |",
        f"| three-way parity pass rate | {result['summary']['three_way_parity_pass_rate']:.3f} |",
        f"| padding no-info pass rate | {result['summary']['padding_no_info_pass_rate']:.3f} |",
        "",
        "## Ratio By Failure Class",
        "",
        "| failure class | n | STRUCT/NL ratio mean | STRUCT/NL ratio std | NL-padded/STRUCT ratio mean | mean absolute token delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in result["summary"]["by_failure_class"].items():
        lines.append(
            f"| {key} | {value['n']} | {value['struct_to_nl_token_ratio']['mean']:.3f} | "
            f"{value['struct_to_nl_token_ratio']['std']:.3f} | {value['nl_padded_to_struct_ratio']['mean']:.3f} | "
            f"{value['mean_abs_delta']:.3f} |"
        )
    router_stats = result["summary"]["by_failure_class"].get("router_fail")
    validator_stats = result["summary"]["by_failure_class"].get("validator_fail")
    if router_stats and validator_stats:
        router_ratio = router_stats["struct_to_nl_token_ratio"]["mean"]
        validator_ratio = validator_stats["struct_to_nl_token_ratio"]["mean"]
        ratio_delta = router_ratio - validator_ratio
        lines.extend(
            [
                "",
                "## Router/Validator Ratio Difference",
                "",
                f"PROBE signal (n={result['summary']['n_pairs']}). Router pair mean ratio is {router_ratio:.3f}; "
                f"validator pair mean ratio is {validator_ratio:.3f}; difference is {ratio_delta:.3f}.",
            ]
        )
    lines.extend(
        [
        "",
        "## Sample Pairs",
        ]
    )
    for index, pair in enumerate(result["pairs"], start=1):
        iu_ids = ", ".join(pair["struct_iu_ids"])
        lines.extend(
            [
                "",
                f"### Pair {index}: `{pair['episode_id']}` ({pair['failure_class']})",
                "",
                f"- STRUCT tokens: {pair['struct_token_count']}",
                f"- NL tokens: {pair['nl_token_count']}",
                f"- NL-padded tokens: {pair['nl_padded_token_count']}",
                f"- padding tokens: {pair['nl_padded_padding_token_count']}",
                f"- ratio: {pair['struct_to_nl_token_ratio']:.3f}",
                f"- NL-padded/STRUCT ratio: {pair['nl_padded_to_struct_ratio']:.3f}",
                f"- padding no-info: {'T' if pair['padding_no_info'] else 'F'}",
                f"- IU ids: `{iu_ids}`",
                "",
                "STRUCT:",
                "",
                "```text",
                pair["struct_patch"],
                "```",
                "",
                "NL:",
                "",
                "```text",
                pair["nl_patch"],
                "```",
                "",
                "NL-padded:",
                "",
                "```text",
                pair["nl_padded_patch"],
                "```",
                "",
                "Padding segment:",
                "",
                "```text",
                pair["nl_padded_padding_text"],
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Assert Results",
            "",
            "| assert | result |",
            "|---|---|",
        ]
    )
    for key, value in result["asserts"].items():
        lines.append(f"| `{key}` | {'T' if value else 'F'} |")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="STRUCT/NL information-parity pairing probe")
    parser.add_argument("--input", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--output", default="results/pairing_probe.json")
    parser.add_argument("--log", default="logs/pairing_probe.md")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--regex-tokenizer", action="store_true")
    args = parser.parse_args(argv)
    tokenizer = None if args.regex_tokenizer else load_tokenizer(args.model_id, args.model_cache_dir)
    result = build_probe_result(args.input, n=args.n, tokenizer=tokenizer, min_per_class=args.min_per_class)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, args.log)
    print(json.dumps({"summary": result["summary"], "asserts": result["asserts"], "tokenizer": result["tokenizer"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
