import argparse
import json
from collections import Counter

from scripts.bfcl_common import write_json


HARNESS_LAYERS = ["tool_interface", "context", "verification", "orchestration", "logging", "validation", "generation"]


def _ordered_unique(values):
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _function_names(failures):
    names = []
    for failure in failures:
        names.append(failure.get("ground_truth_call", {}).get("name"))
        predicted = failure.get("predicted_call") or {}
        names.append(predicted.get("name"))
        names.extend(item.get("name") for item in failure.get("function_pool", []) if isinstance(item, dict))
    return _ordered_unique(names)


def _field_names(value):
    names = []
    if isinstance(value, dict):
        for key, nested in value.items():
            names.append(key)
            names.extend(_field_names(nested))
    elif isinstance(value, list):
        for item in value:
            names.extend(_field_names(item))
    return names


def _collect_fields(failures):
    fields = []
    for failure in failures:
        fields.extend(_field_names(failure.get("ground_truth_call", {}).get("accepted_arguments", {})))
        fields.extend(_field_names((failure.get("predicted_call") or {}).get("arguments", {})))
        fields.extend(_field_names(failure.get("function_pool", [])))
    return _ordered_unique(fields)


def _build_mapping(failures):
    functions = {name: f"fn_{index:04d}" for index, name in enumerate(_function_names(failures), start=1)}
    fields = {name: f"field_{index:04d}" for index, name in enumerate(_collect_fields(failures), start=1)}
    layers = {name: f"layer_{index:02d}" for index, name in enumerate(HARNESS_LAYERS, start=1)}
    errors = {
        name: f"err_{index:02d}"
        for index, name in enumerate(
            _ordered_unique(
                failure.get("failure_class")
                for failure in failures
            )
            + ["ParamError", "SchemaError", "FunctionError"],
            start=1,
        )
    }
    return {"function": functions, "field": fields, "layer": layers, "error": errors}


def _rename_keys(value, mapping):
    if isinstance(value, dict):
        renamed = {}
        for key, nested in value.items():
            new_key = mapping["field"].get(key, key)
            renamed[new_key] = _rename_keys(nested, mapping)
        return renamed
    if isinstance(value, list):
        return [_rename_keys(item, mapping) for item in value]
    return value


def _decouple_failure(failure, mapping):
    item = json.loads(json.dumps(failure, ensure_ascii=False))
    if item.get("ground_truth_call", {}).get("name") in mapping["function"]:
        item["ground_truth_call"]["name"] = mapping["function"][item["ground_truth_call"]["name"]]
    if item.get("predicted_call") and item["predicted_call"].get("name") in mapping["function"]:
        item["predicted_call"]["name"] = mapping["function"][item["predicted_call"]["name"]]
    for function in item.get("function_pool", []):
        if isinstance(function, dict) and function.get("name") in mapping["function"]:
            function["name"] = mapping["function"][function["name"]]
        if isinstance(function, dict) and "parameters" in function:
            function["parameters"] = _rename_keys(function["parameters"], mapping)
    if "accepted_arguments" in item.get("ground_truth_call", {}):
        item["ground_truth_call"]["accepted_arguments"] = _rename_keys(item["ground_truth_call"]["accepted_arguments"], mapping)
    if item.get("predicted_call") and "arguments" in item["predicted_call"]:
        item["predicted_call"]["arguments"] = _rename_keys(item["predicted_call"]["arguments"], mapping)
    trace = item.setdefault("trace", {})
    if "harness_layer" in trace:
        trace["harness_layer"] = mapping["layer"].get(trace["harness_layer"], trace["harness_layer"])
    if "error_label" in trace:
        trace["error_label"] = mapping["error"].get(trace["error_label"], trace["error_label"])
    if item.get("failure_class") in mapping["error"]:
        item["decoupled_failure_label"] = mapping["error"][item["failure_class"]]
    return item


def decouple_failures(failures):
    mapping = _build_mapping(failures)
    return [_decouple_failure(failure, mapping) for failure in failures], mapping


def _distribution(failures):
    return dict(Counter(failure.get("failure_class") for failure in failures))


def _is_bijective(mapping):
    return len(set(mapping.values())) == len(mapping)


def build_decoupled_result(failures):
    decoupled, mapping = decouple_failures(failures)
    original_function_names = set(mapping["function"])
    decoupled_function_names = set(mapping["function"].values())
    asserts = {
        "failure_class_distribution_preserved": _distribution(failures) == _distribution(decoupled),
        "function_mapping_bijective": _is_bijective(mapping["function"]),
        "field_mapping_bijective": _is_bijective(mapping["field"]),
        "layer_mapping_bijective": _is_bijective(mapping["layer"]),
        "error_mapping_bijective": _is_bijective(mapping["error"]),
        "no_decoupled_function_name_equals_original": original_function_names.isdisjoint(decoupled_function_names),
    }
    assert all(asserts.values()), asserts
    return {
        "n_failures": len(failures),
        "original_failure_class_distribution": _distribution(failures),
        "decoupled_failure_class_distribution": _distribution(decoupled),
        "mapping": mapping,
        "decoupled_failures": decoupled,
        "asserts": asserts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="A3 deterministic identifier decoupling")
    parser.add_argument("--input", default="results/a2_failures.json")
    parser.add_argument("--output", default="results/a3_decoupling.json")
    args = parser.parse_args(argv)
    data = json.loads(open(args.input, encoding="utf-8").read())
    result = build_decoupled_result(data.get("failures", []))
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ["n_failures", "original_failure_class_distribution", "decoupled_failure_class_distribution", "asserts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
