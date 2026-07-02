import argparse
import json
import statistics
from pathlib import Path


DEFAULT_PAIRS = [
    ("main", "main_r8_lr2e-5_ep1", "main_r8_lr2e-5_ep2"),
    ("star", "star_r8_lr2e-5_ep1", "star_r8_lr2e-5_ep2"),
]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def mean_score(row):
    score = row.get("correct_score") or {}
    return score.get("mean_logprob")


def summarize_scores(rows):
    scores = [mean_score(row) for row in rows if mean_score(row) is not None]
    if not scores:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def summarize_values(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def validation_records(records, partition=None):
    rows = [row for row in records if row.get("target_family") == "validation_teacher_repaired"]
    if partition is not None:
        rows = [row for row in rows if row.get("partition") == partition]
    return rows


def dose_order(records, arm, ep1_name, ep2_name, partition=None):
    rows = validation_records(records, partition=partition)
    ep1 = []
    ep2_new = []
    not_ep2 = []
    for row in rows:
        success = row.get("grid_success", {})
        if success.get(ep1_name):
            ep1.append(row)
        elif success.get(ep2_name):
            ep2_new.append(row)
        else:
            not_ep2.append(row)
    return {
        "arm": arm,
        "partition": partition or "all_validation_teacher_repaired",
        "ep1_transferred": summarize_scores(ep1),
        "ep2_new_transferred": summarize_scores(ep2_new),
        "not_transferred_by_ep2": summarize_scores(not_ep2),
    }


def scaffold_transfer_contrast(records, arm, grid_name):
    rows = validation_records(records, partition="scaffold_only")
    transferred = [row for row in rows if row.get("grid_success", {}).get(grid_name)]
    failed = [row for row in rows if not row.get("grid_success", {}).get(grid_name)]
    return {
        "arm": arm,
        "grid_name": grid_name,
        "transferred": summarize_scores(transferred),
        "not_transferred": summarize_scores(failed),
    }


def margin_followup(records):
    r_success = [row for row in records if row.get("target_family") == "r_success_correct"]
    first_shared = [row for row in r_success if row.get("first_shared_fragile")]
    ep2_union = [row for row in r_success if row.get("ep2_union_fragile")]
    non_first_shared = [row for row in r_success if not row.get("first_shared_fragile")]
    correct_logprob = {
        "first_shared_fragile": summarize_scores(first_shared),
        "non_first_shared": summarize_scores(non_first_shared),
        "ep2_union_fragile": summarize_scores(ep2_union),
    }
    margin_sets = {}
    for grid_name in sorted({name for row in r_success for name in (row.get("margins") or {})}):
        margin_sets[grid_name] = summarize_values(
            [row.get("margins", {}).get(grid_name) for row in first_shared]
        )
    return {
        "correct_logprob": correct_logprob,
        "first_shared_actual_flip_margins": margin_sets,
        "note": (
            "Margin controls are unavailable for non-fragile episodes because no observed flipped-to "
            "output exists for them in the logged grid JSONs; this tests the registered fragile-24 "
            "actual-flip margin directly."
        ),
    }


def format_float(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_markdown(path, payload):
    lines = [
        "# s04 Log-Prob Follow-Ups",
        "",
        "Registered follow-up analyses computed from `results/s04_logprob_probe.json`; no additional GPU scoring was needed.",
        "",
        "## H-margin follow-up",
        "",
        "| group | n | mean | median |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, summary in payload["h_margin"]["correct_logprob"].items():
        lines.append(
            f"| {name} correct logprob | {summary['n']} | {format_float(summary['mean'])} | {format_float(summary['median'])} |"
        )
    for name, summary in payload["h_margin"]["first_shared_actual_flip_margins"].items():
        lines.append(
            f"| fragile24 margin {name} | {summary['n']} | {format_float(summary['mean'])} | {format_float(summary['median'])} |"
        )
    lines += ["", payload["h_margin"]["note"], "", "## Dose-order prediction", ""]
    lines += ["| arm | partition | ep1 transferred | ep2 new | not by ep2 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in payload["dose_order"]:
        lines.append(
            "| {arm} | {partition} | {a_n} / {a_mean} | {b_n} / {b_mean} | {c_n} / {c_mean} |".format(
                arm=row["arm"],
                partition=row["partition"],
                a_n=row["ep1_transferred"]["n"],
                a_mean=format_float(row["ep1_transferred"]["mean"]),
                b_n=row["ep2_new_transferred"]["n"],
                b_mean=format_float(row["ep2_new_transferred"]["mean"]),
                c_n=row["not_transferred_by_ep2"]["n"],
                c_mean=format_float(row["not_transferred_by_ep2"]["mean"]),
            )
        )
    lines += ["", "## Scaffold-only transfer success/failure", ""]
    lines += ["| arm | grid | transferred | not transferred |", "| --- | --- | ---: | ---: |"]
    for row in payload["scaffold_only_transfer_contrast"]:
        lines.append(
            "| {arm} | {grid} | {a_n} / {a_mean} | {b_n} / {b_mean} |".format(
                arm=row["arm"],
                grid=row["grid_name"],
                a_n=row["transferred"]["n"],
                a_mean=format_float(row["transferred"]["mean"]),
                b_n=row["not_transferred"]["n"],
                b_mean=format_float(row["not_transferred"]["mean"]),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize preregistered s04 log-prob follow-up analyses.")
    parser.add_argument("--input", default="results/s04_logprob_probe.json")
    parser.add_argument("--output", default="results/s04_logprob_followups.json")
    parser.add_argument("--markdown-output", default="logs/s04_logprob_followups.md")
    args = parser.parse_args(argv)

    payload = read_json(args.input)
    records = payload["records"]
    result = {
        "source": args.input,
        "h_margin": margin_followup(records),
        "dose_order": [],
        "scaffold_only_transfer_contrast": [],
    }
    for arm, ep1_name, ep2_name in DEFAULT_PAIRS:
        result["dose_order"].append(dose_order(records, arm, ep1_name, ep2_name))
        result["dose_order"].append(dose_order(records, arm, ep1_name, ep2_name, partition="scaffold_only"))
        result["scaffold_only_transfer_contrast"].append(scaffold_transfer_contrast(records, arm, ep1_name))
        result["scaffold_only_transfer_contrast"].append(scaffold_transfer_contrast(records, arm, ep2_name))
    write_json(args.output, result)
    write_markdown(args.markdown_output, result)
    print(f"wrote {args.output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
