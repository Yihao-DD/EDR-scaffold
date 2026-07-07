import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a human-review sample for v17 parallel judge")
    parser.add_argument("--input", default="results/v17_sibling_base_eval.json")
    parser.add_argument("--output", default="logs/v17_parallel_judge_manual_audit.md")
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    base = json.load(open(args.input, encoding="utf-8"))
    records = [row for row in base["records"] if row.get("segment") == "parallel_family"]
    buckets = {}
    for row in records:
        key = (row.get("category"), bool(row.get("call_success")))
        buckets.setdefault(key, []).append(row)

    selected = []
    for key in sorted(buckets):
        rows = sorted(
            buckets[key],
            key=lambda row: hashlib.sha256(("v17-audit:" + row["episode_id"]).encode("utf-8")).hexdigest(),
        )
        selected.extend(rows[:2])
    selected = selected[: args.samples]

    lines = [
        "# v17 parallel judge manual audit sample",
        "",
        "Scope: differential judge sanity check for parallel-family sibling arena.",
        "Status: PENDING HUMAN CONFIRMATION before judge freeze.",
        "",
    ]
    for index, row in enumerate(selected, 1):
        raw = (row.get("raw_model_output") or "").replace("\n", " ")[:700]
        lines.extend(
            [
                f"## {index}. {row['episode_id']} ({row['category']})",
                f"- judge_call_success: {row.get('call_success')}",
                f"- router_success: {row.get('router_success')}",
                f"- parse_error: {row.get('parse_error')}",
                f"- predicted_calls: `{json.dumps(row.get('predicted_calls'), ensure_ascii=False)}`",
                f"- ground_truth_calls: `{json.dumps(row.get('ground_truth_calls'), ensure_ascii=False)}`",
                f"- raw_output_prefix: `{raw}`",
                "- human_verdict: TODO",
                "",
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
