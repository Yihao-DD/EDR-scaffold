#!/usr/bin/env python3
"""Build MANIFEST files for the rep2 handoff package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = [20260703, 20260704, 20260705, 20260706, 20260707]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_count(payload: dict, candidates: list[str]) -> int | None:
    for key in candidates:
        value = payload
        ok = True
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                ok = False
                break
        if ok and isinstance(value, int):
            return value
    return None


def heldout_count(payload: dict) -> int:
    value = metric_count(
        payload,
        [
            "repair_success_count",
            "heldout_repair_count",
            "success_count",
            "summary.repair_success_count",
            "summary.success_count",
        ],
    )
    if value is not None:
        return value
    if "records" in payload:
        return sum(1 for row in payload["records"] if row.get("success") or row.get("repair_success"))
    if "episodes" in payload:
        return sum(1 for row in payload["episodes"] if row.get("success") or row.get("repair_success"))
    raise ValueError("could not infer heldout repair count")


def sibling_success_count(payload: dict) -> int:
    value = metric_count(
        payload,
        [
            "success_count",
            "sibling_success_count",
            "summary.success",
            "summary.success_count",
            "arena_success_count",
        ],
    )
    if value is not None:
        return value
    if "records" in payload:
        return sum(1 for row in payload["records"] if row.get("success") or row.get("base_success_preserved"))
    raise ValueError("could not infer sibling success count")


def file_info(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", default="MANIFEST.json")
    parser.add_argument("--output-md", default="MANIFEST.md")
    parser.add_argument("--sha256sums", default="SHA256SUMS")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    package_root = root.parent
    artifacts = []
    for seed in SEEDS:
        seed_dir = root / "artifacts" / f"main_rep2_seed{seed}"
        adapter_file = seed_dir / "adapter" / "adapter_model.safetensors"
        required = [adapter_file, seed_dir / "train_eval.json", seed_dir / "heldout.json", seed_dir / "sibling.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError({"seed": seed, "missing": missing})
        heldout = json.loads((seed_dir / "heldout.json").read_text())
        sibling = json.loads((seed_dir / "sibling.json").read_text())
        record = {
            "seed": seed,
            "config": "main rep2: rank16 lr5e-5 ep3 replay2 capped KLlambda2",
            "adapter": file_info(adapter_file, package_root),
            "train_eval_json": file_info(seed_dir / "train_eval.json", package_root),
            "heldout_json": file_info(seed_dir / "heldout.json", package_root),
            "sibling_json": file_info(seed_dir / "sibling.json", package_root),
            "heldout_repair_count": heldout_count(heldout),
            "heldout_total": 158,
            "sibling_success_count": sibling_success_count(sibling),
            "sibling_total": 300,
        }
        artifacts.append(record)

    sorted_by_repair = sorted(artifacts, key=lambda item: (item["heldout_repair_count"], item["seed"]))
    m1 = sorted_by_repair[len(sorted_by_repair) // 2]
    m1 = dict(m1)
    m1["selection_rule"] = "median heldout repair among main rep2 five seeds; ties choose smaller seed"
    m1["reference"] = {
        "heldout_repair_count": m1["heldout_repair_count"],
        "expected_heldout_repair_count": m1["heldout_repair_count"],
        "sibling_success_count": m1["sibling_success_count"],
        "expected_sibling_success_count": m1["sibling_success_count"],
    }

    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {args.output_json, args.output_md, args.sha256sums}:
            files.append(file_info(path, package_root))

    manifest = {
        "probe": "rep2_handoff_manifest",
        "m1_designated": m1,
        "artifacts": artifacts,
        "files": files,
    }
    (root / args.output_json).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with (root / args.sha256sums).open("w", encoding="utf-8") as handle:
        for item in files:
            handle.write(f"{item['sha256']}  {item['path']}\n")

    lines = [
        "# Rep2 Manifest",
        "",
        "## M1_DESIGNATED",
        "",
        f"- seed: `{m1['seed']}`",
        f"- rule: {m1['selection_rule']}",
        f"- adapter: `{m1['adapter']['path']}`",
        f"- adapter_sha256: `{m1['adapter']['sha256']}`",
        f"- heldout_repair: `{m1['heldout_repair_count']}/158`",
        f"- sibling_success: `{m1['sibling_success_count']}/300`",
        "",
        "## Seeds",
        "",
        "| seed | heldout repair | sibling success | adapter sha256 |",
        "|---:|---:|---:|---|",
    ]
    for item in artifacts:
        lines.append(
            f"| {item['seed']} | {item['heldout_repair_count']}/158 | "
            f"{item['sibling_success_count']}/300 | `{item['adapter']['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## SHA256",
            "",
            "Full file checksums are in `SHA256SUMS`. Adapter binaries are tracked through Git LFS.",
            "",
        ]
    )
    (root / args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"M1_DESIGNATED seed={m1['seed']} heldout={m1['heldout_repair_count']}/158 sibling={m1['sibling_success_count']}/300")


if __name__ == "__main__":
    main()
