#!/usr/bin/env python3
"""(Re)build data/round1/MANIFEST.json: SHA256 inventory of every shipped
frozen file + the adapter hashes + the designated-M1 record.

Run after any change to data/round1/ (there should be none — the files are
frozen) or when the adapters' HF repo id is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edr.io_utils import read_json, sha256_file, write_json  # noqa: E402
from edr.paths import ADAPTER_DIR, ROUND1_DIR, ROUND1_MANIFEST  # noqa: E402

ADAPTER_SEEDS = [20260703, 20260704, 20260705, 20260706, 20260707]
ADAPTER_ARMS = {
    "round1": "main (scaffold-taught), recipe r16/lr5e-5/ep3/replay2:1/KLλ2",
    "round1_star": "star (self-sampled A4 contrast), recipe r16/lr1e-4/ep3/replay2:1/KLλ2 (per-arm base lr, preregistered symmetric rule)",
}
STAR_SEED03_NOTE = (
    "star seed20260703 is the reconciled recovered-v2 artifact; its dry-run vs final heldout "
    "evaluation was verified per-episode identical (round-1 audit), unlike the quarantined rep1 line."
)
M1_SEED = 20260704


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default=None, help="HF Hub repo id hosting the adapters (kept if omitted).")
    args = parser.parse_args(argv)

    previous = read_json(ROUND1_MANIFEST) if ROUND1_MANIFEST.exists() else {}
    hf_repo = args.hf_repo or (previous.get("adapters") or {}).get("hf_repo")

    files = []
    for path in sorted(ROUND1_DIR.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": str(path.relative_to(ROUND1_DIR)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    adapter_files = []
    previous_adapter_files = {item["path"]: item for item in (previous.get("adapters") or {}).get("files", [])}
    for arm_dir, arm_note in ADAPTER_ARMS.items():
        for seed in ADAPTER_SEEDS:
            rel = f"{arm_dir}/seed{seed}/adapter_model.safetensors"
            local = ADAPTER_DIR / rel
            entry = {"path": rel, "arm": arm_dir, "seed": seed}
            if local.exists():
                entry.update({"size": local.stat().st_size, "sha256": sha256_file(local)})
            elif rel in previous_adapter_files:
                entry = previous_adapter_files[rel]
            else:
                entry.update({"size": None, "sha256": None, "note": "hash pending: adapter not present locally"})
            if arm_dir == "round1_star" and seed == 20260703:
                entry["provenance_note"] = STAR_SEED03_NOTE
            adapter_files.append(entry)

    manifest = {
        "probe": "round1_manifest",
        "m1_designated": {
            "seed": M1_SEED,
            "adapter_dir": f"data/adapters/round1/seed{M1_SEED}",
            "reference": {"expected_heldout_repair_count": 51, "expected_sibling_success_count": 292},
            "rule": "median heldout repair over the five round-1 seeds; ties take the smaller seed",
        },
        "adapters": {"hf_repo": hf_repo, "arms": ADAPTER_ARMS, "files": adapter_files},
        "files": files,
    }
    write_json(ROUND1_MANIFEST, manifest)
    print(f"wrote {ROUND1_MANIFEST}: {len(files)} data files, {len(adapter_files)} adapters, hf_repo={hf_repo}")


if __name__ == "__main__":
    main()
