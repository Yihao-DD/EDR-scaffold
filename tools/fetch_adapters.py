#!/usr/bin/env python3
"""Fetch the round-1 LoRA adapters from HF Hub and verify SHA256.

The adapters (5 x ~160MB) are not tracked in git. Their SHA256 hashes and the
hosting HF repo id are recorded in data/round1/MANIFEST.json. Usage:

  python3 tools/fetch_adapters.py            # download + verify all
  python3 tools/fetch_adapters.py --verify   # verify already-downloaded files only

If the manifest's hf_repo is null (not yet uploaded), the script prints the
expected local layout so the adapters can be placed manually.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edr.io_utils import read_json, sha256_file  # noqa: E402
from edr.paths import ADAPTER_DIR, ROUND1_MANIFEST  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify local files only; do not download.")
    args = parser.parse_args(argv)

    manifest = read_json(ROUND1_MANIFEST)
    adapters = manifest["adapters"]
    hf_repo = adapters.get("hf_repo")

    if not args.verify:
        if not hf_repo:
            print("adapters.hf_repo is not set in data/round1/MANIFEST.json.")
            print("Place the adapter files manually at:")
            for item in adapters["files"]:
                print(f"  {ADAPTER_DIR / item['path']}")
            return 1
        from huggingface_hub import snapshot_download

        print(f"downloading adapters from {hf_repo} ...")
        snapshot_download(repo_id=hf_repo, local_dir=str(ADAPTER_DIR), repo_type="model")

    failures = []
    for item in adapters["files"]:
        path = ADAPTER_DIR / item["path"]
        if not path.exists():
            failures.append((item["path"], "MISSING"))
            continue
        digest = sha256_file(path)
        if digest != item["sha256"]:
            failures.append((item["path"], f"SHA256 mismatch: {digest}"))
        else:
            print(f"[ok] {item['path']}")
    if failures:
        for path, reason in failures:
            print(f"[FAIL] {path}: {reason}")
        return 1
    print("all adapters verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
