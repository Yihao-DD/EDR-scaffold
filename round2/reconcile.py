#!/usr/bin/env python3
"""Company acceptance check for the rep2 handoff package."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import add_common_args, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--manifest", default="repro_rep2/MANIFEST.json")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    m1 = manifest["m1_designated"]
    heldout = m1["reference"]["heldout_repair_count"]
    sibling = m1["reference"]["sibling_success_count"]
    if not args.skip_pytest:
        subprocess.run(["python3", "-m", "pytest", "repro_rep2/tests"], check=True)
    if abs(heldout - m1["reference"]["expected_heldout_repair_count"]) > 1:
        raise SystemExit("heldout reference outside +/-1 episode")
    if abs(sibling - m1["reference"]["expected_sibling_success_count"]) > 2:
        raise SystemExit("sibling reference outside +/-2 episodes")
    print("ACCEPTED")


if __name__ == "__main__":
    main()
