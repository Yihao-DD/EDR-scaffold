"""Single source of truth for repository paths.

Layout contract:
- frozen inputs live under  data/round1/  (shipped with the repo),
- fetched large artifacts under data/adapters/ (git-ignored, see
  tools/fetch_adapters.py),
- EVERYTHING written at run time lands under outputs/:
    outputs/ablations/<arm>/ ...   five ablation/control arms
    outputs/round2/ ...            second EDR iteration
    outputs/reconcile/ ...         round-1 acceptance re-evaluation
    outputs/logs/ + outputs/state.json   scheduler ledger
No code writes anywhere else.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
ROUND1_DIR = DATA_DIR / "round1"
EPISODE_ID_DIR = ROUND1_DIR / "episode_ids"
ADAPTER_DIR = DATA_DIR / "adapters"
CONFIG_DIR = REPO_ROOT / "configs"

OUTPUT_DIR = REPO_ROOT / "outputs"
ABLATION_OUT = OUTPUT_DIR / "ablations"
ROUND2_OUT = OUTPUT_DIR / "round2"
RECONCILE_OUT = OUTPUT_DIR / "reconcile"
LOG_DIR = OUTPUT_DIR / "logs"
STATE_FILE = OUTPUT_DIR / "state.json"

# Frozen round-1 inputs (shipped)
BASE_FAILURES = ROUND1_DIR / "base_failures.json"           # base-model episodes + failure records
H1_EVOLUTION = ROUND1_DIR / "evolution_h1.json"             # frozen round-1 evolve-loop output (H1 patches)
DISTILL_CORE = ROUND1_DIR / "distill_core.jsonl"            # 148 scaffold-taught core rows
DISTILL_TRAIN = ROUND1_DIR / "distill_train.jsonl"          # 444 rows = core + 2:1 replay (the A3 training set)
HELDOUT_PARTITION = ROUND1_DIR / "heldout_pass16_partition.json"
TRAINVAL_PARTITION = ROUND1_DIR / "train_val_pass16_partition.json"
TEACHER_HELDOUT_REFERENCE = ROUND1_DIR / "teacher_heldout_reference.json"
SIBLING_ARENA = ROUND1_DIR / "sibling_arena.json"
REFERENCE_EVALS = ROUND1_DIR / "reference_evals"   # frozen A3 (round-1) heldout/sibling evals, 5 seeds
ROUND1_MANIFEST = ROUND1_DIR / "MANIFEST.json"


def resolve(path):
    """Resolve a possibly repo-relative path to an absolute Path."""

    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
