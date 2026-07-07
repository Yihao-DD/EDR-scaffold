"""Config loading: configs/base.json + stage overlay + optional local override.

Precedence (later wins): base.json < <stage>.json < local.json (git-ignored).
`local.json` may only override machine-specific keys (model paths, GPU knobs);
seeds/recipes/thresholds in base/stage files are preregistered constants.
"""

from __future__ import annotations

import os

from edr.io_utils import read_json
from edr.paths import CONFIG_DIR


def _merge(base, overlay):
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(stage=None):
    config = read_json(CONFIG_DIR / "base.json")
    if stage:
        config = _merge(config, read_json(CONFIG_DIR / f"{stage}.json"))
    local = CONFIG_DIR / "local.json"
    if local.exists():
        config = _merge(config, read_json(local))
    return config


def resolve_model_id(config):
    return os.environ.get("EDR_MODEL_ID") or config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
