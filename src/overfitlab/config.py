"""Configuration loading.

Every run is described by ``configs/base.yaml`` deep-merged with one override file.
Keeping all knobs in YAML (rather than in code) is what makes a run reproducible:
the config plus the seed fully determine the output.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` (neither input is mutated)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load ``base.yaml``, merge the file at ``path`` (if any), then ``overrides``."""
    with open(BASE_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_config_path"] = str(path) if path is not None else str(BASE_CONFIG)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a directory entry of the config relative to the repository root."""
    p = Path(cfg[key])
    return p if p.is_absolute() else REPO_ROOT / p


def n_jobs(cfg: dict) -> int:
    """Number of worker processes to use (``n_jobs: 0`` means all available cores)."""
    n = int(cfg.get("n_jobs", 0))
    return n if n > 0 else max(1, os.cpu_count() or 1)
