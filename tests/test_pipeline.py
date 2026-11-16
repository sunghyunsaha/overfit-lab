"""End-to-end smoke test: every experiment runs on a tiny synthetic configuration."""

from __future__ import annotations

import json

import pytest

from overfitlab import cli
from overfitlab.config import load_config


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("run")
    return load_config(
        overrides={
            "output_dir": str(out / "reports"),
            "artifacts_dir": str(out / "artifacts"),
            "n_jobs": 1,
            "data": {"source": "synthetic"},
            "synthetic": {"n_assets": 1, "n_bars": 700},
            "diagnostics": {
                "cscv_blocks": 6,
                "bootstrap": {"n_boot": 100},
                "trials": {"n_sim": 300},
                "walkforward": {"train_bars": 250, "test_bars": 60},
                "cpcv": {"n_groups": 6},
            },
            "experiments": {
                "e1": {
                    "mc": {
                        "n_worlds": 2,
                        "kinds": ["gbm"],
                        "n_assets": 1,
                        "n_bars": 500,
                        "n_boot": 80,
                        "cscv_blocks": 6,
                    }
                },
                "e2": {
                    "strengths": [0.0, 4.0],
                    "n_worlds": 2,
                    "n_assets": 1,
                    "n_bars": 500,
                    "n_boot": 80,
                    "cscv_blocks": 6,
                },
                "e4": {"fees_bps": [0, 30]},
                "e5": {
                    "cscv_blocks": [6],
                    "cpcv_groups": [6],
                    "embargo_pcts": [0.0],
                    "mean_blocks": [5],
                    "n_subperiods": 2,
                },
            },
        }
    )


@pytest.mark.parametrize("name", ["e1", "e2", "e3", "e4", "e5"])
def test_experiment_runs_and_writes_results_and_figures(tiny_cfg, name):
    cli._run_experiment(name, tiny_cfg)
    out = tiny_cfg["output_dir"]
    results = json.loads(open(f"{out}/results/{name}.json", encoding="utf-8").read())
    assert "tables" in results
    import pathlib

    assert list(pathlib.Path(out, "figures").glob(f"{name}_*.png"))


def test_synthetic_stand_in_is_labelled_synthetic(tiny_cfg):
    results = json.loads(open(f"{tiny_cfg['output_dir']}/results/e3.json", encoding="utf-8").read())
    assert results["data"]["source"] == "synthetic" and not results["data"]["is_real"]
    assert "SYNTHETIC" in results["data"]["label"]
    assert results["n_variants"] == 500
    # Net is never better than gross for the same variant set.
    assert results["net"]["sharpe"]["mean_annual"] <= results["gross"]["sharpe"]["mean_annual"]
