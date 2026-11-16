"""Shared fixtures."""

from __future__ import annotations

import pytest

from overfitlab.config import load_config
from overfitlab.data.synthetic import simulate_market


@pytest.fixture(scope="session")
def small_cfg() -> dict:
    """Full strategy grid on a small synthetic market."""
    return load_config(
        overrides={
            "data": {"source": "synthetic"},
            "synthetic": {"n_assets": 2, "n_bars": 700},
        }
    )


@pytest.fixture(scope="session")
def small_prices() -> dict:
    return simulate_market(700, 2, seed=7, kind="garch", asset_corr=0.3)
