"""The diagnostic pipeline shared by all experiments.

``diagnose`` takes one T x N matrix of daily strategy returns and runs every overfitting
diagnostic on it, returning a JSON-serialisable summary plus the arrays the figures
need. Every experiment is a different way of producing the matrix (null world, planted
signal, real data, other fees, other periods) followed by this one function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from overfitlab.config import resolve_path
from overfitlab.diagnostics import sharpe as sh
from overfitlab.diagnostics.bootstrap import reality_check_and_spa
from overfitlab.diagnostics.cv import cpcv_select_best
from overfitlab.diagnostics.multiple import haircut_sharpe, survivors
from overfitlab.diagnostics.pbo import cscv
from overfitlab.diagnostics.trials import effective_trials
from overfitlab.diagnostics.walkforward import split_decay, walk_forward


def derive_seed(seed: int, stream: int) -> int:
    """An independent seed for sub-stream ``stream`` of a run seeded with ``seed``.

    Worlds in the Monte Carlo are seeded with consecutive integers. Using ``seed + 1``
    for the bootstrap of world w would hand it the very random stream that generates
    world w + 1; hashing ``(seed, stream)`` through ``SeedSequence`` keeps every stream
    statistically independent of every other.
    """
    return int(np.random.SeedSequence([int(seed), int(stream)]).generate_state(1)[0])


def one_sided_pvalues(moments: dict, n_obs: int) -> np.ndarray:
    """p-value of H0: true Sharpe <= 0 for each variant (see ``sharpe_pvalue``).

    Variants that never trade (zero variance) get p = 1.
    """
    return np.where(moments["live"], sh.sharpe_pvalue(moments["sharpe"], n_obs), 1.0)


def psr_pvalues(moments: dict, n_obs: int) -> np.ndarray:
    """``1 - PSR(0)`` for each variant: kept to document why it is NOT used (see E1)."""
    psr0 = sh.probabilistic_sharpe_ratio(moments["sharpe"], n_obs, moments["skew"], moments["kurt"])
    return np.where(moments["live"], 1.0 - psr0, 1.0)


def diagnose(
    returns: np.ndarray,
    cfg: dict,
    *,
    seed: int,
    heavy: bool = True,
    neff_methods: tuple[str, ...] | None = None,
    n_boot: int | None = None,
    cscv_blocks: int | None = None,
    mean_block: float | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the diagnostics on a T x N return matrix.

    ``heavy=False`` skips the walk-forward, IS/OOS decay, CPCV, MinBTL and haircut parts
    (used inside Monte Carlo loops). Returns ``(summary, arrays)``.
    """
    dcfg = cfg["diagnostics"]
    alpha = dcfg["alpha"]
    ppy = cfg["periods_per_year"]
    ann = np.sqrt(ppy)
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    years = t / ppy

    m = sh.return_moments(x)
    live = m["live"]
    sr = m["sharpe"]
    p = one_sided_pvalues(m, t)
    best = int(np.argmax(sr))
    sr_std = float(np.std(sr[live], ddof=1)) if live.sum() > 1 else 0.0

    out: dict[str, Any] = {
        "n_obs": t,
        "years": years,
        "n_variants": n,
        "n_never_traded": int(n - live.sum()),
        "alpha": alpha,
        "sharpe": {
            "best_annual": float(sr[best] * ann),
            "best_index": best,
            "median_annual": float(np.median(sr[live]) * ann),
            "mean_annual": float(sr[live].mean() * ann),
            "cross_sectional_std_annual": sr_std * ann,
            "frac_positive": float(np.mean(sr[live] > 0)),
        },
        "best": {
            "skew": float(m["skew"][best]),
            "kurt": float(m["kurt"][best]),
            "psr0": float(1.0 - psr_pvalues(m, t)[best]),
            "p_value": float(p[best]),
            "min_trl_years": sh.min_track_record_length(
                float(sr[best]), float(m["skew"][best]), float(m["kurt"][best]), 0.0, alpha
            )
            / ppy,
        },
        "survivors": survivors(p, alpha),
    }
    out["survivors"]["naive_frac"] = out["survivors"]["naive"] / n
    p_psr = psr_pvalues(m, t)
    out["survivors_psr"] = survivors(p_psr, alpha)
    out["survivors_psr"]["most_significant_active_bars"] = int(
        np.count_nonzero(x[:, int(np.argmin(p_psr))])
    )

    # Effective number of trials and the Deflated Sharpe Ratio under each estimate.
    tcfg = dcfg["trials"]
    methods = tuple(neff_methods if neff_methods is not None else tcfg["methods"])
    neff = effective_trials(x, methods, n_sim=tcfg["n_sim"], seed=derive_seed(seed, 2))
    out["neff"] = neff
    primary = tcfg["primary"] if tcfg["primary"] in neff else methods[0]
    dsr = {}
    for name in ("raw", *methods):
        value, sr0 = sh.deflated_sharpe_ratio(
            float(sr[best]), t, float(m["skew"][best]), float(m["kurt"][best]), neff[name], sr_std
        )
        dsr[name] = {"dsr": value, "sr0_annual": sr0 * ann, "n_trials": neff[name]}
    out["dsr"] = dsr
    out["dsr_primary"] = {"method": primary, **dsr[primary]}
    # Sensitivity: the same DSR with the THEORETICAL null dispersion of Sharpe estimates,
    # 1/sqrt(T-1), instead of the cross-sectional dispersion the paper prescribes (which
    # also contains genuine differences between variants, e.g. cost drag).
    value, sr0 = sh.deflated_sharpe_ratio(
        float(sr[best]),
        t,
        float(m["skew"][best]),
        float(m["kurt"][best]),
        neff[primary],
        1.0 / np.sqrt(t - 1.0),
    )
    out["dsr_null_dispersion"] = {"dsr": value, "sr0_annual": sr0 * ann}

    # Family-level data-snooping tests.
    bcfg = dcfg["bootstrap"]
    snoop = reality_check_and_spa(
        x,
        n_boot=n_boot or bcfg["n_boot"],
        mean_block=mean_block or bcfg["mean_block"],
        seed=derive_seed(seed, 1),
    )
    out["snooping"] = {
        "rc_p": snoop.rc_pvalue,
        "spa_lower_p": snoop.spa_lower,
        "spa_p": snoop.spa_consistent,
        "spa_upper_p": snoop.spa_upper,
        "spa_hac_p": snoop.spa_hac_consistent,
        "hac_scale_ratio": snoop.hac_scale_ratio,
        "best_t_index": snoop.best_t_index,
        "n_boot": snoop.n_boot,
        "mean_block": snoop.mean_block,
    }

    # Probability of backtest overfitting.
    pbo = cscv(x, cscv_blocks or dcfg["cscv_blocks"])
    slope, _ = pbo.degradation
    out["pbo"] = {
        "pbo": pbo.pbo,
        "n_blocks": pbo.n_blocks,
        "n_splits": pbo.n_splits,
        "prob_oos_loss": pbo.prob_oos_loss,
        "degradation_slope": slope,
        "median_logit": float(np.median(pbo.logits)),
        "mean_is_annual": float(pbo.is_sharpe.mean() * ann),
        "mean_oos_annual": float(pbo.oos_sharpe.mean() * ann),
    }
    arrays = {
        "sharpe_annual": sr * ann,
        "p": p,
        "live": live,
        "logits": pbo.logits,
        "pbo_is_annual": pbo.is_sharpe * ann,
        "pbo_oos_annual": pbo.oos_sharpe * ann,
    }
    if not heavy:
        return out, arrays

    decay = split_decay(x)
    out["decay"] = {
        "slope": decay["slope"],
        "rank_corr": decay["rank_corr"],
        "best_is_annual": decay["best_is"] * ann,
        "best_oos_annual": decay["best_oos"] * ann,
        "best_oos_percentile": decay["best_oos_percentile"],
        "top50_is_annual": decay["top_is_mean"] * ann,
        "top50_oos_annual": decay["top_oos_mean"] * ann,
        "median_oos_annual": decay["median_oos"] * ann,
    }
    arrays["decay_is_annual"] = decay["sr_is"] * ann
    arrays["decay_oos_annual"] = decay["sr_oos"] * ann

    wcfg = dcfg["walkforward"]
    wf = walk_forward(x, wcfg["train_bars"], wcfg["test_bars"])
    out["walkforward"] = {
        "n_windows": wf["n_windows"],
        "train_bars": wcfg["train_bars"],
        "test_bars": wcfg["test_bars"],
        "mean_is_annual": wf["mean_is_sharpe"] * ann,
        "oos_annual": wf["oos_sharpe"] * ann,
        "frac_windows_oos_negative": wf["frac_windows_oos_negative"],
        "n_distinct_picks": int(len(np.unique(wf["picks"]))),
    }
    arrays["wf_oos_returns"] = wf["oos_returns"]
    arrays["wf_covered"] = wf["covered"]
    arrays["best_returns"] = x[:, best]

    ccfg = dcfg["cpcv"]
    embargo = int(round(ccfg["embargo_pct"] * t))
    cp = cpcv_select_best(
        x, ccfg["n_groups"], ccfg["n_test_groups"], ccfg["label_horizon"], embargo
    )
    out["cpcv"] = cpcv_summary(cp, ann)
    out["cpcv"].update(
        n_groups=ccfg["n_groups"],
        n_test_groups=ccfg["n_test_groups"],
        label_horizon=ccfg["label_horizon"],
        embargo_bars=embargo,
    )
    arrays["cpcv_path_annual"] = cp["path_sharpe"] * ann

    # Minimum backtest length: how many years would be needed before the observed best
    # Sharpe ratio stops being what pure luck is expected to deliver?
    best_annual = out["sharpe"]["best_annual"]
    n_primary = neff[primary]
    out["min_btl"] = {
        "years_available": years,
        "years_needed_primary": sh.min_backtest_length(n_primary, best_annual),
        "years_needed_raw": sh.min_backtest_length(neff["raw"], best_annual),
        "expected_max_annual_primary": sh.expected_max_z(n_primary) / np.sqrt(years),
        "expected_max_annual_raw": sh.expected_max_z(neff["raw"]) / np.sqrt(years),
        "max_trials_allowed": sh.max_trials_for_length(years, best_annual),
    }

    hc = haircut_sharpe(best_annual, t, ppy, all_sr_annual=(sr * ann)[live])
    hc["bonferroni_neff"] = haircut_sharpe(best_annual, t, ppy, n_tests=n_primary)["bonferroni"]
    out["haircut"] = hc
    return out, arrays


def cpcv_summary(cp: dict, ann: float) -> dict:
    """Summary statistics of a ``cpcv_select_best`` result (annualised Sharpe ratios)."""
    paths = cp["path_sharpe"] * ann
    return {
        "n_splits": cp["n_splits"],
        "n_paths": cp["n_paths"],
        "mean_is_annual": float(cp["split_is_sharpe"].mean() * ann),
        "mean_split_oos_annual": float(cp["split_oos_sharpe"].mean() * ann),
        "median_path_annual": float(np.median(paths)),
        "mean_path_annual": float(paths.mean()),
        "min_path_annual": float(paths.min()),
        "max_path_annual": float(paths.max()),
        "frac_paths_negative": float(np.mean(paths < 0)),
        "n_distinct_picks": int(len(np.unique(cp["chosen"]))),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_results(cfg: dict, name: str, results: dict) -> Path:
    """Write ``<output_dir>/results/<name>.json`` and return its path."""
    path = resolve_path(cfg, "output_dir") / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    return path


def save_arrays(cfg: dict, name: str, arrays: dict) -> Path:
    """Store the arrays a figure needs in ``<artifacts_dir>/<name>_arrays.npz``.

    Nested dicts are flattened with ``__``; a DatetimeIndex is stored as ISO strings.
    Keeping these on disk means every figure can be redrawn without re-running anything.
    """
    flat = {}
    for key, value in arrays.items():
        if isinstance(value, dict):
            flat.update({f"{key}__{k}": np.asarray(v) for k, v in value.items()})
        elif hasattr(value, "strftime"):
            flat[key] = np.asarray(value.strftime("%Y-%m-%d"), dtype=str)
        else:
            flat[key] = np.asarray(value)
    path = resolve_path(cfg, "artifacts_dir") / f"{name}_arrays.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    return path


def load_arrays(cfg: dict, name: str) -> dict:
    """Inverse of ``save_arrays`` (nested keys are rebuilt)."""
    path = resolve_path(cfg, "artifacts_dir") / f"{name}_arrays.npz"
    out: dict = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if "__" in key:
                outer, inner = key.split("__", 1)
                out.setdefault(outer, {})[inner] = data[key]
            else:
                out[key] = data[key]
    return out


def load_results(cfg: dict, name: str) -> dict:
    path = resolve_path(cfg, "output_dir") / "results" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def figure_dir(cfg: dict) -> Path:
    path = resolve_path(cfg, "output_dir") / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe_variant(meta, index: int) -> dict:
    """Human-readable description of one column of the universe."""
    row = meta.iloc[int(index)]
    return {
        "variant_id": int(row["variant_id"]),
        "asset": str(row["asset"]),
        "family": str(row["family"]),
        "params": str(row["params"]),
        "vol_target": bool(row["vol_target"]),
        "label": f"{row['asset']} {row['params']}{' +voltarget' if row['vol_target'] else ''}",
    }
