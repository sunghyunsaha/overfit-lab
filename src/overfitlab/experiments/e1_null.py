"""E1 - Null world: validate the tools where the truth is known.

On synthetic markets with NO signal every strategy has a true gross Sharpe ratio of
exactly zero, so every "discovery" is a false positive. Two parts:

1. One full-size null world (same shape as the real-data run): how many of the 5,000
   variants look significant naively, and how many survive each correction?
2. A Monte Carlo over many smaller null worlds: the false-positive RATE of each method,
   with confidence intervals, for both null types (GBM and GARCH with fat tails). A
   method that is working rejects in at most ~alpha of the worlds.

The Monte Carlo is run on GROSS returns: with costs every true Sharpe ratio is negative
and the null holds with room to spare, so gross returns are the harder test.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from overfitlab.backtest.universe import build_universe
from overfitlab.config import deep_merge, n_jobs
from overfitlab.data.synthetic import simulate_market, synthetic_kwargs
from overfitlab.experiments.common import diagnose, save_results, wilson_interval


def world_statistics(
    cfg: dict, synth: dict, mc: dict, seed: int, which: str = "gross", keep_sharpe: bool = False
) -> dict:
    """Simulate one small world, backtest the universe, run the light diagnostics."""
    prices = simulate_market(
        mc["n_bars"],
        mc["n_assets"],
        seed=seed,
        **synthetic_kwargs(synth, cfg["periods_per_year"]),
    )
    uni = build_universe(prices, cfg)
    x = uni.gross if which == "gross" else uni.net()
    out, arrays = diagnose(
        x, cfg, seed=seed, heavy=False, n_boot=mc["n_boot"], cscv_blocks=mc["cscv_blocks"]
    )
    alpha = out["alpha"]
    extra = {"sharpe_annual": arrays["sharpe_annual"]} if keep_sharpe else {}
    return {
        **extra,
        "seed": seed,
        "n_variants": out["n_variants"],
        "naive_frac": out["survivors"]["naive_frac"],
        "any_naive": out["survivors"]["naive"] > 0,
        "any_bonferroni_psr": out["survivors_psr"]["bonferroni"] > 0,
        "psr_offender_active": (
            out["survivors_psr"]["most_significant_active_bars"]
            if out["survivors_psr"]["bonferroni"] > 0
            else None
        ),
        "spa_hac_scale_ratio": (
            out["snooping"]["hac_scale_ratio"] if out["snooping"]["spa_hac_p"] < alpha else None
        ),
        "any_bonferroni": out["survivors"]["bonferroni"] > 0,
        "any_holm": out["survivors"]["holm"] > 0,
        "any_bh": out["survivors"]["bh"] > 0,
        "any_by": out["survivors"]["by"] > 0,
        "n_bh": out["survivors"]["bh"],
        "rc": out["snooping"]["rc_p"] < alpha,
        "spa": out["snooping"]["spa_p"] < alpha,
        "spa_hac": out["snooping"]["spa_hac_p"] < alpha,
        "spa_lower": out["snooping"]["spa_lower_p"] < alpha,
        "spa_upper": out["snooping"]["spa_upper_p"] < alpha,
        "psr_best": out["best"]["psr0"] > 1 - alpha,
        "dsr_null": out["dsr_null_dispersion"]["dsr"] > 1 - alpha,
        **{f"dsr_{k}": v["dsr"] > 1 - alpha for k, v in out["dsr"].items()},
        **{f"sr0_{k}": v["sr0_annual"] for k, v in out["dsr"].items()},
        **{f"neff_{k}": v["n_trials"] for k, v in out["dsr"].items()},
        "best_annual": out["sharpe"]["best_annual"],
        "pbo": out["pbo"]["pbo"],
        "rc_p": out["snooping"]["rc_p"],
        "spa_p": out["snooping"]["spa_p"],
        "dsr_value": out["dsr_primary"]["dsr"],
    }


def _worker(args: tuple) -> dict:
    return world_statistics(*args)


def run_worlds(tasks: list[tuple], workers: int) -> list[dict]:
    """Run ``world_statistics`` over many tasks, in parallel when workers > 1.

    Worker processes are started with ``spawn`` and single-threaded BLAS so that P
    processes use P cores (instead of P x cores threads fighting each other).
    """
    if workers <= 1 or len(tasks) < 2:
        return [_worker(t) for t in tasks]
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            return list(pool.map(_worker, tasks, chunksize=1))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


RATE_KEYS = (
    "any_naive",
    "psr_best",
    "any_bonferroni_psr",
    "any_bonferroni",
    "any_holm",
    "any_bh",
    "any_by",
    "rc",
    "spa",
    "spa_hac",
    "spa_lower",
    "spa_upper",
    "dsr_null",
)


def summarise_worlds(worlds: list[dict]) -> dict:
    """Rejection rates (with 95% Wilson intervals) and averages over worlds."""
    n = len(worlds)
    dsr_keys = sorted(k for k in worlds[0] if k.startswith("dsr_") and k != "dsr_value")
    keys = list(dict.fromkeys([*RATE_KEYS, *dsr_keys]))
    rates = {}
    for key in keys:
        k = int(sum(bool(w[key]) for w in worlds))
        lo, hi = wilson_interval(k, n)
        rates[key] = {"rate": k / n, "count": k, "lo": lo, "hi": hi}
    mean_keys = ["naive_frac", "best_annual", "pbo", "n_bh"]
    mean_keys += sorted(k for k in worlds[0] if k.startswith(("sr0_", "neff_")))
    means = {k: float(np.mean([w[k] for w in worlds])) for k in mean_keys}
    offenders = [w["psr_offender_active"] for w in worlds if w["psr_offender_active"] is not None]
    ratios = [w["spa_hac_scale_ratio"] for w in worlds if w["spa_hac_scale_ratio"] is not None]
    return {
        "n_worlds": n,
        "rates": rates,
        "means": means,
        "psr_offender_active_bars": offenders,
        "spa_hac_scale_ratios": ratios,
        "pbo_values": [w["pbo"] for w in worlds],
        "best_annual_values": [w["best_annual"] for w in worlds],
        "spa_p_values": [w["spa_p"] for w in worlds],
        "rc_p_values": [w["rc_p"] for w in worlds],
    }


MC_ROWS = [
    ("any_naive", "Naive: at least one variant with p < 0.05"),
    ("psr_best", "PSR(0) of the best variant above 0.95"),
    ("any_bonferroni_psr", "Bonferroni on 1 - PSR(0) (in-sample skewness plugged in)"),
    ("any_bonferroni", "Bonferroni"),
    ("any_holm", "Holm"),
    ("any_bh", "Benjamini-Hochberg"),
    ("any_by", "Benjamini-Yekutieli"),
    ("rc", "White's Reality Check"),
    ("spa_hac", "Hansen's SPA, long-run s.d. (as in the paper)"),
    ("spa", "Hansen's SPA, sample s.d. (used in this project)"),
    ("dsr_raw", "DSR > 0.95, N = raw count"),
    ("dsr_avg_corr", "DSR > 0.95, N from average correlation"),
    ("dsr_max_matching", "DSR > 0.95, N from max-matching (used in this project)"),
    ("dsr_cluster", "DSR > 0.95, N from clustering"),
    ("dsr_eigen", "DSR > 0.95, N from eigenvalues"),
    ("dsr_null", "DSR > 0.95, max-matching N, theoretical dispersion 1/sqrt(T-1)"),
]
NEFF_ROWS = [
    ("raw", "Raw count"),
    ("avg_corr", "Average correlation"),
    ("max_matching", "Max-matching simulation"),
    ("cluster", "Clustering (simplified ONC)"),
    ("eigen", "Eigenvalue participation ratio"),
]


def add_tables(results: dict) -> dict:
    """Attach Markdown tables (built only from the numbers in ``results``)."""
    from overfitlab.experiments.e3_real import family_level_table, survival_table
    from overfitlab.reporting.render import md_table

    mc = results["mc"]
    kinds = list(mc)
    titles = {"gbm": "GBM null", "garch": "GARCH fat-tailed null"}

    def cell(kind: str, key: str) -> str:
        r = mc[kind]["rates"][key]
        return f"{r['rate']:.1%} ({r['lo']:.1%} to {r['hi']:.1%})"

    rows = [
        [label, *[cell(k, key) for k in kinds]]
        for key, label in MC_ROWS
        if key in mc[kinds[0]]["rates"]
    ]
    calib = []
    for key, label in NEFF_ROWS:
        row = [label]
        for k in kinds:
            m = mc[k]["means"]
            row += [
                f"{m[f'neff_{key}']:,.0f}",
                f"{m[f'sr0_{key}']:.2f}",
                f"{m[f'sr0_{key}'] / m['best_annual'] - 1:+.0%}",
            ]
        calib.append(row)
    head = ["Estimator of N"]
    for k in kinds:
        name = titles.get(k, k)
        head += [f"{name}: mean N", f"{name}: predicted best Sharpe", f"{name}: error"]
    full = results["full"]
    results["tables"] = {
        "mc_md": md_table(["Method", *[titles.get(k, k) for k in kinds]], rows),
        "calibration_md": md_table(head, calib),
        "full_survival_md": survival_table(full["net"], full["gross"]),
        "full_family_md": family_level_table(full["net"], full["gross"]),
    }
    offenders = [v for k in kinds for v in mc[k]["psr_offender_active_bars"]] or [0]
    ratios = [v for k in kinds for v in mc[k]["spa_hac_scale_ratios"]] or [1.0]
    results["summary"] = {
        "psr_offender_n": len(offenders),
        "psr_offender_min": int(min(offenders)),
        "psr_offender_median": float(np.median(offenders)),
        "psr_offender_max": int(max(offenders)),
        "spa_hac_reject_n": len(ratios),
        "spa_hac_ratio_median": float(np.median(ratios)),
        "realised_best_gbm": mc[kinds[0]]["means"]["best_annual"],
        "realised_best_garch": mc[kinds[-1]]["means"]["best_annual"],
        "max_fwer_holm": max(mc[k]["rates"]["any_holm"]["rate"] for k in kinds),
        "max_fdr_bh": max(mc[k]["rates"]["any_bh"]["rate"] for k in kinds),
        "max_spa": max(mc[k]["rates"]["spa"]["rate"] for k in kinds),
        "max_spa_hac": max(mc[k]["rates"]["spa_hac"]["rate"] for k in kinds),
        "max_rc": max(mc[k]["rates"]["rc"]["rate"] for k in kinds),
        "max_dsr": max(mc[k]["rates"]["dsr_max_matching"]["rate"] for k in kinds),
        "max_dsr_cluster": max(mc[k]["rates"]["dsr_cluster"]["rate"] for k in kinds),
        "max_dsr_null": max(mc[k]["rates"]["dsr_null"]["rate"] for k in kinds),
        "max_bonferroni_psr": max(mc[k]["rates"]["any_bonferroni_psr"]["rate"] for k in kinds),
        "mean_pbo_min": min(mc[k]["means"]["pbo"] for k in kinds),
        "mean_pbo_max": max(mc[k]["means"]["pbo"] for k in kinds),
        "naive_frac_min": min(mc[k]["means"]["naive_frac"] for k in kinds),
        "naive_frac_max": max(mc[k]["means"]["naive_frac"] for k in kinds),
        "n_worlds_total": sum(mc[k]["n_worlds"] for k in kinds),
    }
    return results


def run(cfg: dict) -> dict:
    """Run E1 and write ``results/e1.json``."""
    ecfg = cfg["experiments"]["e1"]
    seed = cfg["seed"]

    # Part 1: one full-size null world.
    synth = deep_merge(cfg["synthetic"], {"signal_strength": 0.0, "drift_annual": 0.0})
    prices = simulate_market(
        synth["n_bars"],
        synth["n_assets"],
        seed=seed + 101,
        **synthetic_kwargs(synth, cfg["periods_per_year"]),
    )
    uni = build_universe(prices, cfg)
    gross, arrays_gross = diagnose(uni.gross, cfg, seed=seed + 101)
    net, arrays_net = diagnose(uni.net(), cfg, seed=seed + 101, heavy=False)
    full = {
        "kind": synth["kind"],
        "n_assets": synth["n_assets"],
        "n_bars": synth["n_bars"],
        "gross": gross,
        "net": net,
    }

    # Part 2: Monte Carlo over smaller null worlds.
    mc = ecfg["mc"]
    monte_carlo = {}
    for i, kind in enumerate(mc["kinds"]):
        synth_k = deep_merge(synth, {"kind": kind})
        tasks = [
            (cfg, synth_k, mc, seed + 10_000 * (i + 1) + w, "gross") for w in range(mc["n_worlds"])
        ]
        monte_carlo[kind] = summarise_worlds(run_worlds(tasks, n_jobs(cfg)))
        monte_carlo[kind]["n_variants"] = mc["n_assets"] * 500
        monte_carlo[kind]["n_bars"] = mc["n_bars"]

    results = {"full": full, "mc": monte_carlo, "alpha": cfg["diagnostics"]["alpha"]}
    save_results(cfg, "e1", add_tables(results))
    keep = ("sharpe_annual", "live", "logits")
    return {
        "results": results,
        "arrays": {
            "gross": {k: arrays_gross[k] for k in keep},
            "net": {k: arrays_net[k] for k in keep},
        },
    }
