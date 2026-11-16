"""E2 - Power: how strong must a real signal be before each method detects it?

A hidden, slowly varying drift of adjustable strength is planted in a GARCH fat-tailed
market. Strength is measured as the annualised Sharpe ratio an ORACLE that sees the
hidden drift would earn; the strategies only see prices and capture part of it. For each
strength many worlds are simulated and each method's detection rate is recorded
("detection" = the method declares at least one variant, or the family, significant).

Net returns are used: the question is whether a tradeable edge is detected. At strength
0 the rates are therefore below the nominal 5% (costs make every true Sharpe negative);
E1 measures the size on gross returns.
"""

from __future__ import annotations

import numpy as np

from overfitlab.config import deep_merge, n_jobs
from overfitlab.experiments.common import save_results
from overfitlab.experiments.e1_null import run_worlds, summarise_worlds
from overfitlab.reporting.render import md_table

METHODS = [
    ("any_naive", "Naive (any p < 0.05)"),
    ("any_bh", "Benjamini-Hochberg"),
    ("any_holm", "Holm"),
    ("rc", "White's Reality Check"),
    ("spa", "Hansen's SPA"),
    ("dsr_max_matching", "Deflated Sharpe Ratio (as in the paper)"),
    ("dsr_null", "Deflated Sharpe Ratio, theoretical dispersion"),
]


def first_crossing(strengths: list[float], rates: list[float], level: float) -> float | None:
    """Smallest strength (linear interpolation) at which the detection rate reaches ``level``."""
    for i, rate in enumerate(rates):
        if rate >= level:
            if i == 0:
                return strengths[0]
            x0, x1, y0, y1 = strengths[i - 1], strengths[i], rates[i - 1], rate
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0) if y1 > y0 else x1
    return None


def run(cfg: dict) -> dict:
    """Run E2 and write ``results/e2.json``."""
    ecfg = cfg["experiments"]["e2"]
    seed = cfg["seed"] + 200_000
    levels = []
    for i, strength in enumerate(ecfg["strengths"]):
        synth = deep_merge(cfg["synthetic"], {"signal_strength": strength, "drift_annual": 0.0})
        tasks = [
            (cfg, synth, ecfg, seed + 1000 * i + w, "net", True) for w in range(ecfg["n_worlds"])
        ]
        worlds = run_worlds(tasks, n_jobs(cfg))
        summary = summarise_worlds(worlds)
        # "True" Sharpe of each variant: its average over worlds and assets (the same 500
        # rules are applied to every simulated asset, and all assets share one law).
        per_variant = np.mean(
            [w["sharpe_annual"].reshape(ecfg["n_assets"], -1) for w in worlds], axis=(0, 1)
        )
        summary["true_best_annual"] = float(per_variant.max())
        summary["true_median_annual"] = float(np.median(per_variant))
        summary["strength"] = strength
        levels.append(summary)

    strengths = list(ecfg["strengths"])
    thresholds = {}
    rows = []
    for key, label in METHODS:
        rates = [lvl["rates"][key]["rate"] for lvl in levels]
        t50, t80 = first_crossing(strengths, rates, 0.5), first_crossing(strengths, rates, 0.8)
        thresholds[key] = {"t50": t50, "t80": t80, "rate_at_zero": rates[0]}

        def fmt(v):
            return "not reached" if v is None else f"{v:.2f}"

        rows.append([label, f"{rates[0]:.0%}", fmt(t50), fmt(t80)])

    def reach(key: str, name: str) -> str:
        t50 = thresholds[key]["t50"]
        if t50 is None:
            return (
                f"{name} never reaches 50% detection in the tested range (up to {strengths[-1]:g})"
            )
        true_sr = float(np.interp(t50, strengths, [lvl["true_best_annual"] for lvl in levels]))
        return (
            f"{name} reaches 50% detection at an oracle Sharpe of about {t50:.1f} "
            f"(true net Sharpe of the best variant about {true_sr:.1f})"
        )

    sentence = (
        f"{reach('spa', 'SPA')}; {reach('any_holm', 'Holm')}; "
        f"{reach('dsr_max_matching', 'the DSR as defined in the paper')}."
    )
    results = {
        "alpha": cfg["diagnostics"]["alpha"],
        "summary": {
            "sentence": sentence,
            "max_strength": strengths[-1],
            "true_best_at_max": levels[-1]["true_best_annual"],
            "true_best_at_zero": levels[0]["true_best_annual"],
            "mean_best_at_max": levels[-1]["means"]["best_annual"],
            "spa_at_max": levels[-1]["rates"]["spa"]["rate"],
            "dsr_paper_at_max": levels[-1]["rates"]["dsr_max_matching"]["rate"],
            "dsr_null_at_max": levels[-1]["rates"]["dsr_null"]["rate"],
            "pbo_at_zero": levels[0]["means"]["pbo"],
            "pbo_at_max": levels[-1]["means"]["pbo"],
        },
        "strengths": strengths,
        "n_worlds": ecfg["n_worlds"],
        "n_variants": ecfg["n_assets"] * 500,
        "n_bars": ecfg["n_bars"],
        "half_life": cfg["synthetic"]["signal_half_life"],
        "levels": levels,
        "thresholds": thresholds,
        "tables": {
            "thresholds_md": md_table(
                [
                    "Method",
                    "Detection rate at zero signal",
                    "Strength for 50% detection",
                    "Strength for 80% detection",
                ],
                rows,
            ),
            "levels_md": md_table(
                [
                    "Oracle Sharpe",
                    "True Sharpe of best variant (net, estimate)",
                    "Mean best in-sample Sharpe",
                    "SPA",
                    "Holm",
                    "BH",
                    "DSR (paper)",
                    "DSR (theoretical dispersion)",
                    "Mean PBO",
                ],
                [
                    [
                        f"{lvl['strength']:.1f}",
                        f"{lvl['true_best_annual']:.2f}",
                        f"{lvl['means']['best_annual']:.2f}",
                        f"{lvl['rates']['spa']['rate']:.0%}",
                        f"{lvl['rates']['any_holm']['rate']:.0%}",
                        f"{lvl['rates']['any_bh']['rate']:.0%}",
                        f"{lvl['rates']['dsr_max_matching']['rate']:.0%}",
                        f"{lvl['rates']['dsr_null']['rate']:.0%}",
                        f"{lvl['means']['pbo']:.2f}",
                    ]
                    for lvl in levels
                ],
            ),
        },
    }
    save_results(cfg, "e2", results)
    return {"results": results}
