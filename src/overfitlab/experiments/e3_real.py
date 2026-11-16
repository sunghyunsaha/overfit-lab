"""E3 - Real data: the full pipeline on the crypto universe.

Loads the market (Binance cache, else CoinMetrics cache, else the synthetic stand-in -
the source actually used is recorded in the results and printed in every report),
backtests all variants and runs every diagnostic on the NET return matrix. The gross
matrix gets the light diagnostics for comparison, and so do the ACTIVE returns (each
variant minus buy-and-hold of its own asset), net and gross.

Headline: how many of N variants survive each correction, the PBO, and the Deflated
Sharpe Ratio of the best variant.
"""

from __future__ import annotations

import numpy as np

from overfitlab.backtest.indicators import simple_returns
from overfitlab.backtest.universe import Universe, _daily_sum, build_universe
from overfitlab.config import resolve_path
from overfitlab.data.loader import ValidationLog, load_market
from overfitlab.diagnostics.sharpe import return_moments
from overfitlab.experiments.common import (
    describe_variant,
    diagnose,
    one_sided_pvalues,
    save_results,
)
from overfitlab.reporting.render import md_table


def load_universe(cfg: dict) -> tuple[Universe, dict, dict]:
    """Load the configured market and build its universe: ``(universe, info, prices)``."""
    prices, info = load_market(cfg)
    return build_universe(prices, cfg), info, prices


def family_table(uni: Universe, net: np.ndarray, cfg: dict) -> list[dict]:
    """Per-family summary of the net results."""
    ann = np.sqrt(cfg["periods_per_year"])
    m = return_moments(net)
    p = one_sided_pvalues(m, net.shape[0])
    sr = m["sharpe"] * ann
    rows = []
    for family, idx in uni.meta.groupby("family", sort=False).indices.items():
        live = m["live"][idx]
        rows.append(
            {
                "family": family,
                "n": int(len(idx)),
                "median_sharpe": float(np.median(sr[idx][live])),
                "best_sharpe": float(sr[idx].max()),
                "frac_positive": float(np.mean(sr[idx][live] > 0)),
                "naive_significant": int(np.sum(p[idx] < cfg["diagnostics"]["alpha"])),
                "turnover_per_year": float(uni.turnover[:, idx].mean() * cfg["periods_per_year"]),
            }
        )
    return rows


def asset_table(prices: dict, cfg: dict) -> list[dict]:
    """Buy-and-hold statistics of each asset, for context."""
    ann = np.sqrt(cfg["periods_per_year"])
    rows = []
    for asset, df in prices.items():
        r = simple_returns(df["close"].to_numpy(dtype=float))[1:]
        if cfg["data"]["interval"] != "1d":
            r = df["close"].resample("1D").last().pct_change().dropna().to_numpy()
        rows.append(
            {
                "asset": asset,
                "buy_hold_sharpe": float(r.mean() / r.std(ddof=1) * ann),
                "vol_annual": float(r.std(ddof=1) * ann),
                "total_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0),
            }
        )
    return rows


def buy_and_hold_matrix(uni: Universe, prices: dict, cfg: dict) -> np.ndarray:
    """T x N matrix whose column j is the buy-and-hold return of variant j's asset."""
    columns = {}
    for asset, df in prices.items():
        r = simple_returns(df["close"].to_numpy(dtype=float))[:, None]
        if cfg["data"]["interval"] != "1d":
            r = _daily_sum(df.index, r)[1]
        columns[asset] = r[:, 0]
    return np.column_stack([columns[a] for a in uni.meta["asset"]])


def survival_rows(net: dict, gross: dict) -> list[list]:
    """Rows of the survival table (shared by E1 and E3 reports)."""
    labels = [
        ("naive", "Naive: p < alpha, no correction"),
        ("bh", "Benjamini-Hochberg (FDR)"),
        ("by", "Benjamini-Yekutieli (FDR, any dependence)"),
        ("holm", "Holm (FWER)"),
        ("bonferroni", "Bonferroni (FWER)"),
    ]
    n = net["n_variants"]
    return [
        [
            text,
            f"{gross['survivors'][key]:,}",
            f"{gross['survivors'][key] / n:.1%}",
            f"{net['survivors'][key]:,}",
            f"{net['survivors'][key] / n:.1%}",
        ]
        for key, text in labels
    ]


def survival_table(net: dict, gross: dict) -> str:
    return md_table(
        [
            "Test on each variant",
            "Gross: survivors",
            "Gross: share",
            "Net: survivors",
            "Net: share",
        ],
        survival_rows(net, gross),
    )


def family_level_table(
    net: dict, gross: dict, vs_hold: dict | None = None, gross_vs_hold: dict | None = None
) -> str:
    """Family-level tests and selection diagnostics, gross and net.

    The optional columns repeat the analysis on active returns (variant minus
    buy-and-hold of its own asset), before and after costs.
    """

    def col(d: dict) -> list[str]:
        return [
            f"{d['sharpe']['best_annual']:.2f}",
            f"{d['best']['psr0']:.4f}",
            f"{d['snooping']['rc_p']:.3f}",
            f"{d['snooping']['spa_p']:.3f}",
            f"{d['dsr_primary']['sr0_annual']:.2f}",
            f"{d['dsr_primary']['dsr']:.3f}",
            f"{d['pbo']['pbo']:.3f}",
        ]

    names = [
        "Best annualised Sharpe ratio",
        "PSR(0) of the best variant (no correction)",
        "White's Reality Check p-value",
        "Hansen's SPA p-value (consistent)",
        "Expected max Sharpe under the null, SR0 (annualised)",
        "Deflated Sharpe Ratio of the best variant",
        "Probability of Backtest Overfitting",
    ]
    cols = [col(gross), col(net)]
    head = ["Quantity", "Gross", "Net"]
    if gross_vs_hold is not None:
        cols.append(col(gross_vs_hold))
        head.append("Gross, minus buy-and-hold")
    if vs_hold is not None:
        cols.append(col(vs_hold))
        head.append("Net, minus buy-and-hold")
    return md_table(head, [[name, *vals] for name, *vals in zip(names, *cols, strict=True)])


def neff_table(out: dict) -> str:
    names = {
        "raw": "Raw count (variants that traded)",
        "avg_corr": "Average correlation (Bailey & Lopez de Prado 2014)",
        "max_matching": "Max-matching simulation (this project)",
        "cluster": "Clustering (simplified ONC)",
        "eigen": "Eigenvalue participation ratio",
    }
    rows = [
        [
            names[k],
            f"{out['dsr'][k]['n_trials']:,.0f}",
            f"{out['dsr'][k]['sr0_annual']:.2f}",
            f"{out['dsr'][k]['dsr']:.3f}",
        ]
        for k in names
        if k in out["dsr"]
    ]
    return md_table(["Estimator of N", "Effective N", "SR0 (annualised)", "DSR of best"], rows)


def _join(names: list[str]) -> str:
    """``["A"] -> "A"``, ``["A", "B"] -> "A and B"``, ``["A", "B", "C"] -> "A, B and C"``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def top10_summary(top10: list[dict], assets: list[dict]) -> dict:
    """Where the ten best variants come from, for the prose of the report."""
    in_top = list(dict.fromkeys(r["asset"] for r in top10))
    by_hold = sorted(assets, key=lambda r: r["buy_hold_sharpe"], reverse=True)
    rank = {r["asset"]: i + 1 for i, r in enumerate(by_hold)}
    return {
        "assets_text": _join(in_top),
        "n_assets": len(in_top),
        "n_long_only": sum("long_only" in r["label"] for r in top10),
        "worst_hold_rank": max(rank[a] for a in in_top),
        "n_hold_sharpe_positive": sum(r["buy_hold_sharpe"] > 0 for r in assets),
        "n_total_return_positive": sum(r["total_return"] > 0 for r in assets),
        "best_hold_asset": by_hold[0]["asset"],
        "best_hold_sharpe": by_hold[0]["buy_hold_sharpe"],
    }


def verdict(net: dict) -> dict:
    """Plain-language reading of the net results, generated from the numbers."""
    alpha = net["alpha"]
    holm = net["survivors"]["holm"]
    bh = net["survivors"]["bh"]
    spa = net["snooping"]["spa_p"]
    holm_psr = net["survivors_psr"]["holm"]
    spa_hac = net["snooping"]["spa_hac_p"]
    dsr = net["dsr_primary"]["dsr"]
    pbo = net["pbo"]["pbo"]
    checks = {
        "holm": holm > 0,
        "spa": spa < alpha,
        "dsr": dsr > 1 - alpha,
    }
    passed = sum(checks.values())
    if passed == 0:
        overall = (
            "No variant survives: after correcting for the search, the best backtest is "
            "indistinguishable from the luckiest of many skill-less strategies."
        )
    elif passed == 3:
        overall = (
            "The best variants survive every correction: the evidence of real performance "
            "holds up after accounting for the search."
        )
    else:
        overall = (
            "The evidence is mixed: some corrections are passed and others are not, so the "
            "result should be treated as unproven."
        )
    return {
        "overall": overall,
        "n_checks_passed": passed,
        "holm_text": f"{holm:,} survive Holm" if holm else "none survive Holm",
        "bh_text": f"{bh:,} survive Benjamini-Hochberg"
        if bh
        else "none survive Benjamini-Hochberg",
        "spa_text": "rejects" if checks["spa"] else "does not reject",
        # Do the two corrections adopted after the null-world validation change a decision?
        "pvalue_fix_text": (
            "changes the family-wise answer"
            if (holm_psr > 0) != (holm > 0)
            else "does not change whether anything survives Holm"
        ),
        "spa_scale_text": (
            "changes the SPA decision at the 5% level"
            if (spa_hac < alpha) != (spa < alpha)
            else "does not change the SPA decision at the 5% level"
        ),
        "dsr_text": "above" if checks["dsr"] else "below",
        "pbo_text": (
            "the in-sample winner usually stays in the top half out of sample"
            if pbo < 0.25
            else "the in-sample winner often falls into the bottom half out of sample"
            if pbo < 0.6
            else "the in-sample winner usually falls into the bottom half out of sample"
        ),
    }


def run(cfg: dict) -> dict:
    """Run E3 and write ``results/e3.json``."""
    seed = cfg["seed"] + 300
    uni, info, prices = load_universe(cfg)
    uni.save(resolve_path(cfg, "artifacts_dir") / "universe")
    log = ValidationLog(records=info["validation"])
    log.to_json(resolve_path(cfg, "output_dir") / "results" / "data_validation.json")

    net_matrix = uni.net()
    net, arrays = diagnose(net_matrix, cfg, seed=seed)
    gross, arrays_gross = diagnose(uni.gross, cfg, seed=seed, heavy=False)
    arrays["sharpe_annual_gross"] = arrays_gross["sharpe_annual"]
    arrays["index"] = uni.index

    # Same diagnostics on ACTIVE returns: each variant minus buy-and-hold of its own
    # asset. This asks "does anything beat simply holding the coin?", which strips out
    # the market's own drift that long-only variants earn without any timing skill.
    hold = buy_and_hold_matrix(uni, prices, cfg)
    primary = cfg["diagnostics"]["trials"]["primary"]
    active = net_matrix - hold
    vs_hold, _ = diagnose(active, cfg, seed=seed + 1, heavy=False, neff_methods=(primary,))
    # The same before costs: is there anything in the signals themselves, beyond the
    # asset's drift, that costs then remove?
    active = uni.gross - hold
    gross_vs_hold, _ = diagnose(active, cfg, seed=seed + 2, heavy=False, neff_methods=(primary,))
    del active, hold

    best = describe_variant(uni.meta, net["sharpe"]["best_index"])
    best_gross = describe_variant(uni.meta, gross["sharpe"]["best_index"])
    order = np.argsort(arrays["sharpe_annual"])[::-1][:10]
    top10 = [
        {
            **describe_variant(uni.meta, j),
            "sharpe": float(arrays["sharpe_annual"][j]),
            "p": float(arrays["p"][j]),
        }
        for j in order
    ]
    families = family_table(uni, net_matrix, cfg)
    assets = asset_table(prices, cfg)

    tables = {
        "survival_md": survival_table(net, gross),
        "family_level_md": family_level_table(net, gross, vs_hold, gross_vs_hold),
        "neff_md": neff_table(net),
        "families_md": md_table(
            [
                "Family",
                "Variants",
                "Median Sharpe",
                "Best Sharpe",
                "Share with Sharpe > 0",
                "Naively significant",
                "Turnover (x equity / year)",
            ],
            [
                [
                    r["family"],
                    f"{r['n']:,}",
                    f"{r['median_sharpe']:.2f}",
                    f"{r['best_sharpe']:.2f}",
                    f"{r['frac_positive']:.0%}",
                    f"{r['naive_significant']:,}",
                    f"{r['turnover_per_year']:.0f}",
                ]
                for r in families
            ],
        ),
        "top10_md": md_table(
            ["Rank", "Variant", "Net Sharpe (annualised)", "Unadjusted p-value"],
            [
                [str(i + 1), f"`{r['label']}`", f"{r['sharpe']:.2f}", f"{r['p']:.2g}"]
                for i, r in enumerate(top10)
            ],
        ),
        "assets_md": md_table(
            ["Asset", "Buy-and-hold Sharpe", "Annualised volatility", "Total return"],
            [
                [
                    r["asset"],
                    f"{r['buy_hold_sharpe']:.2f}",
                    f"{r['vol_annual']:.0%}",
                    f"{r['total_return']:+.0%}",
                ]
                for r in assets
            ],
        ),
        "validation_md": md_table(
            ["Check", "Bars affected (all assets)"],
            [[k.replace("_", " "), f"{v:,}"] for k, v in info["validation_counts"].items()],
        ),
    }
    results = {
        "data": {k: v for k, v in info.items() if k != "validation"},
        "costs": {**cfg["costs"], "execution_lag": cfg["execution_lag"]},
        "n_variants": net["n_variants"],
        "n_assets": len(prices),
        "net": net,
        "gross": gross,
        "vs_buy_hold": vs_hold,
        "gross_vs_buy_hold": gross_vs_hold,
        "best": best,
        "best_gross": best_gross,
        "top10": top10,
        "families": families,
        "assets": assets,
        "top10_summary": top10_summary(top10, assets),
        "verdict": verdict(net),
        "tables": tables,
    }
    save_results(cfg, "e3", results)
    arrays.pop("p", None)
    return {"results": results, "arrays": arrays, "universe": uni}
