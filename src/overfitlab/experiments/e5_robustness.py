"""E5 - Robustness: do the conclusions depend on the knobs?

Varies, one at a time: the number of CSCV blocks, the number of CPCV groups and the
embargo, the sample period (equal sub-periods), the bootstrap block length, and the
execution lag (one extra bar of delay between signal and trade).
"""

from __future__ import annotations

import numpy as np

from overfitlab.backtest.universe import build_universe
from overfitlab.config import deep_merge
from overfitlab.data.loader import load_market
from overfitlab.diagnostics.bootstrap import reality_check_and_spa
from overfitlab.diagnostics.cv import cpcv_select_best, purged_kfold
from overfitlab.diagnostics.pbo import cscv
from overfitlab.diagnostics.sharpe import sharpe_ratio
from overfitlab.experiments.common import cpcv_summary, derive_seed, diagnose, save_results
from overfitlab.experiments.e3_real import load_universe
from overfitlab.reporting.render import md_table


def purged_kfold_selection(net: np.ndarray, n_splits: int, horizon: int, embargo: int) -> dict:
    """Out-of-sample Sharpe of "pick the best training-set variant" under purged k-fold."""
    oos = np.zeros(net.shape[0])
    is_sr = []
    for train, test in purged_kfold(net.shape[0], n_splits, horizon, embargo):
        sr = sharpe_ratio(net[train])
        best = int(np.argmax(sr))
        is_sr.append(float(sr[best]))
        oos[test] = net[test, best]
    return {
        "n_splits": n_splits,
        "is_sharpe": float(np.mean(is_sr)),
        "oos_sharpe": float(oos.mean() / oos.std(ddof=1)),
    }


def run(cfg: dict) -> dict:
    """Run E5 and write ``results/e5.json``."""
    ecfg = cfg["experiments"]["e5"]
    dcfg = cfg["diagnostics"]
    seed = cfg["seed"] + 500
    ann = np.sqrt(cfg["periods_per_year"])
    primary = dcfg["trials"]["primary"]
    uni, info, _ = load_universe(cfg)
    net = uni.net()
    t = net.shape[0]

    cscv_rows = []
    for s in ecfg["cscv_blocks"]:
        res = cscv(net, s)
        cscv_rows.append(
            {
                "n_blocks": s,
                "n_splits": res.n_splits,
                "pbo": res.pbo,
                "prob_oos_loss": res.prob_oos_loss,
            }
        )

    cpcv_rows = []
    horizon = dcfg["cpcv"]["label_horizon"]
    for g in ecfg["cpcv_groups"]:
        for pct in ecfg["embargo_pcts"]:
            cp = cpcv_select_best(
                net, g, dcfg["cpcv"]["n_test_groups"], horizon, int(round(pct * t))
            )
            cpcv_rows.append({"n_groups": g, "embargo_pct": pct, **cpcv_summary(cp, ann)})
    kfold_rows = []
    for k in (5, 10):
        r = purged_kfold_selection(net, k, horizon, int(round(dcfg["cpcv"]["embargo_pct"] * t)))
        kfold_rows.append(
            {"n_splits": k, "is_annual": r["is_sharpe"] * ann, "oos_annual": r["oos_sharpe"] * ann}
        )

    bounds = np.linspace(0, t, ecfg["n_subperiods"] + 1).astype(int)
    subperiods = []
    for i in range(ecfg["n_subperiods"]):
        lo, hi = bounds[i], bounds[i + 1]
        out, _ = diagnose(net[lo:hi], cfg, seed=seed + i, heavy=False, neff_methods=(primary,))
        out["label"] = f"{uni.index[lo].date()} to {uni.index[hi - 1].date()}"
        # How does this period's winner do in the other periods?
        winner = out["sharpe"]["best_index"]
        rest = np.r_[net[:lo, winner], net[hi:, winner]]
        out["winner_elsewhere_annual"] = (
            float(rest.mean() / rest.std(ddof=1) * ann) if rest.std() > 0 else 0.0
        )
        subperiods.append(out)

    block_rows = []
    for mb in ecfg["mean_blocks"]:
        r = reality_check_and_spa(
            net, dcfg["bootstrap"]["n_boot"], float(mb), seed=derive_seed(seed, 50)
        )
        block_rows.append({"mean_block": mb, "rc_p": r.rc_pvalue, "spa_p": r.spa_consistent})

    lag_cfg = deep_merge(cfg, {"execution_lag": cfg["execution_lag"] + 1})
    prices, _ = load_market(lag_cfg)
    lagged, _ = diagnose(
        build_universe(prices, lag_cfg).net(),
        cfg,
        seed=seed + 60,
        heavy=False,
        neff_methods=(primary,),
    )
    base, _ = diagnose(net, cfg, seed=seed + 60, heavy=False, neff_methods=(primary,))

    def brief(d: dict) -> list[str]:
        return [
            f"{d['sharpe']['best_annual']:.2f}",
            f"{d['dsr_primary']['sr0_annual']:.2f}",
            f"{d['survivors']['naive']:,}",
            f"{d['survivors']['holm']:,}",
            f"{d['snooping']['spa_p']:.3f}",
            f"{d['dsr_primary']['dsr']:.3f}",
            f"{d['pbo']['pbo']:.2f}",
        ]

    brief_head = ["Best Sharpe", "SR0", "Naive", "Holm", "SPA p", "DSR", "PBO"]
    results = {
        "alpha": dcfg["alpha"],
        "data": {k: v for k, v in info.items() if k != "validation"},
        "cscv": cscv_rows,
        "cpcv": cpcv_rows,
        "purged_kfold": kfold_rows,
        "subperiods": subperiods,
        "block_length": block_rows,
        "lag": {"base": base, "lagged": lagged, "lag_bars": lag_cfg["execution_lag"]},
        "subperiod_holm_text": (
            "no sub-period has a single Holm survivor"
            if max(p["survivors"]["holm"] for p in subperiods) == 0
            else f"no sub-period has more than {max(p['survivors']['holm'] for p in subperiods):,} "
            "Holm survivors"
        ),
        "ranges": {
            "pbo_min": min(r["pbo"] for r in cscv_rows),
            "pbo_max": max(r["pbo"] for r in cscv_rows),
            "cpcv_median_min": min(r["median_path_annual"] for r in cpcv_rows),
            "cpcv_median_max": max(r["median_path_annual"] for r in cpcv_rows),
            "spa_p_min": min(r["spa_p"] for r in block_rows),
            "spa_p_max": max(r["spa_p"] for r in block_rows),
            "subperiod_holm_max": max(p["survivors"]["holm"] for p in subperiods),
            "subperiod_dsr_max": max(p["dsr_primary"]["dsr"] for p in subperiods),
        },
        "tables": {
            "cscv_md": md_table(
                ["CSCV blocks S", "Splits", "PBO", "P(OOS Sharpe of winner < 0)"],
                [
                    [
                        str(r["n_blocks"]),
                        f"{r['n_splits']:,}",
                        f"{r['pbo']:.3f}",
                        f"{r['prob_oos_loss']:.3f}",
                    ]
                    for r in cscv_rows
                ],
            ),
            "cpcv_md": md_table(
                [
                    "Groups",
                    "Embargo",
                    "Splits",
                    "Paths",
                    "Mean training Sharpe",
                    "Median path Sharpe",
                    "Worst path",
                    "Best path",
                    "Paths below zero",
                ],
                [
                    [
                        str(r["n_groups"]),
                        f"{r['embargo_pct']:.0%}",
                        str(r["n_splits"]),
                        str(r["n_paths"]),
                        f"{r['mean_is_annual']:.2f}",
                        f"{r['median_path_annual']:.2f}",
                        f"{r['min_path_annual']:.2f}",
                        f"{r['max_path_annual']:.2f}",
                        f"{r['frac_paths_negative']:.0%}",
                    ]
                    for r in cpcv_rows
                ],
            ),
            "kfold_md": md_table(
                [
                    "Purged k-fold splits",
                    "Mean training Sharpe of pick",
                    "Stitched out-of-sample Sharpe",
                ],
                [
                    [str(r["n_splits"]), f"{r['is_annual']:.2f}", f"{r['oos_annual']:.2f}"]
                    for r in kfold_rows
                ],
            ),
            "subperiods_md": md_table(
                ["Period", *brief_head, "Winner's Sharpe in the other periods"],
                [
                    [p["label"], *brief(p), f"{p['winner_elsewhere_annual']:.2f}"]
                    for p in subperiods
                ],
            ),
            "block_md": md_table(
                ["Mean block (days)", "Reality Check p", "SPA p"],
                [
                    [str(r["mean_block"]), f"{r['rc_p']:.3f}", f"{r['spa_p']:.3f}"]
                    for r in block_rows
                ],
            ),
            "lag_md": md_table(
                ["Execution", *brief_head],
                [
                    ["next bar (baseline)", *brief(base)],
                    [f"{lag_cfg['execution_lag']} bars after the signal", *brief(lagged)],
                ],
            ),
        },
    }
    save_results(cfg, "e5", results)
    return {"results": results}
