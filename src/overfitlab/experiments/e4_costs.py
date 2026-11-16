"""E4 - Cost sensitivity: how does survival change as the fee goes from 0 to 30 bps?

Net returns are linear in the fee (``net = gross - fee * turnover - slippage``), so each
fee level reuses the same backtest and only the diagnostics are re-run. Slippage is held
at its configured level throughout; only the exchange fee varies.
"""

from __future__ import annotations

from overfitlab.experiments.common import diagnose, save_results
from overfitlab.experiments.e3_real import load_universe
from overfitlab.reporting.render import md_table


def run(cfg: dict) -> dict:
    """Run E4 and write ``results/e4.json``."""
    seed = cfg["seed"] + 400
    uni, info, _ = load_universe(cfg)
    primary = cfg["diagnostics"]["trials"]["primary"]
    levels = []
    for fee in cfg["experiments"]["e4"]["fees_bps"]:
        out, _ = diagnose(uni.net(float(fee)), cfg, seed=seed, heavy=False, neff_methods=(primary,))
        out["fee_bps"] = fee
        levels.append(out)
    rows = [
        [
            f"{d['fee_bps']:g}",
            f"{d['sharpe']['median_annual']:.2f}",
            f"{d['sharpe']['best_annual']:.2f}",
            f"{d['survivors']['naive']:,}",
            f"{d['survivors']['bh']:,}",
            f"{d['survivors']['holm']:,}",
            f"{d['snooping']['spa_p']:.3f}",
            f"{d['dsr_primary']['dsr']:.3f}",
            f"{d['pbo']['pbo']:.2f}",
        ]
        for d in levels
    ]
    holm = [d["survivors"]["holm"] for d in levels]
    if min(holm) == max(holm):
        holm_text = f"there are {holm[0]:,} Holm survivors at every fee level"
    else:
        holm_text = f"Holm survivors range from {min(holm):,} to {max(holm):,} across fee levels"
    results = {
        "data": {k: v for k, v in info.items() if k != "validation"},
        "fees_bps": list(cfg["experiments"]["e4"]["fees_bps"]),
        "levels": levels,
        "summary": {
            "holm_text": holm_text,
            "holm_text_cap": holm_text[0].upper() + holm_text[1:],
            "holm_min": min(d["survivors"]["holm"] for d in levels),
            "holm_max": max(d["survivors"]["holm"] for d in levels),
            "bh_at_min_fee": levels[0]["survivors"]["bh"],
            "bh_at_max_fee": levels[-1]["survivors"]["bh"],
            "naive_at_min_fee": levels[0]["survivors"]["naive"],
            "naive_at_max_fee": levels[-1]["survivors"]["naive"],
            "best_at_min_fee": levels[0]["sharpe"]["best_annual"],
            "best_at_max_fee": levels[-1]["sharpe"]["best_annual"],
            "median_at_min_fee": levels[0]["sharpe"]["median_annual"],
            "median_at_max_fee": levels[-1]["sharpe"]["median_annual"],
            "dsr_max": max(d["dsr_primary"]["dsr"] for d in levels),
            "spa_p_min": min(d["snooping"]["spa_p"] for d in levels),
            "min_fee": levels[0]["fee_bps"],
            "max_fee": levels[-1]["fee_bps"],
        },
        "tables": {
            "costs_md": md_table(
                [
                    "Fee (bps per side)",
                    "Median Sharpe",
                    "Best Sharpe",
                    "Naive",
                    "BH",
                    "Holm",
                    "SPA p",
                    "DSR of best",
                    "PBO",
                ],
                rows,
            )
        },
    }
    save_results(cfg, "e4", results)
    return {"results": results}
