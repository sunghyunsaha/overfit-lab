"""All figures. Each function reads an experiment's results (and arrays) and writes PNGs.

Style rules followed throughout: one y-axis per panel (never dual axes), thin marks,
recessive grid, a legend whenever a panel has two or more series, text in ink colours
rather than series colours, and a fixed categorical colour order.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from overfitlab.diagnostics.sharpe import min_backtest_length  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
NEUTRAL = "#a3a29c"
CRITICAL = "#d03b3b"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
        }
    )


def _save(fig, outdir: Path, name: str) -> Path:
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _vline(ax, x: float, label: str, color: str = INK2, y: float = 0.97, ha: str = "left") -> None:
    ax.axvline(x, color=color, linewidth=1.2)
    dx = 4 if ha == "left" else -4
    ax.annotate(
        label,
        (x, y),
        xycoords=("data", "axes fraction"),
        xytext=(dx, 0),
        textcoords="offset points",
        ha=ha,
        va="top",
        fontsize=8,
        color=INK2,
    )


def _best_vs_luck(ax, best: float, sr0: float, best_label: str = "best") -> None:
    """Mark the best Sharpe and the luck benchmark; labels sit on the outer sides."""
    left_is_best = best <= sr0
    _vline(ax, best, best_label, color=CRITICAL, y=0.62, ha="right" if left_is_best else "left")
    _vline(ax, sr0, "expected best\nunder the null", y=0.97, ha="left" if left_is_best else "right")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, max(best, sr0) + 0.22 * (hi - lo)))


def _legend_below(ax, ncols: int = 1) -> None:
    """Legend under the axis: it can never cover the data, whatever the numbers are."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=ncols)


def _hbars(
    ax, labels: list[str], values: list[float], color: str = BLUE, fmt: str = "{:,.0f}"
) -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=color)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    top = max(max(values), 1e-12)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.012 * top, yi, fmt.format(v), va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.16)


SURVIVAL_LABELS = [
    ("naive", "Naive (p < 0.05)"),
    ("bh", "Benjamini-Hochberg"),
    ("by", "Benjamini-Yekutieli"),
    ("holm", "Holm"),
    ("bonferroni", "Bonferroni"),
]


def _survival_panel(ax, diag: dict, title: str) -> None:
    labels = ["All variants"] + [text for _, text in SURVIVAL_LABELS]
    values = [diag["n_variants"]] + [diag["survivors"][k] for k, _ in SURVIVAL_LABELS]
    _hbars(ax, labels, values)
    ax.set_title(title)
    ax.set_xlabel("Number of variants")


def _sharpe_hist_panel(ax, sharpe: np.ndarray, live: np.ndarray, diag: dict, title: str) -> None:
    ax.hist(sharpe[live], bins=70, color=BLUE, alpha=0.85, linewidth=0)
    naive = 1.645 / np.sqrt(diag["years"])
    _vline(ax, naive, "naive 5%\ncut-off", y=0.97, ha="right")
    _best_vs_luck(ax, diag["sharpe"]["best_annual"], diag["dsr_primary"]["sr0_annual"])
    ax.set_title(title)
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_ylabel("Number of variants")


def _logit_panel(ax, logits: np.ndarray, pbo: float, title: str) -> None:
    bins = np.linspace(min(logits.min(), -1), max(logits.max(), 1), 60)
    ax.hist(
        logits[logits <= 0],
        bins=bins,
        color=ORANGE,
        linewidth=0,
        label="in-sample winner in bottom half",
    )
    ax.hist(
        logits[logits > 0], bins=bins, color=BLUE, linewidth=0, label="in-sample winner in top half"
    )
    ax.axvline(0, color=INK2, linewidth=1.2)
    ax.set_title(f"{title}: PBO = {pbo:.2f}")
    ax.set_xlabel("Logit of the winner's out-of-sample rank")
    ax.set_ylabel("Number of CSCV splits")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)  # head-room for the legend
    ax.legend(loc="upper left")


def fig_e1(results: dict, arrays: dict, outdir: Path) -> list[Path]:
    """E1 figures: the null world."""
    setup_style()
    full = results["full"]
    out = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _sharpe_hist_panel(
        axes[0],
        arrays["gross"]["sharpe_annual"],
        arrays["gross"]["live"],
        full["gross"],
        "Null world: gross Sharpe ratios of 5,000 skill-less variants",
    )
    _survival_panel(axes[1], full["gross"], "Null world: variants flagged as significant")
    out.append(_save(fig, outdir, "e1_null_world.png"))

    rows = [
        ("any_naive", "Naive: any variant p < 0.05"),
        ("any_bonferroni_psr", "Bonferroni on 1 - PSR(0)"),
        ("any_bonferroni", "Bonferroni"),
        ("any_holm", "Holm"),
        ("any_bh", "Benjamini-Hochberg"),
        ("any_by", "Benjamini-Yekutieli"),
        ("rc", "White's Reality Check"),
        ("spa_hac", "SPA, long-run s.d. (Hansen 2005)"),
        ("spa", "SPA, sample s.d. (used here)"),
        ("dsr_cluster", "DSR, N from clustering"),
        ("dsr_max_matching", "DSR, N from max-matching"),
        ("dsr_raw", "DSR, raw N"),
        ("dsr_null", "DSR, max-matching N, theoretical dispersion"),
    ]
    kinds = list(results["mc"])
    rows = [r for r in rows if r[0] in results["mc"][kinds[0]]["rates"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(rows))[::-1]
    names = {"gbm": "GBM null", "garch": "GARCH fat-tailed null"}
    for i, kind in enumerate(kinds):
        rates = results["mc"][kind]["rates"]
        vals = np.array([rates[k]["rate"] for k, _ in rows])
        lo = np.array([rates[k]["lo"] for k, _ in rows])
        hi = np.array([rates[k]["hi"] for k, _ in rows])
        offset = 0.17 * (1 - 2 * i) if len(kinds) > 1 else 0.0
        ax.errorbar(
            vals,
            y + offset,
            xerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)],
            fmt="o",
            color=SERIES[i],
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            elinewidth=1.4,
            label=names.get(kind, kind),
        )
    ax.axvline(results["alpha"], color=INK2, linewidth=1.2)
    ax.annotate(
        "nominal 5%",
        (results["alpha"], 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -2),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color=INK2,
    )
    ax.set_yticks(y, [text for _, text in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Share of null worlds with at least one false discovery (95% interval)")
    n_worlds = results["mc"][kinds[0]]["n_worlds"]
    ax.set_title(f"False-positive rate of each method over {n_worlds} null worlds per type")
    ax.legend(loc="lower right")
    out.append(_save(fig, outdir, "e1_false_positive_rates.png"))
    return out


def fig_e2(results: dict, outdir: Path) -> list[Path]:
    """E2 figure: detection rate against planted signal strength."""
    setup_style()
    x = np.array(results["strengths"])
    series = [
        ("spa", "Hansen's SPA"),
        ("any_bh", "Benjamini-Hochberg"),
        ("any_holm", "Holm"),
        ("dsr_null", "DSR, theoretical dispersion"),
        ("dsr_max_matching", "DSR as in the paper"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for i, (key, label) in enumerate(series):
        yv = np.array([lvl["rates"][key]["rate"] for lvl in results["levels"]])
        ax.plot(
            x, yv, color=SERIES[i], marker="o", markersize=5, markeredgecolor=SURFACE, label=label
        )
    ax.axhline(results["alpha"], color=INK2, linewidth=1.0)
    ax.annotate(
        "5%",
        (x[-1], results["alpha"]),
        xytext=(4, 2),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Planted signal strength (annualised Sharpe of an oracle)")
    ax.set_ylabel("Share of worlds where the method detects something")
    ax.set_title("Power: detection rate against signal strength")
    _legend_below(ax, ncols=2)

    ax = axes[1]
    pbo = np.array([lvl["means"]["pbo"] for lvl in results["levels"]])
    ax.plot(x, pbo, color=BLUE, marker="o", markersize=5, markeredgecolor=SURFACE)
    ax.axhline(0.5, color=INK2, linewidth=1.0)
    ax.annotate(
        "pure noise",
        (x[-1], 0.5),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK2,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Planted signal strength (annualised Sharpe of an oracle)")
    ax.set_ylabel("Mean PBO")
    ax.set_title("Probability of Backtest Overfitting against signal strength")
    return [_save(fig, outdir, "e2_power.png")]


def fig_e3(results: dict, arrays: dict, outdir: Path) -> list[Path]:
    """E3 figures: the real-data run."""
    setup_style()
    net = results["net"]
    out = []
    source = "real data" if results["data"]["is_real"] else "SYNTHETIC stand-in"

    # Key figure: survival funnel + PBO logits.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    _survival_panel(axes[0], net, f"How many variants survive ({source}, net of costs)")
    _logit_panel(
        axes[1], arrays["logits"], net["pbo"]["pbo"], "Out-of-sample rank of the in-sample winner"
    )
    out.append(_save(fig, outdir, "e3_key_figure.png"))

    # Sharpe distribution, gross vs net.
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    live = arrays["live"]
    lo = min(arrays["sharpe_annual"][live].min(), arrays["sharpe_annual_gross"][live].min())
    hi = max(arrays["sharpe_annual"][live].max(), arrays["sharpe_annual_gross"][live].max())
    bins = np.linspace(lo, hi, 70)
    ax.hist(
        arrays["sharpe_annual_gross"][live],
        bins=bins,
        histtype="step",
        color=ORANGE,
        linewidth=1.8,
        label="gross",
    )
    ax.hist(
        arrays["sharpe_annual"][live],
        bins=bins,
        color=BLUE,
        alpha=0.8,
        linewidth=0,
        label="net of costs",
    )
    _best_vs_luck(ax, net["sharpe"]["best_annual"], net["dsr_primary"]["sr0_annual"], "best net")
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_ylabel("Number of variants")
    ax.set_title(f"Sharpe ratios of all {net['n_variants']:,} variants ({source})")
    ax.legend(loc="upper left")
    out.append(_save(fig, outdir, "e3_sharpe_distribution.png"))

    # In-sample vs out-of-sample.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    ax = axes[0]
    xi, yo = arrays["decay_is_annual"][live], arrays["decay_oos_annual"][live]
    ax.scatter(xi, yo, s=4, color=BLUE, alpha=0.25, linewidths=0, rasterized=True)
    grid = np.linspace(xi.min(), xi.max(), 10)
    slope = net["decay"]["slope"]
    intercept = np.mean(yo) - slope * np.mean(xi)
    ax.plot(grid, intercept + slope * grid, color=ORANGE, label=f"fitted slope {slope:.2f}")
    ax.plot(grid, grid, color=NEUTRAL, linewidth=1.2, label="no decay (slope 1)")
    ax.scatter(
        [net["decay"]["best_is_annual"]],
        [net["decay"]["best_oos_annual"]],
        s=70,
        color=CRITICAL,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=5,
        label="in-sample winner",
    )
    ax.set_xlabel("Sharpe ratio, first half of the sample")
    ax.set_ylabel("Sharpe ratio, second half")
    ax.set_title("Every variant: first-half Sharpe against second-half Sharpe")
    ax.legend(loc="upper left")

    ax = axes[1]
    index = arrays["index"]
    covered = arrays["wf_covered"]
    wf = np.where(covered, arrays["wf_oos_returns"], 0.0)
    best = np.where(covered, arrays["best_returns"], 0.0)
    ax.plot(
        index, np.cumsum(best), color=NEUTRAL, label="full-sample winner (chosen with hindsight)"
    )
    ax.plot(index, np.cumsum(wf), color=BLUE, label="walk-forward: trade last window's winner")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("Cumulative net return (sum of daily returns)")
    ax.set_title(
        f"Walk-forward Sharpe {net['walkforward']['oos_annual']:.2f} "
        f"vs {net['walkforward']['mean_is_annual']:.2f} in the selection windows"
    )
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.22 * (hi_y - lo_y))
    ax.legend(loc="upper left")
    out.append(_save(fig, outdir, "e3_is_vs_oos.png"))

    # CPCV paths and minimum backtest length.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    ax = axes[0]
    paths = np.sort(arrays["cpcv_path_annual"])
    ax.scatter(
        paths, np.zeros_like(paths), s=70, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3
    )
    _vline(
        ax,
        net["cpcv"]["mean_is_annual"],
        "mean training-set Sharpe\nof the selected variant",
        y=0.95,
        ha="right",
    )
    _vline(ax, 0.0, "zero", y=0.3)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_title(f"CPCV: Sharpe of the {net['cpcv']['n_paths']} out-of-sample backtest paths")

    ax = axes[1]
    n_grid = np.geomspace(2, max(net["neff"]["raw"] * 2, 10), 80)
    target = net["sharpe"]["best_annual"]
    ax.plot(n_grid, [min_backtest_length(n, target) for n in n_grid], color=BLUE)
    ax.axhline(net["years"], color=INK2, linewidth=1.2)
    ax.annotate(
        f"sample length: {net['years']:.1f} years",
        (n_grid[0], net["years"]),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
    )
    for key, label in (("years_needed_primary", "effective N"), ("years_needed_raw", "raw N")):
        n_val = net["dsr_primary"]["n_trials"] if "primary" in key else net["neff"]["raw"]
        ax.scatter(
            [n_val],
            [net["min_btl"][key]],
            s=60,
            color=ORANGE,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=4,
        )
        ax.annotate(
            label,
            (n_val, net["min_btl"][key]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_ylim(top=max(net["years"], net["min_btl"]["years_needed_raw"]) * 1.18)
    ax.set_xscale("log")
    ax.set_xlabel("Number of independent trials")
    ax.set_ylabel("Minimum backtest length (years)")
    ax.set_title(f"Years needed before a Sharpe of {target:.2f} stops being expected by luck")
    out.append(_save(fig, outdir, "e3_cpcv_minbtl.png"))
    return out


def fig_e4(results: dict, outdir: Path) -> list[Path]:
    """E4 figure: sensitivity to the fee."""
    setup_style()
    fees = np.array(results["fees_bps"])
    lv = results["levels"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax = axes[0]
    for i, (key, label) in enumerate(
        (("naive", "naive (p < 0.05)"), ("bh", "Benjamini-Hochberg"), ("holm", "Holm"))
    ):
        ax.plot(
            fees,
            [d["survivors"][key] for d in lv],
            color=SERIES[i],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label=label,
        )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=-0.2)
    ax.set_xlabel("Fee per side (basis points)")
    ax.set_ylabel("Surviving variants (log scale above 1)")
    ax.set_title("Survivors against the fee")
    _legend_below(ax)

    ax = axes[1]
    ax.plot(
        fees,
        [d["sharpe"]["best_annual"] for d in lv],
        color=BLUE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="best net Sharpe",
    )
    ax.plot(
        fees,
        [d["dsr_primary"]["sr0_annual"] for d in lv],
        color=ORANGE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="expected best under the null",
    )
    ax.plot(
        fees,
        [d["sharpe"]["median_annual"] for d in lv],
        color=AQUA,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="median variant",
    )
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Fee per side (basis points)")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sharpe ratios against the fee")
    _legend_below(ax)

    ax = axes[2]
    ax.plot(
        fees,
        [d["pbo"]["pbo"] for d in lv],
        color=BLUE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="PBO",
    )
    ax.plot(
        fees,
        [d["dsr_primary"]["dsr"] for d in lv],
        color=ORANGE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="DSR of the best variant",
    )
    ax.plot(
        fees,
        [d["snooping"]["spa_p"] for d in lv],
        color=AQUA,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="SPA p-value",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Fee per side (basis points)")
    ax.set_ylabel("Probability")
    ax.set_title("PBO, DSR and SPA p-value against the fee")
    _legend_below(ax)
    return [_save(fig, outdir, "e4_costs.png")]


def fig_e5(results: dict, outdir: Path) -> list[Path]:
    """E5 figure: robustness of the conclusions."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    ax = axes[0, 0]
    blocks = [r["n_blocks"] for r in results["cscv"]]
    ax.plot(
        blocks,
        [r["pbo"] for r in results["cscv"]],
        color=BLUE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(blocks)
    ax.set_xlabel("Number of CSCV blocks S")
    ax.set_ylabel("PBO")
    ax.set_title("PBO against the number of blocks")

    ax = axes[0, 1]
    groups = sorted({r["n_groups"] for r in results["cpcv"]})
    for i, g in enumerate(groups):
        rows = [r for r in results["cpcv"] if r["n_groups"] == g]
        ax.plot(
            [100 * r["embargo_pct"] for r in rows],
            [r["median_path_annual"] for r in rows],
            color=SERIES[i],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label=f"{g} groups",
        )
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Embargo (% of the sample after each test block)")
    ax.set_ylabel("Median Sharpe of CPCV paths")
    ax.set_title("CPCV out-of-sample Sharpe against folds and embargo")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(min(lo_y, -0.05), hi_y + 0.25 * (hi_y - lo_y))
    ax.legend(loc="upper left", ncols=4)

    ax = axes[1, 0]
    periods = results["subperiods"]
    xpos = np.arange(len(periods))
    width = 0.2
    ax.bar(
        xpos - width / 2 - 0.02,
        [p["sharpe"]["best_annual"] for p in periods],
        width,
        color=BLUE,
        label="best Sharpe in the period",
    )
    ax.bar(
        xpos + width / 2 + 0.02,
        [p["dsr_primary"]["sr0_annual"] for p in periods],
        width,
        color=ORANGE,
        label="expected best under the null",
    )
    ax.set_xticks(xpos, [p["label"] for p in periods], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sample period: best variant against its luck benchmark")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_xlim(-0.6, len(periods) - 0.4)
    ax.legend(loc="upper left", ncols=2)

    ax = axes[1, 1]
    mb = [r["mean_block"] for r in results["block_length"]]
    ax.plot(
        mb,
        [r["spa_p"] for r in results["block_length"]],
        color=BLUE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="Hansen's SPA",
    )
    ax.plot(
        mb,
        [r["rc_p"] for r in results["block_length"]],
        color=ORANGE,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        label="Reality Check",
    )
    ax.axhline(results["alpha"], color=INK2, linewidth=1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xscale("log")
    ax.set_xticks(mb, [str(m) for m in mb])
    ax.minorticks_off()
    ax.set_xlabel("Mean block length of the stationary bootstrap (days)")
    ax.set_ylabel("p-value")
    ax.set_title("Data-snooping p-values against the block length")
    ax.legend(loc="center right")
    return [_save(fig, outdir, "e5_robustness.png")]
