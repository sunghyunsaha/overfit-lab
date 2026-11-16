"""Command line entry point: ``python -m overfitlab.cli <command> --config <file>``.

Commands
--------
download      fetch market data into the local cache (``--source binance|coinmetrics``)
check-data    pre-flight check of the Binance source for every configured symbol
e1 .. e5      run one experiment (results JSON + figures)
all           run e1..e5, then render the documents
figures       redraw every figure from the results and arrays already on disk
docs          render README.md / REPORT.md / INTERVIEW_PREP.md from the results JSON
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from overfitlab.config import load_config, resolve_path
from overfitlab.experiments.common import figure_dir


def _run_experiment(name: str, cfg: dict) -> None:
    from overfitlab.experiments import e1_null, e2_power, e3_real, e4_costs, e5_robustness
    from overfitlab.experiments.common import save_arrays

    start = time.time()
    modules = {"e1": e1_null, "e2": e2_power, "e3": e3_real, "e4": e4_costs, "e5": e5_robustness}
    res = modules[name].run(cfg)
    if "arrays" in res:
        save_arrays(cfg, name, res["arrays"])
    if name == "e3":
        print(f"     data source: {res['results']['data']['label']}")
    _draw_figures(name, cfg)
    print(f"[{name}] done in {time.time() - start:.0f}s")


def _draw_figures(name: str, cfg: dict) -> None:
    """Draw one experiment's figures from what is on disk (results JSON + saved arrays)."""
    import pandas as pd

    from overfitlab.experiments.common import load_arrays, load_results
    from overfitlab.reporting import figures

    outdir = figure_dir(cfg)
    results = load_results(cfg, name)
    if name == "e1":
        figures.fig_e1(results, load_arrays(cfg, "e1"), outdir)
    elif name == "e2":
        figures.fig_e2(results, outdir)
    elif name == "e3":
        arrays = load_arrays(cfg, "e3")
        arrays["index"] = pd.to_datetime(arrays["index"])
        figures.fig_e3(results, arrays, outdir)
    elif name == "e4":
        figures.fig_e4(results, outdir)
    elif name == "e5":
        figures.fig_e5(results, outdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="overfitlab", description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=["download", "check-data", "e1", "e2", "e3", "e4", "e5", "all", "figures", "docs"],
    )
    parser.add_argument(
        "--config", default=None, help="YAML file merged on top of configs/base.yaml"
    )
    parser.add_argument("--source", default="binance", choices=["binance", "coinmetrics"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)

    if args.command == "check-data":
        from overfitlab.data import binance

        try:
            rows = binance.check_source(cfg)
        except (OSError, ValueError) as err:
            print(f"Binance check failed: {err}", file=sys.stderr)
            return 1
        missing = [r["symbol"] for r in rows if not r["ok"]]
        if missing:
            print(f"no data at the start or end of the sample for: {', '.join(missing)}")
            print(
                "(the loader trims all assets to their common window, so this shortens the sample)"
            )
        return 0

    if args.command == "download":
        from overfitlab.data import binance, coinmetrics

        cache = resolve_path(cfg["data"], "cache_dir")
        module = binance if args.source == "binance" else coinmetrics
        try:
            module.download_all(cfg, cache)
        except OSError as err:  # URLError, ConnectionError, timeouts
            print(f"download from {args.source} failed: {err}", file=sys.stderr)
            print("`make all` will fall back to the next available source.", file=sys.stderr)
            return 1
        return 0

    if args.command == "docs":
        from overfitlab.reporting.render import build_documents

        for name in build_documents(cfg):
            print(f"rendered {name}")
        return 0

    if args.command == "figures":
        for name in ("e1", "e2", "e3", "e4", "e5"):
            _draw_figures(name, cfg)
        print(f"figures -> {figure_dir(cfg)}")
        return 0

    start = time.time()
    names = ["e1", "e2", "e3", "e4", "e5"] if args.command == "all" else [args.command]
    for name in names:
        _run_experiment(name, cfg)
    if args.command == "all" and cfg.get("render_docs", True):
        from overfitlab.reporting.render import build_documents

        build_documents(cfg)
    print(f"total {time.time() - start:.0f}s -> {resolve_path(cfg, 'output_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
