# Limitations

What this project does not show, and where its results should not be stretched.

## Data

- **Survivorship bias in the asset list.** The ten coins were chosen because they were large and long-lived. Coins that collapsed or were delisted are absent, which flatters long-only results. A proper study would fix the universe at each point in time using only information available then.
- **The committed run uses close-only daily reference prices.** They come from the CoinMetrics community dataset, not from an exchange order book. There is no intraday high or low (Donchian channels are built from closes), no bid-ask spread, and the price is an aggregate that nobody could trade at exactly.
- **Hourly data was not analysed.** The pipeline supports 1-hour bars and aggregates them to daily returns, and this path is tested on synthetic bars, but no real hourly data was available on the build machine.
- **One asset class, one period.** Crypto from mid-2018 onwards contains two large bull markets and two deep drawdowns. Results say nothing about equities, futures or other periods.
- **Short history.** About eight years of daily data is roughly 2,900 observations. Sharpe ratios estimated on that much data have a standard error of about 0.35 in annualised terms, so only large effects can be resolved at all.
- **The data ends when the source snapshot ends.** The fallback dataset stopped updating before the build date, so the most recent months are missing.

## Backtest realism

- **Bar data simplifications.** Signals and fills both use the bar close. Real execution happens after the signal at an unknown price, with latency and partial fills.
- **Costs are a model.** A flat fee plus slippage proportional to volatility. No market impact, no dependence on order size or liquidity, no exchange-specific fee tiers.
- **Shorting is free.** Long-short variants pay no funding or borrow cost and face no margin calls or liquidations. In crypto, perpetual-swap funding can be large exactly when trends are strong.
- **No capacity analysis.** Every variant trades as if its size were negligible.
- **No portfolio effects.** Each variant trades one asset in isolation. There is no risk budgeting across assets and no portfolio-level leverage constraint.
- **Leverage up to 2x in the volatility-target overlay** is assumed to be available at no cost.

## Statistics

- **The benchmark is cash.** The main tests ask whether expected return is positive. Long-only rules on rising assets pass that test by holding the asset. The active-return analysis in E3 addresses this but uses a simple return difference, not a risk-adjusted comparison.
- **The universe is small and conventional.** Five technical families and 5,000 variants. The conclusions are about this search. A different search (other features, machine-learning models, cross-sectional signals) could behave differently.
- **The paper's DSR loses power when a signal is real.** Its luck benchmark uses the cross-sectional spread of Sharpe ratios, which a real signal widens (REPORT section 5.2). A low DSR here is therefore weaker evidence against skill than it looks; the variant with theoretical dispersion is reported next to it and does not pass either.
- **Only the reported trials are counted.** The Deflated Sharpe Ratio corrects for the 5,000 variants in the matrix. It cannot correct for choices made outside it, such as which families to include, and the families here were chosen because the literature discusses them, which is itself a form of selection.
- **Effective number of trials is an estimate.** The four estimators disagree by orders of magnitude. The one used for the headline was the best calibrated in null worlds, but those worlds have a simpler correlation structure than real data.
- **Monte Carlo sampling error.** E1 and E2 use hundreds of simulated worlds per setting, so rejection rates carry intervals of a few percentage points, and the simulated worlds are smaller than the real-data run.
- **Null worlds are stylised.** GBM and GARCH with a single common factor. Real returns have jumps, regime changes, asymmetric volatility and time-varying correlations that the simulations do not reproduce.
- **Stationarity is assumed throughout.** Sharpe-ratio inference, the bootstrap and cross-validation all treat the sample as draws from one stable process. Crypto markets changed structurally over the sample.
- **Purging parameters are judgement calls.** The label horizon and embargo are not derived from the strategies' actual holding periods.
- **PBO measures rank stability, not profitability.** A low PBO means the in-sample winner tends to stay in the top half out of sample. It does not mean the winner makes money, and here part of that stability comes from shared market exposure.
- **Simplified methods.** The clustering estimator is a reduced form of ONC, the haircut uses observed p-values in place of the paper's simulation, and the long-run variance is truncated. Each is flagged where it appears.

## Scope

- **This is not a trading system.** Nothing here is evidence that any variant should be traded, and nothing here is financial advice.
- **A negative result about a search is not a negative result about a market.** Finding that these rules do not survive correction does not show that crypto markets are efficient. It shows that this evidence cannot distinguish these rules from luck.
