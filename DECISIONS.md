# Design decisions

Every non-obvious choice, what the alternatives were, and why this one was made. Numbers quoted as evidence are in [REPORT.md](REPORT.md); this file explains the reasoning.

## A. Data

**A1. Real data comes from CoinMetrics in the committed run, not Binance.**
The spec's primary source is Binance bulk data. The build machine's network policy blocked `data.binance.vision` but allowed GitHub, where CoinMetrics publishes its community dataset. Alternatives: (a) run E3 on synthetic data only, as the spec allows; (b) use real daily prices from the reachable source. Chosen: (b), because the project's main question is about real data, with the source labelled everywhere and Binance kept as a one-command drop-in (`make data && make all`). Cost: the CoinMetrics data has one price per day (no intraday high/low, no hourly bars) and is a reference rate aggregated across venues, not one exchange's prints.

**A2. Source resolution is automatic and recorded.** `data.source: auto` picks the Binance cache if complete, else the CoinMetrics cache, else a synthetic stand-in. The choice is written into `e3.json` and rendered into README and REPORT, so a result can never be shown without its data source.

**A3. Ten assets, sample starting 2018-07-01.** The ten coins were all listed on Binance against USDT by mid-June 2018, so the same symbol list works for both sources and gives the longest common window. Alternative: start in 2018-01 with fewer coins, or add later listings (SOL, DOGE) and lose years. A longer sample matters more than more assets for Sharpe-ratio inference.

**A4. Outliers are logged, not removed.** A 30% daily move in crypto is usually real. Deleting it would make backtests look better than reality. The loader repairs only what is impossible (duplicates, missing bars, non-positive prices, high below low) and has an opt-in switch for outlier repair.

**A5. Missing bars are filled flat with zero volume.** Alternatives: drop the bar (misaligns assets) or interpolate (invents prices). A flat bar gives a zero return and cannot trigger a signal.

**A6. Diagnostics always run on daily returns; hourly strategies are aggregated.** The spec asks for daily and 1-hour data and for one T x N matrix. A matrix needs one time index, so hourly backtests are summed to daily P&L. The hourly path is implemented and tested on synthetic bars but has not been run on real hourly data (see NEXT_STEPS).

## B. Synthetic markets

**B1. The null world is a martingale in simple returns.** With zero conditional mean of the simple return, every past-dependent strategy has a true gross Sharpe ratio of exactly zero. Alternative: zero drift in log returns, which leaves a small positive expected simple return and would make the "null" slightly false for long-only rules.

**B2. Returns are clipped symmetrically at plus or minus 50% per bar.** Student-t shocks can otherwise produce a return below -100%. A symmetric clip of a symmetric distribution keeps the mean at exactly zero.

**B3. GARCH(1,1) with alpha 0.06, beta 0.92, Student-t with 6 degrees of freedom.** These give persistent volatility and a finite unconditional kurtosis of about 11, similar to daily crypto returns. With heavier tails (4 or 5 degrees of freedom) the fourth moment does not exist and the PSR formula, which uses kurtosis, has no meaning.

**B4. Assets share a common factor (correlation 0.5).** Coins move together. Independent synthetic assets would overstate the number of independent trials relative to real data.

**B5. The planted signal is a hidden AR(1) drift, measured by an oracle's Sharpe ratio.** Alternatives: AR(1) in returns (ties the signal to one lag) or a deterministic trend (too easy). A latent slow drift is what trend-following is supposed to capture, and "the Sharpe ratio of someone who could see it" is an interpretable unit. E2 also reports the true Sharpe ratio of the best price-based rule, which is what the tests are actually up against.

## C. Backtester

**C1. Signal at the close of bar t, position held over bar t+1.** Crypto trades continuously, so the close of one bar is the open of the next; the usual objection to close-to-close execution (you cannot trade at the price you just observed) is covered by the slippage charge and tested in E5 with an extra bar of delay.

**C2. Costs: 10 bps per side plus slippage proportional to trailing daily volatility.** The volatility used is measured up to the bar before the trade. The multiplier (1% of daily volatility, roughly 3 to 6 bps for these assets) is a judgement call, not an estimate from order-book data. E4 varies the fee.

**C3. Gross returns, turnover and slippage are stored separately.** Net returns are then linear in the fee, so E4 re-uses one backtest for all fee levels and there is no way for gross and net results to come from different runs.

**C4. Simple returns on constant notional, no compounding inside the Sharpe ratio.** Sharpe ratios are computed on per-bar simple returns, which is standard. Position drift within a bar is ignored.

**C5. A written-from-scratch engine rather than a library.** The spec requires it. The benefit is that the timing convention is one line (`held[lag:] = pos[:-lag]`) and can be tested directly.

**C6. Shorts cost the same as longs.** No funding or borrow cost is charged. This flatters long-short variants (LIMITATIONS).

## D. Strategy universe

**D1. Exactly 250 base variants per asset, doubled by the volatility-target overlay.** Grids were chosen to cover the conventional range of each family's parameters and to total 5,000 with ten assets. They were written into the config before any backtest was run on real data and have not been changed since.

**D2. Variants that never trade stay in the universe.** They were tried, so they count as trials. They get Sharpe 0 and p = 1 and are ignored by the bootstrap tests.

**D3. Z-scores use the log price; Donchian compares the close with the previous N bars' extremes.** Both are standard definitions; the "previous N bars" form guarantees no lookahead.

## E. Diagnostics

**E1. Per-variant p-values use the standard error under the null, not 1 - PSR(0).** PSR evaluates the standard error at the estimated Sharpe ratio using the in-sample skewness. In null worlds Bonferroni on those p-values broke its own error bound, because for variants with a handful of active bars the sample mean and sample skewness rise together and the standard error collapses. The classical t-test, `p = 1 - Phi(SR * sqrt(T-1))`, does not have the problem. PSR is still reported and still used inside the DSR, as in the papers.

**E2. The headline DSR uses the max-matching estimate of the number of trials.** The spec suggests clustering. A simplified ONC clustering is implemented and reported, but in null worlds its cluster count was far too small to reproduce the expected maximum Sharpe ratio, and a DSR built on it passed in most GARCH null worlds. The max-matching estimator answers the question the DSR actually asks (how many independent normals have the same expected maximum as this correlated family) by simulation from the empirical correlation matrix. It was the best calibrated of the four in E1. The raw count and the average-correlation formula are conservative alternatives and are also reported.

**E3. The DSR's dispersion term is the cross-sectional variance of Sharpe estimates, as in the paper.** That variance also contains real differences between variants (cost drag, or a real signal that makes some families good and others bad), which inflates the luck benchmark. The power experiment showed how far this goes: with a strong planted signal the paper's DSR hardly ever passed, because the benchmark rose with the signal. The alternative, the theoretical sampling variance 1/(T-1), is computed everywhere and reported beside the headline value. The headline keeps the paper's definition so that the number called "DSR" means what readers expect; this is a judgement call and is listed in NEXT_STEPS.

**E4. SPA is studentised by the sample standard deviation, not the long-run standard deviation.** Hansen (2005) uses the long-run estimate. In null worlds that version rejected more often than the sample-s.d. version; the long-run estimate is several times noisier and a maximum over thousands of ratios selects its under-estimates. Under the null hypothesis strategy returns are uncorrelated, so both scales estimate the same thing, and the block bootstrap still handles time dependence. The long-run p-value is reported next to the default.

**E5. One set of bootstrap resamples serves the Reality Check and all SPA variants.** They only need resampled column means, which are a single matrix product of resampling counts with the return matrix. This is why a 5,000-column bootstrap takes seconds.

**E6. Stationary bootstrap with mean block 10 days.** Strategy returns have little autocorrelation but volatility clusters. E5 varies the block length from 5 to 40.

**E7. CSCV with 16 blocks and the Sharpe ratio as the performance measure.** Sixteen blocks give 12,870 splits, the paper's own example size. E5 varies S. Ties in the out-of-sample ranking get the average rank, and a logit of exactly zero counts as overfit.

**E8. What purging means here.** The observations are one-day strategy returns, whose own label window is one day. The dependence that leaks across a train/test boundary comes from positions that persist and indicators that overlap. The purge horizon (20 bars) and embargo (1% of the sample) are therefore modelling choices, not derived quantities; E5 varies the embargo and the number of groups.

**E9. Haircut Sharpe ratios use the actual p-values of all variants.** Harvey and Liu simulate the unobserved tests from a model fitted to published research. Here all trials are observed, which is simpler and more direct, but it is not the paper's exact procedure.

**E10. The long-run variance sum is truncated.** Lags whose kernel weight is below 1e-8 are dropped, and the wrap-around term of the stationary-bootstrap formula is omitted. A test checks the result against the variance of actual bootstrap means.

**E11. Tests against a cash benchmark, plus an active-return check.** PSR, DSR, the Reality Check and SPA are all run against zero. Because long-only variants on rising assets earn market drift, E3 repeats the light diagnostics on each variant's return minus buy-and-hold of its own asset.

## F. Experiments and reporting

**F1. Monte Carlo worlds are smaller than the full run.** E1 and E2 simulate hundreds of worlds; each uses 2 assets (1,000 variants). Full-size worlds would take hours. The full-size null world is run once for the headline counts.

**F2. E1's Monte Carlo uses gross returns, E2 uses net.** Gross returns put the null exactly on the boundary (true Sharpe zero), which is the hardest case for size. E2 asks whether a tradeable edge is detected, so costs are included.

**F3. "Survive" is defined per correction, and the headline quotes Holm.** Holm controls the probability of any false discovery, which matches the question "can I trust this specific backtest?" Benjamini-Hochberg counts are always shown next to it.

**F4. README, REPORT and INTERVIEW_PREP are rendered from templates.** `docs/templates/*.tmpl` contain placeholders that are filled from `reports/results/*.json`. A test fails if the committed documents differ from the rendered templates. Interpretive prose was written for the committed run and says so.

**F5. Figures are redrawn from saved results.** Experiments save the arrays their figures need, so `make figures` restyles without recomputation.

**F6. Seeds.** One base seed in the config; each experiment and each simulated world derives its own seed by a fixed offset, so results are reproducible and worlds are independent.

**F7. Float32 on disk, float64 in memory.** The saved matrices are for inspection; all diagnostics run on the float64 matrices built in memory.

**F8. Order of work, stated plainly.** The per-variant p-value correction (E1 above) was made before the real-data experiment was first run. The SPA scale change (E4 above) was made after a first real-data run, and was decided from the null-world Monte Carlo alone. Its effect on the real data is reported in REPORT section 5.3: it moved the SPA p-value towards significance, and no conclusion changed. The strategy grids, cost settings and sample window were never changed after the first real-data run.
