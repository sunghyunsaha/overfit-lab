# Next steps

What is unfinished, what is known to be imperfect, and what could be built next.

## Not done, or done but not exercised

- **The Binance downloader has not been run against the live site.** The build machine could not reach `data.binance.vision`. The URL builder, archive parser (millisecond and microsecond timestamps, with and without a header row) and checksum check are unit-tested on archives fabricated in the tests, but the first real `make data` may still surface something (a listing date, a renamed symbol, a rate limit). If a symbol has no data before its listing month, that is expected and logged.
- **No results on Binance data.** All real-data numbers in the committed README and REPORT come from the CoinMetrics fallback. Run `make data && make all` and re-read the report prose against the regenerated tables: the interpretive sentences were written for the committed run.
- **Hourly bars were never run on real data.** The 1-hour path (hourly backtest, summed to daily returns) is implemented and tested on synthetic bars only. To try it, copy `configs/full.yaml`, set `data.interval: 1h`, download, and run E3. Indicator windows are in bars, so hourly variants use much shorter horizons in calendar time, and the run is about 24 times heavier.
- **Python 3.11 was not tested.** The code uses only 3.11-compatible syntax, but it was built and tested on Python 3.13 and the exact pins (numpy 2.5, scipy 1.18) need Python 3.12 or newer. `requirements.txt` has version ranges for 3.11 that pip can resolve, but nobody has executed the test-suite on them.
- **GitHub Actions has never run.** The workflow file exists (`.github/workflows/ci.yml`) and mirrors `make lint` and `make test`, but it has not been triggered.

## Known issues and soft spots

- **Monte Carlo precision.** E1 uses a few hundred worlds per null type and E2 a few dozen per signal level, each with 1,000 variants. Rates have intervals of several percentage points. Raising `experiments.e1.mc.n_worlds` and `experiments.e2.n_worlds` in the config tightens them at the cost of runtime.
- **The size of the SPA effect is uncertain.** The long-run-studentised SPA rejected more often than the sample-s.d. version, but different sets of simulated worlds gave noticeably different rates during development. The direction is supported by a unit test; the magnitude is not pinned down.
- **The paper's DSR has almost no power in E2.** Its dispersion term grows with a real signal. The variant with theoretical dispersion is reported next to it, but the headline still uses the paper's definition. Which one should be primary is an open design question.
- **The "true Sharpe of the best variant" in E2 is biased upwards** (it is a maximum of noisy averages), as its own zero-signal row shows.
- **The clustering estimator is a simplified ONC**: average-linkage clustering, one pass, no recursive refinement, no aggregation of clusters into composite strategies.
- **CPCV summaries are noisy**: with 5 to 11 paths the median path Sharpe moves a lot between settings (see E5).
- **Purging parameters are not derived from holding periods.** A better version would compute each variant's average holding time and set the horizon from it.
- **One-sided p-values rely on the normal approximation.** They held up in the null worlds tested, but those worlds are symmetric; strongly skewed real returns could behave differently in the far tail.
- **The full-size null world in E1 is a single draw.** Its family-level p-values are one sample from a uniform distribution and will change with the seed.
- **Report prose is static.** Tables and numbers regenerate; sentences that interpret them do not.

## Extensions you could build yourself

Roughly in order of value for an interview.

1. **Romano-Wolf StepM.** A step-down bootstrap that says *which* variants are significant while controlling the family-wise error rate and using the dependence between them. The bootstrap means are already computed in `diagnostics/bootstrap.py`; StepM is a loop that removes rejected variants and recomputes the maximum.
2. **Studentised bootstrap for SPA.** Recompute each variant's standard deviation inside every resample (needs a second matrix product with squared returns). This is the textbook cure for a noisy scale and would let you compare three versions in E1.
3. **Point-in-time universe.** Choose assets each year by market capitalisation known at the time, including coins that later died, and measure how much survivorship bias changes the long-only results.
4. **Better benchmarks.** Test each variant against a buy-and-hold position scaled to the same average exposure or the same volatility, or regress strategy returns on the asset's returns and test the intercept (alpha).
5. **Full ONC with cluster-level selection.** Implement the recursive step, aggregate each cluster into one return series, select among clusters, and compute the DSR as Lopez de Prado (2019) does. Compare with max-matching in E1.
6. **Funding costs for shorts.** Binance publishes perpetual funding rates; charge them on short positions and re-run E3 and E4.
7. **A second asset class.** Daily futures or equity index data through the same pipeline; does the picture change when the drift is smaller?
8. **Machine-learning strategies.** Add a family whose signal is a fitted model. This is where purged cross-validation is used for its original purpose, model training.
9. **Portfolio-level selection.** Select a combination of variants (for example equal-weight top decile) instead of a single winner and repeat the walk-forward and CPCV analysis.
10. **Power curves in Sharpe units.** Calibrate the planted signal so the x-axis of the E2 figure is the true Sharpe of the best rule, estimated from one very long simulation per strength.
11. **Speed.** Batch the indicator computations per family to make hourly runs practical, and cache the correlation matrix between diagnostics.
12. **Property-based tests.** Use randomly generated price paths and parameters to fuzz the lookahead test and the CSCV implementation against the brute-force reference.
