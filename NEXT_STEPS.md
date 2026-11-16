# Next steps

What is unfinished, what is known to be imperfect, and what could be built next.

## Status of the four follow-up items (2026-11-17)

| Item | Status | What is left |
|---|---|---|
| Python 3.11 | **Done.** The full test-suite, lint and `make demo` pass on Python 3.11, 3.12 and 3.13 in clean virtual environments. `requirements.txt` pins exact versions for each. | Nothing. |
| GitHub Actions | **Running.** The workflow runs on every push, on Ubuntu with Python 3.11, 3.12 and 3.13 and on Windows with Python 3.12. The badge in the README shows the state of the latest run. | Nothing. |
| Binance data | **Done, by an indirect route.** The committed results use real Binance archives, each verified against its published checksum by the project's downloader. The archives reached the downloader through a local mirror, not a direct connection, because the build machine cannot reach `data.binance.vision` (DECISIONS A1). | Run `make data-check` once on your own machine. It exercises the one thing the committed run did not: the downloader talking to the live host. |
| Hourly data | **Done for E3.** E3 has been run on real one-hour Binance bars (REPORT section 5.6). | E4 and E5 on hourly bars (see extensions). |

### What has and has not been checked for the downloader

- **Checked on real files:** the archive format (millisecond timestamps in older files, microsecond in recent ones, no header row), the checksum-file format, monthly archives for every month from July 2018, and the daily-file fallback for the current month. All ten symbols, daily and hourly.
- **Checked in the test-suite against a generated replica** (`tests/test_binance_replica.py`): HTTP 404 for months before a listing, failure on a wrong checksum, files with and without a header row, the per-month cache, the pre-flight check.
- **Not checked:** a direct connection from the downloader to the live host. The downloader uses plain HTTPS requests with retries, so the remaining risks are local ones: a proxy, a firewall, or rate limiting on a slow connection.

### Reproducing the committed run on your own machine

1. `make data-check`. Every symbol should say `ok`.
2. `make data && make all`. `make all` prints the data source it used; it should say Binance. The end of the sample is pinned in `configs/base.yaml` (`data.end`), so the numbers should match the committed ones.
3. `make data-hourly && make hourly`, then `make docs` to refresh the hourly section of the documents.
4. To extend the sample, set `data.end: null`, repeat the steps, and re-read the prose of REPORT.md against the regenerated tables. Tables and numbers update automatically; most sentences that interpret them were written for the committed run. Sections 5.3 and 5.6 and the abstract matter most.

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
- **Report prose is static.** Tables and numbers regenerate; most sentences that interpret them do not.
- **The buy-and-hold comparison is not a test of timing.** It compares mean returns, so a variant is penalised for the drift it gives up while out of the market. The first extension below fixes this.
- **The gross hourly result is unexplained.** Before costs, hourly rules pass Holm and SPA against cash. The project shows that costs remove the effect, not what it is.
- **The Holm headline depends on the p-value recipe.** One variant passes with PSR-based p-values and none with the t-test (REPORT section 5.3).
- **Windows is covered by CI only.** The test-suite runs on Windows in GitHub Actions; the full pipeline has only been run on Linux.

## Extensions you could build yourself

Roughly in order of value for an interview.

1. **Better benchmarks.** This is the most direct follow-up to the results. Test each variant against buy-and-hold scaled to the variant's own average exposure, or regress its returns on the asset's returns and test the intercept (alpha). That isolates the timing term and answers the question the current comparison leaves open: is a Sharpe ratio above the asset's own evidence of anything?
2. **Hourly cost sweep.** Run E4 on the hourly matrix and find the fee at which the gross hourly result stops surviving Holm. Then run E5's sub-period split on it.
3. **Romano-Wolf StepM.** A step-down bootstrap that says *which* variants are significant while controlling the family-wise error rate and using the dependence between them. The bootstrap means are already computed in `diagnostics/bootstrap.py`; StepM is a loop that removes rejected variants and recomputes the maximum.
4. **Studentised bootstrap for SPA.** Recompute each variant's standard deviation inside every resample (needs a second matrix product with squared returns). This is the textbook cure for a noisy scale and would let you compare three versions in E1.
5. **Point-in-time universe.** Choose assets each year by market capitalisation known at the time, including coins that later died, and measure how much survivorship bias changes the long-only results.
6. **Full ONC with cluster-level selection.** Implement the recursive step, aggregate each cluster into one return series, select among clusters, and compute the DSR as Lopez de Prado (2019) does. Compare with max-matching in E1.
7. **Funding costs for shorts.** Binance publishes perpetual funding rates; charge them on short positions and re-run E3 and E4.
8. **A second asset class.** Daily futures or equity index data through the same pipeline; does the picture change when the drift is smaller?
9. **Machine-learning strategies.** Add a family whose signal is a fitted model. This is where purged cross-validation is used for its original purpose, model training.
10. **Portfolio-level selection.** Select a combination of variants (for example equal-weight top decile) instead of a single winner and repeat the walk-forward and CPCV analysis.
11. **Power curves in Sharpe units.** Calibrate the planted signal so the x-axis of the E2 figure is the true Sharpe of the best rule, estimated from one very long simulation per strength.
12. **Speed.** Batch the indicator computations per family to speed up hourly runs, and cache the correlation matrix between diagnostics.
13. **Property-based tests.** Use randomly generated price paths and parameters to fuzz the lookahead test and the CSCV implementation against the brute-force reference.
