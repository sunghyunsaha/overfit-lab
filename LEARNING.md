# LEARNING.md: study guide

This guide teaches every part of the project well enough to defend it in an interview. It assumes strong maths (calculus, probability, basic statistics) and no finance background.

**How to use it.** Each module has the same layout: what it does in plain language, the maths, the key formulas with every symbol defined, where the code lives, common mistakes, five interview questions with model answers, and a five-question self-quiz with hidden answers. Read a module, open the file it points to, then do the quiz without looking. Numbers from the actual runs are in [REPORT.md](REPORT.md); this guide deliberately uses small hand-checkable examples instead. Where modules 12 to 16 describe what an experiment found, they describe the committed run (see the "Data used" note in the README); if you re-run on other data, check those sentences against the regenerated report.

**Modules**

1. [Data](#1-data)
2. [The backtester](#2-the-backtester)
3. [The strategy universe](#3-the-strategy-universe)
4. [Naive selection and walk-forward decay](#4-naive-selection-and-walk-forward-decay)
5. [Sharpe ratio inference: PSR and MinTRL](#5-sharpe-ratio-inference-psr-and-mintrl)
6. [Expected maximum Sharpe, the Deflated Sharpe Ratio and effective trials](#6-expected-maximum-sharpe-the-deflated-sharpe-ratio-and-effective-trials)
7. [Probability of Backtest Overfitting (CSCV)](#7-probability-of-backtest-overfitting-cscv)
8. [Purged k-fold and CPCV](#8-purged-k-fold-and-cpcv)
9. [Reality Check and SPA](#9-reality-check-and-spa)
10. [Multiple-testing corrections and the haircut Sharpe ratio](#10-multiple-testing-corrections-and-the-haircut-sharpe-ratio)
11. [Minimum Backtest Length](#11-minimum-backtest-length)
12. [E1: the null world](#12-e1-the-null-world)
13. [E2: power](#13-e2-power)
14. [E3: real data](#14-e3-real-data)
15. [E4: cost sensitivity](#15-e4-cost-sensitivity)
16. [E5: robustness](#16-e5-robustness)

**Vocabulary you need first**

| Term | Meaning |
|---|---|
| Bar | One row of price data for a fixed interval (here one day): open, high, low, close, volume (OHLCV). |
| Return | $`r_t = C_t / C_{t-1} - 1`$, the fractional price change over bar t. |
| Position | The fraction of your capital held in the asset: +1 fully long (you profit if the price rises), -1 fully short (you profit if it falls), 0 flat. |
| Backtest | Applying a trading rule to historical data to see what it would have earned. |
| Sharpe ratio (SR) | Mean return divided by the standard deviation of returns. A signal-to-noise ratio. Annualised by multiplying the daily value by $`\sqrt{365}`$ (crypto trades every day). |
| Basis point (bp) | 0.01%. A fee of 10 bps is 0.1% of the amount traded. |
| In-sample (IS) / out-of-sample (OOS) | Data used to choose the strategy / data not used to choose it. |
| Overfitting | Choosing a strategy because it fitted the noise in the in-sample data. |
| Null hypothesis | Here: "the strategy has no skill", meaning its true Sharpe ratio is zero or negative. |
| p-value | Probability, if the null is true, of a result at least as good as the one observed. |

---

## 1. Data

### What it does

Three jobs. (a) Make **synthetic** markets where we know the truth, so that the statistical tools can be tested. (b) **Download** real crypto prices and cache them locally. (c) **Validate** every price series and log each repair, so no data is changed silently.

### The maths

A synthetic return for asset i on bar t is

```math
R_{i,t} = \mu_{i,t} + \sigma_{i,t} z_{i,t}
```

- $`z_{i,t}`$: a random shock with mean 0 and variance 1. It is $`\sqrt{\rho} f_t + \sqrt{1-\rho}\, e_{i,t}`$, where $`f_t`$ is shared by all assets and $`e_{i,t}`$ is private, so any two assets have correlation $`\rho`$.
- $`\sigma_{i,t}`$: the volatility (standard deviation) of that bar.
- $`\mu_{i,t}`$: the expected return of that bar.

**GBM generator.** $`\sigma`$ constant, z Gaussian, $`\mu = 0`$. A pure random walk.

**GARCH generator.** Volatility depends on its own past:

```math
\sigma^2_t = \omega + \alpha\,\varepsilon^2_{t-1} + \beta\,\sigma^2_{t-1}, \qquad \varepsilon_t = \sigma_t z_t
```

A large shock yesterday raises variance today ($`\alpha`$), and high variance persists ($`\beta`$). This is "volatility clustering". The long-run variance is $`\omega / (1-\alpha-\beta)`$, so we set $`\omega`$ to hit a target volatility. The shocks are Student-t with 6 degrees of freedom, rescaled to variance 1, which gives fat tails (extreme days are far more common than under a normal distribution).

**Planted signal.** $`\mu_{i,t} = s\,\sigma_{i,t}\,x_{i,t}`$ where x is a hidden AR(1) process, $`x_t = \varphi x_{t-1} + \sqrt{1-\varphi^2}\,\eta_t`$, with variance 1. Because x is persistent, a positive drift today tends to be followed by a positive drift tomorrow, and past returns carry information about future returns. An oracle holding position $`x_t/\sigma_t`$ earns $`s x_t^2 + x_t z_t`$ per bar, which has mean s and standard deviation about 1: a per-bar Sharpe ratio of s. We quote strength as $`s\sqrt{365}`$, the oracle's annualised Sharpe ratio.

**Why the null is exact.** With $`\mu = 0`$ the price is a martingale: $`E[R_t \mid \text{past}] = 0`$. A strategy's return is $`w_{t-1} R_t`$ where $`w_{t-1}`$ depends only on the past, so $`E[w_{t-1}R_t] = E[w_{t-1}\,E[R_t \mid \text{past}]] = 0`$. Every strategy has true gross Sharpe ratio zero, whatever it does.

**High and low.** Given a bar's log return $`\ell`$ and variance v, the maximum of a Brownian bridge from 0 to $`\ell`$ can be sampled exactly as $`\tfrac12(\ell + \sqrt{\ell^2 - 2v\ln U})`$ with U uniform. This guarantees high ≥ max(open, close).

**Outlier rule in the loader.** A robust scale is $`1.4826 \times \text{MAD}`$ (median absolute deviation), which equals the standard deviation for normal data but is not pulled up by the outliers themselves. A bar is flagged if its log return is more than 12 of these away from the rolling median.

### Where it lives

| File | Function | Role |
|---|---|---|
| `data/synthetic.py` | `simulate_returns` | the model above; returns the returns, volatilities and hidden state |
| | `returns_to_ohlcv` | turns a return path into OHLCV bars |
| | `simulate_market` | several assets at once |
| `data/binance.py` | `kline_url`, `parse_kline_zip`, `verify_checksum`, `download_symbol`, `check_source` | monthly and daily archives, SHA-256 check, millisecond/microsecond timestamps, per-month cache, pre-flight check (`make data-check`) |
| `data/coinmetrics.py` | `parse_coinmetrics_csv` | fallback source, close-only bars |
| `data/loader.py` | `validate_ohlcv` | duplicates, bad prices, gaps, OHLC consistency, zero volume, outliers |
| | `load_market` | picks the source, validates, aligns all assets on one calendar |

### Common mistakes and how the code avoids them

- **A "null" that is not null.** If log returns have zero mean, simple returns have a small positive mean and long-only strategies have real edge. The generator makes the simple return's mean exactly zero.
- **Negative prices from fat tails.** A Student-t shock can give a return below -100%. Returns are clipped symmetrically at ±50%, which keeps the mean at zero.
- **Kurtosis that does not exist.** With 4 degrees of freedom the fourth moment is infinite and PSR's kurtosis term is meaningless. The parameters are chosen so the unconditional kurtosis is finite (about 11).
- **Silent data repair.** Every fix is recorded in a `ValidationLog` and written to `reports/results/data_validation.json`.
- **Deleting real crashes.** Outliers are logged but not altered by default.
- **Timestamp unit change.** Binance spot files switched from milliseconds to microseconds in 2025; the parser detects the unit from the magnitude.
- **Misaligned assets.** All assets are trimmed to the common window so every column of the matrix has the same T rows.
- **A downloader that has only ever been unit-tested.** `tests/test_binance_replica.py` starts a local web server that copies the layout of the real site and runs the actual download code against it, including missing months (HTTP 404), a wrong checksum, and a month whose monthly archive is not published yet.

### Interview questions

**Q1. Why build synthetic data at all when you have real data?**
On real data nobody knows which strategies are truly good, so you cannot tell whether a statistical test is working. On a simulated martingale every strategy is worthless by construction, so anything a test flags is a false positive and its error rate can be measured. With a planted signal the opposite check is possible: does the test find something that is really there?

**Q2. Why GARCH with Student-t shocks rather than just Gaussian noise?**
Real returns have volatility clustering and fat tails. Many formulas (the t-test, PSR) lean on approximate normality, so the tools should be tested under conditions at least as awkward as reality. Both nulls are run so that differences between them can be seen.

**Q3. How do you know your null world has no signal?**
By construction the conditional mean of the simple return is zero, which makes any past-dependent strategy's expected return zero. It is also tested: over 200,000 simulated bars the mean is within sampling error of zero and autocorrelations at lags 1, 5 and 20 are below 0.02 (`tests/test_synthetic.py`).

**Q4. What does "signal strength 2" mean in your power experiment?**
An oracle who could see the hidden drift would earn an annualised Sharpe ratio of about 2. Strategies only see prices and recover a fraction of that; the report gives the true Sharpe ratio of the best price-based rule at each strength.

**Q5. Your real data has outliers. Why did you not remove them?**
In crypto, extreme daily moves are usually genuine and a live strategy would have experienced them. Removing them biases backtests upwards. The loader removes only impossible values and reports the rest.

### Self-quiz

1. In GARCH(1,1), what is the long-run variance in terms of ω, α, β?
2. Why is a strategy's expected gross return zero in the null world even if it uses a clever rule?
3. What is the correlation between two synthetic assets when ρ = 0.5?
4. Why is MAD used for outlier detection instead of the standard deviation?
5. What would go wrong if missing bars were dropped instead of filled?

<details><summary>Answers</summary>

1. ω / (1 - α - β).
2. The position is fixed before the return is drawn and the return has conditional mean zero, so E[position × return] = 0 by the tower property.
3. 0.5: the shared factor has weight √ρ in each asset, so the covariance of the unit-variance shocks is ρ.
4. Outliers inflate the standard deviation and can hide themselves; the median-based scale is barely affected by them.
5. Assets would have different numbers of rows and the T x N matrix could not be formed; dates would no longer line up across columns.

</details>

---

## 2. The backtester

### What it does

Takes a price series and a series of target positions, and returns what the strategy would have earned each bar, before and after trading costs. It is deliberately tiny so that every line can be checked.

### The maths

Let $`C_t`$ be the close of bar t and $`w_t`$ the position the strategy **decides** at the close of bar t.

```math
r_t = \frac{C_t}{C_{t-1}} - 1, \qquad \text{held}_t = w_{t-1}, \qquad g_t = \text{held}_t\, r_t
```

- $`r_t`$: the asset's return over bar t.
- $`\text{held}_t`$: the position actually carried through bar t. It is yesterday's decision. This one-bar lag is what prevents the strategy from using a price to decide a trade and also profiting from that same price.
- $`g_t`$: gross strategy return.

Costs are charged when the position changes:

```math
\text{turnover}_t = |\text{held}_t - \text{held}_{t-1}|, \qquad \text{net}_t = g_t - \text{turnover}_t \times (\text{fee} + \text{slip}_t)
```

- turnover: how much was traded, as a fraction of capital. Going from +1 to -1 is a turnover of 2.
- fee: 10 bps = 0.001 per unit traded.
- $`\text{slip}_t = k \times \hat\sigma^{\text{daily}}_{t-1}`$: slippage, the gap between the price you see and the price you get, modelled as a fraction k of recent daily volatility, measured up to the bar before the trade.

**Worked example** (this is `tests/test_engine.py`). Closes 100, 110, 99, 108.9, 108.9, so returns are 0, +10%, -10%, +10%, 0. Decisions w = 1, 1, -1, 0, 0.

| t | r | held = w[t-1] | gross | turnover | cost at 10 bps | net |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | +0.10 | 1 | +0.10 | 1 | 0.001 | +0.099 |
| 2 | -0.10 | 1 | -0.10 | 0 | 0 | -0.100 |
| 3 | +0.10 | -1 | -0.10 | 2 | 0.002 | -0.102 |
| 4 | 0 | 0 | 0 | 1 | 0.001 | -0.001 |

Because `net = gross - fee × turnover - slippage`, net returns are linear in the fee. Storing the three pieces separately lets the cost experiment change the fee without re-running anything.

### Where it lives

| File | Function | Role |
|---|---|---|
| `backtest/indicators.py` | `sma`, `ema`, `rolling_std`, `rolling_max`, `rolling_min`, `rsi` | rolling indicators; element t uses inputs 0..t only |
| | `simple_returns`, `trailing_vol` | returns and the volatility used for slippage |
| `backtest/engine.py` | `backtest` | the lag, gross returns, turnover, slippage |
| | `slippage_rate` | slippage per unit traded, from past volatility |
| | `BacktestResult.net` | net returns for any fee |

The line that matters most is `held[lag:] = pos[:-lag]`.

### Common mistakes and how the code avoids them

- **Lookahead bias**: using information from bar t to earn bar t's return. `backtest` refuses a lag below 1, and `tests/test_lookahead.py` replaces every price after a cut point and checks that all positions and returns up to it are bit-for-bit unchanged.
- **Indicators that peek**: a centred moving average uses future bars. Every rolling function writes its result at the last bar of its window.
- **Free trading**: ignoring costs makes high-turnover strategies look best. Costs are charged on every change of position, including the first entry.
- **Costs that use the future**: slippage on bar t uses volatility up to t-1 (tested).
- **Double counting a flip**: +1 to -1 is two units of turnover, not one.
- **Mixing gross and net from different runs**: one run produces all three matrices.

### Interview questions

**Q1. Walk me through exactly when a trade happens in your backtest.**
The signal is computed at the close of bar t from data up to and including bar t. The position is assumed to be established at that close and is held through bar t+1, earning the close-to-close return of bar t+1. Costs for the change are charged to bar t+1. Crypto trades continuously, so the close of one bar is effectively the open of the next; the robustness experiment also delays execution by one more bar.

**Q2. How do you know there is no lookahead?**
Structurally, positions are shifted by one bar before being multiplied by returns. Empirically, a test tampers with all data after bar t for several t and asserts that every variant's positions and returns up to t are identical. The test also checks that the future part did change, so it cannot pass vacuously.

**Q3. Why model slippage as proportional to volatility?**
Spreads and price impact widen when markets are volatile, so a fixed slippage understates costs in exactly the periods when trend strategies trade most. It is still a crude model with one free parameter and no dependence on trade size.

**Q4. Why simple returns instead of log returns?**
A portfolio's return is the weighted sum of simple returns of its holdings, so position × simple return is exact for one bar. Log returns add across time but not across positions. Sharpe ratios are conventionally computed on simple per-period returns.

**Q5. What is the biggest unrealistic assumption in the engine?**
Trading at the close price with only a modelled cost, at any size, and shorting for free. Real fills come later at an unknown price, impact grows with size, and shorts pay funding.

### Self-quiz

1. If the decision series is w = [0, 1, 1, 0], what is the held series?
2. A strategy flips from -1 to +1. What is the turnover and the cost at 10 bps?
3. Why is the cost of the first entry charged on the bar after the decision?
4. Net returns at 0 and 30 bps are known. How do you get net returns at 15 bps without re-running?
5. Would a centred 5-bar moving average pass the lookahead test? Why?

<details><summary>Answers</summary>

1. [0, 0, 1, 1].
2. Turnover 2, cost 0.002 (20 bps of capital), plus slippage.
3. Because that is the first bar in which the new position is held; the trade happens at the boundary between the two bars.
4. Average them: net is linear in the fee.
5. No. It uses two future bars, so changing data after t changes the indicator at t.

</details>

---

## 3. The strategy universe

### What it does

Generates the 5,000 strategy variants and stacks their daily returns into one matrix with T rows (days) and N columns (variants). A variant is an asset, a rule family, a parameter set and a yes/no for the volatility-target overlay.

### The maths

All rules output a position in {-1, 0, +1} at each close.

| Family | Rule (long side; shorts mirror it) | Parameters |
|---|---|---|
| MA crossover | long while fast average > slow average | fast, slow, SMA or EMA |
| Time-series momentum | long while $`C_t / C_{t-L} > 1`$ | lookback L |
| Z-score mean reversion | $`z_t = (\ln C_t - m_t)/s_t`$ with rolling mean m and standard deviation s; go long when z < -entry, exit when z ≥ -exit | window, entry, exit |
| Donchian breakout | go long when $`C_t`$ exceeds the highest high of the previous N bars; exit when it falls below the lowest low of the previous M bars | N, M = N × exit fraction |
| RSI | $`RSI = 100 \cdot \frac{\text{avg gain}}{\text{avg gain} + \text{avg loss}}`$ with Wilder smoothing; reversal style buys below a band, trend style buys above 100 - band, both exit at 50 | period, band, style |

Trend rules (MA, momentum, Donchian, RSI-trend) bet that moves continue. Mean-reversion rules (z-score, RSI-reversal) bet that moves reverse. On the same asset the two groups are negatively correlated.

**EMA**: $`y_t = a x_t + (1-a) y_{t-1}`$ with $`a = 2/(\text{span}+1)`$.

**Stateful rules.** "Enter on one condition, stay in until another" depends on history. It is vectorised by writing +1 where an entry fires, 0 where an exit fires and "no event" elsewhere, then carrying the last event forward (`_hold_last_event`).

**Volatility-target overlay.** Multiply the position by

```math
\text{scale}_t = \min\!\left(\frac{\sigma_{\text{target}}}{\hat\sigma_t},\ L_{\max}\right)
```

where $`\hat\sigma_t`$ is the annualised volatility of the last 30 returns, the target is 40% a year and the cap $`L_{\max}`$ is 2. The position shrinks when the market is wild and grows when it is calm.

**Counting.** 68 + 26 + 80 + 28 + 48 = 250 base variants, × 2 overlays × 10 assets = 5,000.

### Where it lives

| File | Function | Role |
|---|---|---|
| `backtest/strategies.py` | `ma_cross`, `tsmom`, `zscore_mr`, `donchian`, `rsi_threshold` | one function per family |
| | `_hold_last_event`, `_two_legs` | the entry/exit state machine |
| | `vol_target_scale` | the overlay |
| | `expand_grid` | turns the YAML grids into a list of (family, parameters) |
| `backtest/universe.py` | `asset_positions` | all 500 position series for one asset |
| | `build_universe` | backtests everything, returns a `Universe` (gross, turnover, slippage, metadata) |
| `configs/base.yaml` | `universe:` | the grids |

### Common mistakes and how the code avoids them

- **Tuning the grid after seeing results.** The grids are in a config file, fixed before real data was loaded; changing them means changing a tracked file.
- **Trading on unseasoned indicators.** Warm-up values are NaN and produce a flat position.
- **Breakout rules that include the current bar in the channel.** Then the close can never exceed the channel. The channel uses the previous N bars.
- **Dropping variants that did nothing.** Variants that never traded were still tried, so they stay in the count.
- **Forgetting that variants are correlated.** Addressed by the effective number of trials (module 6).

### Interview questions

**Q1. Why these five families?**
They are the standard technical rules in the literature and in practice, they cover both trend and mean reversion, and each has a small number of interpretable parameters. The aim was a realistic search space, not good strategies.

**Q2. Your variants are highly correlated. Is 5,000 a meaningful number?**
Not as a count of independent bets. That is why the project estimates the effective number of trials from the correlation matrix, and why the bootstrap tests resample whole rows so that the dependence between variants is preserved.

**Q3. What does the volatility-target overlay do to a Sharpe ratio?**
Scaling by a constant leaves the Sharpe ratio unchanged; scaling by a time-varying factor changes it. If volatility is persistent and expected returns do not rise proportionally with it, cutting exposure in volatile periods raises the Sharpe ratio. It also adds turnover.

**Q4. How do you vectorise a rule with entry and exit thresholds?**
Mark entries and exits as events, leave other bars empty and forward-fill the last event. The forward-fill is a running maximum of event indices, which is one NumPy call.

**Q5. How would you add a new family?**
Write a function from prices and parameters to positions, register it in `FAMILIES`, add a grid to the config. The lookahead test then covers it automatically because it runs over the whole grid.

### Self-quiz

1. An MA crossover with fast = slow. Why is that excluded?
2. For a long-short Donchian rule with exit fraction 1.0, when is the position ever flat?
3. Why are z-scores computed on the log price?
4. What is the EMA weight a for span 9?
5. Realised volatility doubles. What happens to a volatility-targeted position that was at scale 0.5?

<details><summary>Answers</summary>

1. The two averages are identical, so the signal is always zero; the grid requires fast < slow.
2. Only before the first breakout. After that it is always long or short (stop-and-reverse).
3. So that a given percentage deviation gives the same z whatever the price level.
4. 2 / (9 + 1) = 0.2.
5. It halves to 0.25.

</details>

---

## 4. Naive selection and walk-forward decay

### What it does

Shows the problem the whole project is about. Pick the variant with the best Sharpe ratio in one part of the data, then look at how it does in a different part. The gap is the "decay".

### The maths

**Regression to the mean.** Write each variant's estimated Sharpe ratio as truth plus noise, $`\widehat{SR}_k = SR_k + \epsilon_k`$. The variant with the largest estimate was chosen partly because its noise term was large. Out of sample the noise is redrawn with mean zero, so the expected OOS value is closer to the truth:

```math
E[\widehat{SR}^{OOS}_k \mid \widehat{SR}^{IS}_k] \approx \overline{SR} + b\,(\widehat{SR}^{IS}_k - \overline{SR}), \qquad b = \frac{\text{Var}(SR_k)}{\text{Var}(SR_k) + \text{Var}(\epsilon_k)}
```

- b is the slope of a regression of OOS on IS Sharpe ratios across variants. b = 1 means performance persists perfectly; b = 0 means in-sample ranking is pure noise.

**Split decay.** Cut the sample in half, compute both Sharpe ratios for every variant, report the slope, the rank correlation, and what happened to the IS winner and the IS top 50.

**Walk-forward.** Window j trains on days $`[s_j, s_j + 730)`$, picks the best variant there, and trades it on the next 90 days. Then $`s_{j+1} = s_j + 90`$. The out-of-sample pieces are stitched into one return series and its Sharpe ratio is compared with the average Sharpe ratio the picks had when they were chosen.

### Where it lives

`diagnostics/walkforward.py`: `split_decay` and `walk_forward`. Called from `experiments/common.py: diagnose`.

### Common mistakes and how the code avoids them

- **Reporting the in-sample Sharpe of the winner as the strategy's Sharpe.** The walk-forward output always shows both numbers side by side.
- **Letting test data into the selection.** `walk_forward` only reads training rows to pick; a test rewrites the future and checks the picks do not change.
- **Reading one split as proof.** A single cut point is one draw. CSCV (module 7) and CPCV (module 8) repeat the idea over many splits.
- **Treating walk-forward as immune to overfitting.** If you try many walk-forward configurations and keep the best, you have overfitted the walk-forward.

### Interview questions

**Q1. Why does the best backtest disappoint out of sample even without any bug?**
Selection. The maximum of many noisy estimates is biased upwards because it is chosen where noise happened to be positive. Out of sample the noise is fresh. This is regression to the mean, and it is larger when there are more candidates and when their true differences are small compared with the noise.

**Q2. What does a slope of 0.2 between first-half and second-half Sharpe ratios tell you?**
Only about a fifth of the cross-sectional differences in in-sample Sharpe carry over; the rest is noise. A strategy that is one unit above average in sample is expected to be about 0.2 units above average afterwards.

**Q3. What is walk-forward analysis and what is its weakness?**
It repeatedly selects on a trailing window and trades the next window, which mimics how a researcher would have operated in real time. Weaknesses: it tests only one historical path, results depend on window lengths, and early data is never used for testing.

**Q4. Is walk-forward an unbiased estimate of future performance?**
It is an honest estimate of how the selection rule performed on that one history, provided nothing was tuned on its output. It is noisy, and it describes the rule "pick the recent winner", not any single variant.

**Q5. If the split slope were 1, would overfitting be impossible?**
It would mean in-sample differences are entirely real, so selecting on them is sound. The winner could still fail in the future if the market changes, which is a different problem (non-stationarity).

### Self-quiz

1. In the regression-to-the-mean formula, what is b if all variants have the same true Sharpe ratio?
2. With 730 training days, 90 test days and 2,900 days of data, roughly how many walk-forward windows are there?
3. Why is the top-50 average a more stable summary than the single winner?
4. The walk-forward OOS Sharpe is far below the IS Sharpe of the picks. Name two possible causes.
5. Why can walk-forward not test the first two years of data?

<details><summary>Answers</summary>

1. Zero: Var(SR) = 0, so nothing persists.
2. (2900 - 730) / 90 ≈ 24.
3. Averaging 50 variants reduces the noise in the out-of-sample estimate.
4. Selection on noise (overfitting) and a change in market behaviour between windows.
5. They are needed as the first training window; no earlier data exists to select from.

</details>

---

## 5. Sharpe ratio inference: PSR and MinTRL

### What it does

A Sharpe ratio computed from data is an estimate with error bars. The Probabilistic Sharpe Ratio turns an estimate into a probability that the true Sharpe ratio is above a benchmark, taking into account how many observations there are and how non-normal the returns are. The Minimum Track Record Length inverts it: how much data is needed for a given confidence.

### The maths

For returns $`r_1,\dots,r_T`$ with sample mean $`\bar r`$ and standard deviation $`\hat\sigma`$, the estimate is $`\widehat{SR} = \bar r / \hat\sigma`$.

**Why it has error.** $`\bar r`$ has standard error $`\sigma/\sqrt{T}`$, so $`\widehat{SR}`$ has standard error of about $`1/\sqrt{T}`$ when SR is small. In annual terms: with y years of data the annualised Sharpe ratio has a standard error of about $`1/\sqrt{y}`$. Eight years gives ±0.35. That single fact explains most of this project.

**Full formula** (Mertens 2002), from the delta method applied to the ratio of two estimated moments:

```math
\hat\sigma(\widehat{SR}) = \sqrt{\frac{1 - \gamma_3\, SR + \frac{\gamma_4 - 1}{4}\, SR^2}{T - 1}}
```

- $`\gamma_3`$: skewness, $`E[(r-\mu)^3]/\sigma^3`$. Negative means occasional large losses.
- $`\gamma_4`$: kurtosis, $`E[(r-\mu)^4]/\sigma^4`$. Equal to 3 for a normal distribution; larger means fatter tails.
- Negative skew and fat tails make the estimate less precise. Both corrections vanish when SR = 0.

**PSR** (Bailey and Lopez de Prado 2012):

```math
PSR(SR^*) = \Phi\!\left(\frac{\widehat{SR} - SR^*}{\hat\sigma(\widehat{SR})}\right)
```

- $`\Phi`$: the standard normal cumulative distribution function.
- $`SR^*`$: the benchmark Sharpe ratio to beat (0 for "any skill").

**Worked examples.**
(a) Daily SR 0.1, T = 101, normal returns: $`\hat\sigma = \sqrt{(1 + 0.005)/100} = 0.10025`$, z = 0.9975, PSR(0) = 0.8407.
(b) Daily SR 0.05 (annualised 0.96) over two years, T = 730: z ≈ 0.05 × √729 = 1.35, PSR(0) = 0.911. An annualised Sharpe near 1 over two years is not significant at 95%.

**MinTRL**: set PSR = 1 - α and solve for T:

```math
MinTRL = 1 + \left(1 - \gamma_3 SR + \frac{\gamma_4-1}{4}SR^2\right)\left(\frac{z_{1-\alpha}}{SR - SR^*}\right)^2
```

For an annualised Sharpe of 1 (daily 0.0523) and normal returns at 95%: about 990 days, 2.7 years.

**The p-value used for multiple testing.** Under the null SR = 0 the standard error is $`1/\sqrt{T-1}`$ regardless of skewness and kurtosis, so $`p = 1 - \Phi(\widehat{SR}\sqrt{T-1})`$. This is the classical t-test. The project uses it instead of 1 - PSR(0) for the corrections, because PSR plugs in the in-sample skewness and that fails for strategies that are almost never in the market (see module 12).

### Where it lives

`diagnostics/sharpe.py`: `return_moments`, `sharpe_std_error`, `probabilistic_sharpe_ratio`, `sharpe_pvalue`, `min_track_record_length`.

### Common mistakes and how the code avoids them

- **Mixing units.** PSR needs the per-period Sharpe ratio and the number of periods. Everything inside the module is per-period; names containing `annual` are annualised.
- **Excess versus raw kurtosis.** The formula uses raw kurtosis (normal = 3). `return_moments` returns raw kurtosis and a test compares it with SciPy.
- **Reading PSR as "the probability the strategy is good".** It is a statement about sampling error for one pre-chosen strategy. It ignores selection (module 6).
- **Using PSR-based p-values at extreme significance levels.** Covered above.
- **Dividing by zero for strategies that never trade.** They get Sharpe 0 and p = 1.

### Interview questions

**Q1. What is the standard error of a Sharpe ratio?**
Approximately $`\sqrt{(1 + SR^2/2)/T}`$ for normal returns, and with skewness and kurtosis terms in general. For small Sharpe ratios it is about $`1/\sqrt{T}`$ per period, or $`1/\sqrt{\text{years}}`$ for the annualised value.

**Q2. Why do skewness and kurtosis enter?**
The Sharpe ratio divides the mean by an estimated standard deviation. The error in that estimate depends on the fourth moment, and its correlation with the error in the mean depends on the third. With negative skew, samples with a high mean tend to be those that missed the rare large losses, which also have a low measured volatility, so the ratio is doubly flattered.

**Q3. A fund shows an annualised Sharpe of 1.5 over three years of daily data. Convincing?**
The standard error is roughly 1/√3 ≈ 0.58, so the estimate is about 2.6 standard errors from zero: significant on its own. If it was the best of many tried, or returns are negatively skewed, much less so. I would ask how many alternatives were tested.

**Q4. What does PSR not account for?**
Selection among multiple trials, serial correlation in returns (the formula assumes independent observations), and non-stationarity.

**Q5. Why did you not use 1 - PSR(0) as your p-value?**
For testing SR = 0 the correct standard error is the one under the null, which has no skewness term. PSR evaluates it at the estimate with the in-sample skewness. For a strategy with a handful of active days, lucky returns raise both the mean and the sample skewness, the standard error shrinks, and the p-value becomes far too small. In null simulations Bonferroni on those p-values broke its guaranteed error bound.

### Self-quiz

1. Daily Sharpe 0.04. Annualised (365 days)?
2. Roughly how many years of data make an annualised Sharpe of 0.5 significant at the one-sided 5% level?
3. Does negative skewness raise or lower PSR for a positive Sharpe ratio?
4. What is PSR(SR*) when the estimate equals the benchmark?
5. Why do the skewness and kurtosis terms not matter under the null SR = 0?

<details><summary>Answers</summary>

1. 0.04 × √365 ≈ 0.76.
2. Need SR × √years ≥ 1.645, so years ≥ (1.645 / 0.5)² ≈ 10.8.
3. Lowers it: the term -γ₃·SR is positive, the standard error grows.
4. 0.5.
5. Both are multiplied by SR (or SR²), which is zero.

</details>

---

## 6. Expected maximum Sharpe, the Deflated Sharpe Ratio and effective trials

### What it does

PSR treats a strategy as if it were the only one ever tested. The Deflated Sharpe Ratio (DSR) corrects for the search: it asks whether the best Sharpe ratio is higher than the best you would expect from that many strategies with no skill at all. To use it you need the number of *independent* trials, which this module also estimates.

### The maths

**The maximum of N noisy zeros.** Let $`\widehat{SR}_1,\dots,\widehat{SR}_N`$ be independent, each normal with mean 0 and standard deviation $`\sqrt{V}`$ (no skill, only estimation error). Extreme value theory gives the expected maximum:

```math
E[\max_n \widehat{SR}_n] \approx \sqrt{V}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right] =: SR_0
```

- $`\gamma \approx 0.5772`$: the Euler-Mascheroni constant.
- $`\Phi^{-1}`$: the inverse normal CDF (quantile function).
- e: Euler's number.
- The bracket is the expected maximum of N standard normals. It grows like $`\sqrt{2\ln N}`$: about 1.57 for N = 10, 2.53 for 100, 3.26 for 1,000, 3.69 for 5,000.

**Example.** Eight years of daily data: the annualised Sharpe estimate has standard deviation about $`1/\sqrt{8} = 0.354`$. With 5,000 independent skill-less trials the best is expected to show 0.354 × 3.69 ≈ 1.30. With 600 independent trials, 0.354 × 3.11 ≈ 1.10. A backtest Sharpe of 1.2 is therefore unremarkable.

**DSR** (Bailey and Lopez de Prado 2014): the PSR with benchmark $`SR_0`$:

```math
DSR = \Phi\!\left(\frac{(\widehat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3\widehat{SR} + \frac{\gamma_4-1}{4}\widehat{SR}^2}}\right)
```

where $`\widehat{SR}`$, $`\gamma_3`$, $`\gamma_4`$ belong to the selected strategy, V is the variance of the Sharpe estimates across all trials, and N is the number of independent trials.

**The paper's worked example** (reproduced in `tests/test_sharpe.py`). Annualised SR 2.5 from 1,250 daily observations (250 per year), skewness -3, kurtosis 10, best of N = 100, variance of annualised Sharpe ratios across trials 1/2.
- Daily SR = 2.5/√250 = 0.1581. Daily $`\sqrt{V} = \sqrt{0.5/250} = 0.0447`$.
- $`SR_0 = 0.0447 × 2.531 = 0.1132`$.
- Numerator: (0.1581 - 0.1132) × √1249 = 1.587. Denominator: √(1 + 3 × 0.1581 + 2.25 × 0.025) = 1.237.
- DSR = Φ(1.283) = 0.9004. Below 0.95: not significant, despite an annualised Sharpe of 2.5.

**Effective number of trials.** Correlated strategies are not independent draws. Define $`N_{\text{eff}}`$ as the number of independent trials whose expected maximum equals that of the correlated family. Four estimators from the correlation matrix C:

| Name | Formula | Idea |
|---|---|---|
| Average correlation | $`\bar\rho + (1-\bar\rho)N`$ | interpolate between 1 (all identical) and N (all independent) |
| Eigenvalue ratio | $`(\sum\lambda_i)^2/\sum\lambda_i^2 = N^2/\lVert C\rVert_F^2`$ | how many directions carry the variance |
| Clustering | number of clusters at the best silhouette score | each cluster of similar strategies counts once |
| Max-matching | simulate $`z\sim\mathcal N(0,C)`$, average $`\max_k z_k`$, invert the bracket formula | apply the definition directly |

The max-matching simulation is cheap: if Z is the T x N matrix of standardised returns and g a vector of T standard normals, $`g^\top Z/\sqrt{T-1}`$ has covariance exactly C.

**Silhouette score.** For point i, a = mean distance to its own cluster, b = mean distance to the nearest other cluster, silhouette = (b - a)/max(a, b). The quality of a clustering is mean/standard deviation of the silhouettes, as in Lopez de Prado's ONC algorithm. Distance between strategies is $`\sqrt{(1-C_{ij})/2}`$.

### Where it lives

- `diagnostics/sharpe.py`: `expected_max_z` (the bracket), `expected_max_sharpe`, `deflated_sharpe_ratio`.
- `diagnostics/trials.py`: `neff_avg_corr`, `neff_eigen`, `neff_cluster`, `neff_max_matching`, `effective_trials`.
- `experiments/common.py: diagnose` computes the DSR under every estimator and marks one as primary.

### Common mistakes and how the code avoids them

- **Using the raw count for N.** Overstates the luck benchmark; the DSR becomes too strict.
- **Using a cluster count for N.** Understates it badly: variants inside a cluster are correlated, not identical, so the best of a cluster is still inflated. E1 measured this (module 12).
- **Annualised and per-period quantities in one formula.** $`\widehat{SR}`$, $`SR_0`$ and $`\sqrt V`$ must share units; the function takes per-period values only.
- **Forgetting that V comes from the trials.** V is the cross-sectional variance of the trial Sharpe ratios, not the sampling variance of one of them.
- **Hiding trials.** The DSR can only deflate for trials it is told about. Every variant is in the matrix, including the ones that never traded.

### Interview questions

**Q1. Explain the Deflated Sharpe Ratio in two sentences.**
The best of N backtests has an inflated Sharpe ratio even if none has skill, and the expected size of that inflation can be computed from N and the spread of the trial Sharpe ratios. The DSR is the probability that the selected strategy's true Sharpe ratio is positive after subtracting that expected inflation and allowing for sample length and non-normal returns.

**Q2. How does the expected maximum grow with the number of trials?**
Like $`\sqrt{2\ln N}`$ standard deviations: slowly. Going from 100 to 10,000 trials raises it from about 2.5 to about 3.9. Trying ten times more strategies costs a fixed increment, not a proportional one, but with a standard error of 0.35 each increment is a meaningful amount of Sharpe.

**Q3. Your strategies are correlated. How did you choose N?**
I implemented four estimators and checked them where the truth is known. In null simulations I compared each estimator's predicted best Sharpe with the average realised best. Raw count and average correlation overshot, clustering and eigenvalues undershot badly, and a simulation that directly matches the expected maximum of correlated normals was closest, so that one feeds the headline. The others are reported alongside.

**Q4. Why did clustering fail?**
The silhouette criterion finds the coarsest clear structure, here essentially trend versus mean reversion, and returns a handful of clusters. But the DSR needs the number of independent draws that reproduces the maximum, and hundreds of imperfectly correlated variants inside one cluster produce a much higher maximum than a single draw. Lopez de Prado's full procedure aggregates each cluster into one return series and selects among those; if you still select the best raw variant, the cluster count is the wrong N.

**Q5. What are the DSR's weaknesses?**
It assumes independent, normally distributed trial Sharpe ratios with a common variance, depends heavily on an N that must be estimated, and counts only reported trials. Its dispersion term mixes sampling noise with real differences between strategies. It is also conservative as a test: the selected Sharpe must exceed the expected maximum by a further 1.645 standard errors.

### Self-quiz

1. What is the expected maximum of 100 independent standard normals, to one decimal?
2. In the paper's example, why is $`\sqrt{V}`$ divided by √250?
3. If all N strategies were perfectly correlated, what should N_eff be?
4. Does a larger V make the DSR larger or smaller?
5. Why does g'Z/√(T-1) have covariance C?

<details><summary>Answers</summary>

1. About 2.5.
2. To convert the annualised dispersion to daily units, matching the daily Sharpe ratio.
3. One.
4. Smaller: SR₀ rises.
5. Cov = Z'E[gg']Z/(T-1) = Z'Z/(T-1), which is the sample correlation matrix of standardised columns.

</details>

---

## 7. Probability of Backtest Overfitting (CSCV)

### What it does

Asks a direct question about the selection process: if I pick the in-sample winner, how often does it turn out to be worse than the median strategy out of sample? It answers by replaying the selection over thousands of different in-sample/out-of-sample splits of the same data.

### The maths

Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado, Zhu 2017):

1. Cut the T rows into S contiguous blocks of equal size (S even).
2. Choose S/2 blocks as the in-sample set J; the other S/2 are the out-of-sample set. There are $`\binom{S}{S/2}`$ ways: 70 for S = 8, 12,870 for S = 16.
3. For each split: compute every strategy's Sharpe ratio on J; let n* be the best. Compute every strategy's Sharpe ratio out of sample and find the rank of n* among the N (1 = worst, N = best).
4. Relative rank $`\bar\omega = \text{rank}/(N+1) \in (0,1)`$ and logit $`\lambda = \ln\frac{\bar\omega}{1-\bar\omega}`$.
5. $`PBO = \frac{\#\{\lambda \le 0\}}{\#\text{splits}}`$.

- $`\lambda > 0`$: the in-sample winner is above the out-of-sample median. $`\lambda < 0`$: below.
- Example: N = 99, the winner ranks 25th from the bottom: $`\bar\omega = 0.25`$, $`\lambda = \ln(1/3) = -1.10`$.

**Why "symmetric".** Both sets always have the same size, so Sharpe ratios are comparable, and every split's complement is also a split.

**Reading PBO.** If selection is pure noise, the winner's out-of-sample rank is uniform and PBO ≈ 0.5. If one strategy is truly best, it wins in and out of sample and PBO ≈ 0. PBO above 0.5 means selection is actively harmful (what looks best tends to do worst).

**Companion outputs.** The slope of out-of-sample on in-sample Sharpe of the winners across splits ("performance degradation") and the fraction of splits where the winner's out-of-sample Sharpe is negative ("probability of loss").

**Fast computation.** A Sharpe ratio over a union of blocks only needs each block's sum and sum of squares. With those two S x N tables, the in-sample statistics of every split are a matrix product of a 0/1 selection matrix with the tables, so 12,870 splits of 5,000 strategies take about a second.

### Where it lives

`diagnostics/pbo.py`: `cscv` returns a `PBOResult` (pbo, logits, the winners' IS and OOS Sharpe ratios). `_sharpe_from_sums` does the block arithmetic.

### Common mistakes and how the code avoids them

- **Shuffling rows.** Blocks are contiguous and keep time order, so serial dependence inside a block is preserved.
- **Reading PBO as the probability of losing money.** It is about rank relative to the other strategies. The probability of loss is reported separately.
- **Ignoring ties.** Strategies with identical out-of-sample Sharpe (for example several that did not trade) get the average rank.
- **Trusting one implementation.** `tests/test_pbo.py` compares the fast version with a slow literal implementation and checks three known cases: a truly superior strategy (PBO = 0), pure noise (≈ 0.5), and anti-persistent performance (PBO = 1).
- **Forgetting that the splits are not independent.** They reuse the same blocks, so PBO has more sampling error than 12,870 splits suggests.

### Interview questions

**Q1. What is PBO, precisely?**
The probability that the strategy chosen as best in sample has an out-of-sample performance below the median of all candidates, estimated by repeating the selection over all balanced partitions of the data into in-sample and out-of-sample blocks.

**Q2. What value would you expect for strategies with no skill, and why?**
About 0.5. With no persistent differences, the in-sample winner's out-of-sample rank is uniformly distributed, so it is below the median half the time.

**Q3. Your PBO on real data is below 0.5. Does that prove the strategies work?**
No. It shows that rankings persist to some degree, which can come from a shared exposure rather than skill. In this project long-only trend rules on rising assets stay near the top in most splits because they all hold the asset. PBO says nothing about whether the level of performance is significant; SPA and DSR address that.

**Q4. How does the number of blocks S matter?**
More blocks give more splits and shorter blocks. Very short blocks make each block's statistics noisy and break up long holding periods; very few blocks give too few splits. The paper uses 16. The robustness experiment varies S from 6 to 16.

**Q5. How is CSCV different from k-fold cross-validation?**
k-fold trains on k-1 folds and tests on one, so the training set is much larger than the test set and there are only k splits. CSCV uses equal halves, giving comparable Sharpe ratios and a combinatorial number of splits, and it evaluates a selection among many strategies rather than the fit of one model.

### Self-quiz

1. How many splits does CSCV produce for S = 6?
2. The in-sample winner is the very best out of sample among N = 199. What is its logit?
3. What PBO do you get if the in-sample winner is always the out-of-sample worst?
4. Why must S be even?
5. Why can block sums and sums of squares replace the raw data?

<details><summary>Answers</summary>

1. C(6,3) = 20.
2. ω = 199/200 = 0.995, λ = ln(199) ≈ 5.29.
3. 1.
4. So the data can be split into two halves with the same number of blocks.
5. Mean and variance over any union of blocks are functions of the total sum, total sum of squares and count.

</details>

---

## 8. Purged k-fold and CPCV

### What it does

Cross-validation estimates out-of-sample performance by holding part of the data back. On time series, neighbouring observations share information, so the held-back part is not really unseen. Purging and embargoing remove the contaminated observations. Combinatorial Purged CV (CPCV) then produces many out-of-sample backtest paths instead of one.

### The maths

**Label windows.** Observation i carries information over the interval $`[i,\ \text{end}_i]`$, with $`\text{end}_i = i + h`$ for a horizon h. In machine learning h is how far ahead the label looks. Here h models how long positions and indicators link neighbouring days.

**Purging.** For a contiguous test block [a, b], let $`e = \max_{j\in[a,b]} \text{end}_j`$. Training observation i is removed if its window overlaps the test information, that is if $`i \le e`$ and $`\text{end}_i \ge a`$.

**Embargo.** Also remove training observations with $`e < i \le e + \text{embargo}`$. Serial dependence leaks forward past the label window, so a buffer is left after the test block.

**Example** (`tests/test_cv.py`). 100 observations, test block [40, 59], h = 5, embargo 3. Observations 35 to 39 are purged (their windows reach day 40). Test labels reach day 64, so 60 to 64 are purged and 65 to 67 embargoed. Training set: 0 to 34 and 68 to 99.

**CPCV.** Split the data into N contiguous groups; use every combination of k groups as the test set.

```math
\#\text{splits} = \binom{N}{k}, \qquad \#\text{paths} = \varphi = \frac{k}{N}\binom{N}{k} = \binom{N-1}{k-1}
```

Each group is a test group in $`\binom{N-1}{k-1}`$ splits. Taking, for each group, its first appearance, then its second, and so on, assembles $`\varphi`$ complete paths, each covering the whole sample with out-of-sample predictions. With N = 6, k = 2: 15 splits, 5 paths. With N = 10, k = 2: 45 splits, 9 paths.

**What is being cross-validated here.** The model is the rule "trade the variant with the best training-set Sharpe ratio". On each split the rule is fitted on the training rows and its pick is evaluated on the test groups. The paths' Sharpe ratios form a distribution of out-of-sample outcomes for that rule.

### Where it lives

`diagnostics/cv.py`: `label_ends`, `purged_train_indices`, `purged_kfold`, `cpcv_splits`, `cpcv_path_assignments`, `cpcv_select_best`.

### Common mistakes and how the code avoids them

- **Ordinary k-fold on time series.** Training on data immediately before and after the test block leaks.
- **Purging only one side.** Both the observations before the test block (whose windows reach into it) and those after it (inside the test labels' reach) are removed.
- **Forgetting the embargo with several test blocks.** In CPCV the test groups may be non-adjacent; purging is applied around each contiguous block separately.
- **Miscounting paths.** A test checks that every (split, test group) pair is used exactly once.
- **Asserting no leakage without checking.** `_assert_no_leakage` verifies by brute force, for every training and test observation, that windows do not overlap.

### Interview questions

**Q1. Why does standard k-fold cross-validation fail in finance?**
Observations are not independent. Labels overlap in time and features are serially correlated, so a training point next to a test point contains almost the same information. The model is effectively tested on data it has seen and the score is too high.

**Q2. What is the difference between purging and an embargo?**
Purging removes training observations whose information window overlaps the test set's. The embargo removes an extra stretch of training observations after each test block, because dependence persists beyond the formal label window.

**Q3. What does CPCV give you that walk-forward does not?**
Walk-forward gives one out-of-sample path and never tests the earliest data. CPCV tests every part of the sample and recombines the results into several full-length paths, so you get a distribution of out-of-sample Sharpe ratios instead of a single number. The cost is that training sets include data from after the test period, which is not how trading works.

**Q4. How many paths does CPCV with 8 groups and 2 test groups produce?**
C(7,1) = 7 paths from C(8,2) = 28 splits.

**Q5. In your project nothing is "trained". What does purging mean?**
The trained object is the choice of variant. A day's return on a variant depends on a position built from earlier days, and one trade's profit spans several days, so days on either side of a boundary share information. The purge horizon and embargo are a model of that dependence and are choices, which is why the robustness experiment varies them.

### Self-quiz

1. 200 observations, test block [100, 119], h = 10, embargo 5. Which training observations remain?
2. How many splits and paths for N = 12, k = 2?
3. Why is the embargo placed after the test block rather than before?
4. With h = 0 and embargo 0, what does purged k-fold reduce to?
5. What fraction of the data is in the test set for N = 10, k = 2?

<details><summary>Answers</summary>

1. 0 to 89 and 135 to 199 (90 to 99 purged before; test labels reach 129, so 120 to 129 purged, 130 to 134 embargoed).
2. 66 splits, 11 paths.
3. Before the block, purging already handles windows that reach into it. After the block, information from the test period keeps influencing later observations.
4. Ordinary k-fold with contiguous folds.
5. 20%.

</details>

---

## 9. Reality Check and SPA

### What it does

Two tests of one family-wide question: "is the best strategy in this whole collection really better than doing nothing?" They build the distribution of "the best of N" under the null hypothesis by resampling the actual data, so the correlation between strategies is handled automatically.

### The maths

Let $`d_{k,t}`$ be the return of strategy k on day t (its excess over cash) and $`\bar d_k`$ its mean. The null is $`H_0: \max_k E[d_k] \le 0`$.

**Stationary bootstrap** (Politis and Romano 1994). Build a resampled series of length T by: start at a random day; with probability 1 - q continue to the next day (wrapping around the end), with probability q jump to a new random day. Block lengths are geometric with mean 1/q (10 days here). Whole rows are resampled, so all strategies move together as they did in reality, and blocks preserve short-term time dependence such as volatility clustering.

**White's Reality Check (2000).**

```math
V = \max_k \sqrt{T}\,\bar d_k, \qquad V^*_b = \max_k \sqrt{T}\,(\bar d^*_{k,b} - \bar d_k), \qquad p = \frac{\#\{V^*_b > V\}}{B}
```

- $`\bar d^*_{k,b}`$: mean of strategy k in bootstrap resample b.
- Subtracting $`\bar d_k`$ re-centres every strategy at zero, imposing the null in which all means are zero.
- p is the fraction of resamples in which luck alone produced a better "best".

**Hansen's SPA (2005).** Two changes.

1. *Studentise*: divide by a scale $`s_k`$ so that every strategy is measured in its own standard errors. Without this the maximum is dominated by the most volatile strategies.
2. *Re-centre selectively*: a strategy that is significantly bad should not be moved up to zero, because that adds a contender that could never win and inflates the null distribution.

```math
T^{SPA} = \max\Bigl(\max_k \frac{\sqrt T\,\bar d_k}{s_k},\ 0\Bigr), \qquad T^*_b = \max\Bigl(\max_k \frac{\sqrt T\,(\bar d^*_{k,b} - g(\bar d_k))}{s_k},\ 0\Bigr)
```

```math
g_c(\bar d_k) = \bar d_k \cdot 1\Bigl\{\bar d_k \ge -s_k\sqrt{\tfrac{2\ln\ln T}{T}}\Bigr\}
```

- $`g_c`$ ("consistent"): re-centre strategy k only if it is not clearly below zero. The threshold comes from the law of the iterated logarithm.
- Lower bound $`g_l = \max(\bar d_k, 0)`$ re-centres only winners (most liberal); upper bound $`g_u = \bar d_k`$ re-centres everything (most conservative, a studentised Reality Check).

**The scale.** Hansen uses the long-run standard deviation $`\omega_k`$:

```math
\omega_k^2 = \hat\gamma_0 + 2\sum_{i\ge 1}\kappa(i)\,\hat\gamma_i, \qquad \kappa(i) = \tfrac{T-i}{T}(1-q)^i + \tfrac{i}{T}(1-q)^{T-i}
```

with $`\hat\gamma_i`$ the lag-i autocovariance. This is exactly the variance the stationary bootstrap assigns to $`\sqrt T\,\bar d`$. This project uses the plain sample standard deviation by default and reports the $`\omega`$ version alongside, because E1 found the $`\omega`$ version rejects too often (module 12).

**Fast computation.** Both tests need only the mean of each resampled column. If $`c_{b,t}`$ counts how often day t appears in resample b, then all means are the matrix product $`c\,X/T`$: one multiplication for the whole bootstrap.

### Where it lives

`diagnostics/bootstrap.py`: `stationary_bootstrap_indices`, `bootstrap_counts`, `bootstrap_means`, `stationary_long_run_variance`, `reality_check_and_spa`.

### Common mistakes and how the code avoids them

- **Resampling each strategy separately.** Destroys the correlation between strategies, which is the whole point. Rows are resampled.
- **Resampling single days.** Destroys time dependence. Blocks are used.
- **Forgetting to re-centre.** Without it the bootstrap describes the data, not the null.
- **Letting junk strategies hide a good one** (Reality Check's weakness). A test builds one modest winner plus sixty terrible volatile strategies and shows SPA finds the winner while the Reality Check does not.
- **A noisy scale under a maximum.** See module 12; a test (`test_long_run_scale_is_noisier_and_inflates_the_maximum`) reproduces the effect.

### Interview questions

**Q1. What is data snooping and how does the Reality Check address it?**
Data snooping is reusing the same data to evaluate many models and reporting the best. The Reality Check tests the best model against a benchmark using the bootstrap distribution of the maximum over all models, so the p-value already accounts for the search.

**Q2. Why bootstrap instead of Bonferroni?**
Bonferroni treats tests as if they could be arbitrarily dependent and is very conservative when strategies are highly correlated. The bootstrap resamples whole days, so the joint distribution of all strategies is preserved and the null distribution of the maximum reflects the true dependence.

**Q3. How does SPA improve on the Reality Check?**
It studentises, which stops volatile strategies dominating, and it avoids re-centring clearly inferior strategies, which stops irrelevant alternatives from inflating the null distribution. Adding many poor strategies to the universe can drive the Reality Check's p-value towards one; SPA is far less affected.

**Q4. What does the block length do, and how did you choose it?**
It sets how much time dependence the resamples keep. Too short treats returns as independent; too long leaves too few distinct blocks. Daily strategy returns have little autocorrelation but clustered volatility, so a mean of 10 days is used and the robustness experiment varies it from 5 to 40.

**Q5. These tests say whether the best strategy beats zero. What do they not say?**
They do not identify which strategies are good (that needs a step-down extension such as Romano and Wolf's), they test mean return rather than risk-adjusted return unless studentised, and against a cash benchmark a long-only strategy on a rising asset can pass through market exposure alone.

### Self-quiz

1. Mean block length 20: what is q?
2. Why is $`\bar d_k`$ subtracted inside the bootstrap statistic?
3. In SPA, what happens to a strategy with a strongly negative mean?
4. Which of the three SPA p-values is the largest?
5. Why can the entire bootstrap be done with one matrix product?

<details><summary>Answers</summary>

1. 0.05.
2. To centre the bootstrap distribution at the null (zero mean); otherwise resampled means would be centred on the observed means.
3. It is not re-centred, so it stays negative in every resample and cannot contribute to the maximum.
4. The upper one (everything re-centred, most contenders).
5. The statistic depends on each resample only through column means, and a resampled mean is a weighted sum of the original rows with weights equal to the resampling counts.

</details>

---

## 10. Multiple-testing corrections and the haircut Sharpe ratio

### What it does

Each variant gets a p-value. With 5,000 of them, 250 are expected below 0.05 by chance. Corrections adjust the p-values so that a stated error rate holds for the whole family. The haircut Sharpe ratio expresses the same correction as a reduced Sharpe ratio.

### The maths

Let $`p_{(1)} \le \dots \le p_{(N)}`$ be the sorted p-values.

**Two error rates.**
- FWER (family-wise error rate): probability of *at least one* false discovery.
- FDR (false discovery rate): expected *fraction* of discoveries that are false.

**Bonferroni** (FWER): $`p^{adj}_i = \min(1, N p_i)`$. Reject if $`p_i < \alpha/N`$. Valid under any dependence, by the union bound: $`P(\text{any false rejection}) \le \sum_i P(p_i < \alpha/N) \le \alpha`$.

**Holm** (FWER): $`p^{adj}_{(i)} = \max_{j\le i}\min(1, (N-j+1)\,p_{(j)})`$. Step down from the smallest p-value; once one hypothesis is rejected, the next is compared with a slightly easier threshold. Never rejects less than Bonferroni and is also valid under any dependence.

**Benjamini-Hochberg** (FDR): $`p^{adj}_{(i)} = \min_{j\ge i}\min(1, N p_{(j)}/j)`$. Equivalently, find the largest i with $`p_{(i)} \le \frac{i}{N}\alpha`$ and reject the i smallest. Valid under independence and positive dependence.

**Benjamini-Yekutieli** (FDR, any dependence): BH with α replaced by $`\alpha / c(N)`$, $`c(N) = \sum_{i=1}^N 1/i \approx \ln N + 0.577`$.

**Worked example** (`tests/test_multiple.py`), p = 0.001, 0.008, 0.039, 0.041, 0.60 (N = 5):

| p | Bonferroni | Holm | BH | BY |
|---|---|---|---|---|
| 0.001 | 0.005 | 0.005 | 0.005 | 0.0114 |
| 0.008 | 0.040 | 0.032 | 0.020 | 0.0457 |
| 0.039 | 0.195 | 0.117 | 0.051 | 0.117 |
| 0.041 | 0.205 | 0.117 | 0.051 | 0.117 |
| 0.60 | 1 | 0.60 | 0.60 | 1 |

At α = 0.05: four are naively significant, two after any correction.

**Scale of the problem.** For Bonferroni with N = 5,000 a variant needs p < 0.00001, a t-statistic of about 4.27. With eight years of data that is an annualised Sharpe of about 1.5. The naive 5% cut-off is a t of 1.645, an annualised Sharpe of about 0.58.

**Haircut Sharpe ratio** (Harvey and Liu 2015).
1. t-statistic: $`t = SR_{\text{annual}}\sqrt{\text{years}}`$. Two-sided p-value: $`p = 2(1-\Phi(|t|))`$.
2. Adjust p for multiple tests (Bonferroni, Holm, or BHY).
3. Convert back: $`t^{adj} = \Phi^{-1}(1 - p^{adj}/2)`$, $`SR^{adj} = t^{adj}/\sqrt{\text{years}}`$.
4. Haircut = $`1 - SR^{adj}/SR`$.

Example: annualised Sharpe 1.0 over 9 years, t = 3, p = 0.0027. With 10 tests under Bonferroni p = 0.027, t = 2.21, haircut Sharpe 0.74: a 26% haircut. The haircut is non-linear: marginal Sharpe ratios are cut almost to zero while very high ones are barely touched.

### Where it lives

`diagnostics/multiple.py`: `bonferroni`, `holm`, `benjamini_hochberg`, `benjamini_yekutieli`, `survivors`, `haircut_sharpe`. P-values come from `diagnostics/sharpe.py: sharpe_pvalue`.

### Common mistakes and how the code avoids them

- **Applying a correction to invalid p-values.** A correction cannot repair p-values that are wrong in the tail. E1 found exactly this for PSR-based p-values and the project switched to the t-test.
- **Confusing FWER and FDR.** "BH survivors" may contain a controlled fraction of false positives; "Holm survivors" probably contain none.
- **One-sided versus two-sided.** The corrections use one-sided p-values (only positive Sharpe is of interest). The haircut follows the paper's two-sided convention; the code says so.
- **Home-made implementations.** All four adjustments are tested against `statsmodels`.
- **Using BH under negative dependence without saying so.** Trend and reversal variants are negatively correlated, which is outside BH's guarantee; BY is reported as the dependence-proof alternative.

### Interview questions

**Q1. What is the difference between controlling FWER and FDR?**
FWER control bounds the probability of any false discovery, which suits a decision to trade one strategy. FDR control bounds the expected share of false discoveries among those reported, which suits screening a list for further research. FDR methods are more powerful and admit some false positives by design.

**Q2. Why is Holm preferred to Bonferroni?**
It gives the same guarantee under the same assumptions and rejects at least as many hypotheses, because after rejecting the smallest p-value it only needs to correct for the remaining ones.

**Q3. On your data BH keeps a few hundred variants and Holm keeps none. Which is right?**
Both, for different questions. No single variant has evidence strong enough to be trusted on its own. But the distribution of p-values is not what pure noise would give, so a list of the best contains a controlled share of false positives. The reason is a broad shared effect: long-only trend rules on assets that rose.

**Q4. What does the haircut Sharpe ratio add?**
It converts a statistical correction into the unit practitioners think in. Instead of "the adjusted p-value is 0.3", it says "after accounting for the search this 1.4 is worth about 0.3".

**Q5. Do these corrections assume independent tests?**
Bonferroni and Holm do not. BH requires independence or positive dependence; BY does not. None of them exploits the dependence to gain power, which is what the bootstrap tests do.

### Self-quiz

1. N = 1,000, α = 0.05: the Bonferroni threshold?
2. Sorted p-values 0.01, 0.02, 0.03, 0.5 (N = 4). How many does BH reject at 0.05?
3. Can Holm reject fewer hypotheses than Bonferroni?
4. Why is the haircut on a Sharpe of 3 much smaller, in percentage terms, than on a Sharpe of 1?
5. What is c(N) for N = 5?

<details><summary>Answers</summary>

1. 0.00005.
2. Thresholds are 0.0125, 0.025, 0.0375, 0.05. The largest i with p ≤ threshold is 3, so three.
3. No.
4. Its p-value is so small that multiplying it by N still leaves a large t-statistic; the map from p to t is very flat in the far tail.
5. 1 + 1/2 + 1/3 + 1/4 + 1/5 = 2.283.

</details>

---

## 11. Minimum Backtest Length

### What it does

Turns the expected-maximum formula around. Given how many strategies you tried, how long must the backtest be before a given Sharpe ratio stops being what pure luck would deliver?

### The maths

With y years of data, an annualised Sharpe estimate of a skill-less strategy has standard deviation about $`1/\sqrt{y}`$. The best of N independent ones is expected to be

```math
E[\max] \approx \frac{1}{\sqrt{y}}\Bigl[(1-\gamma)\Phi^{-1}(1-\tfrac1N) + \gamma\,\Phi^{-1}(1-\tfrac{1}{Ne})\Bigr]
```

Setting this equal to a target $`\overline{SR}`$ and solving for y:

```math
MinBTL \approx \left(\frac{(1-\gamma)\Phi^{-1}(1-\frac1N) + \gamma\,\Phi^{-1}(1-\frac{1}{Ne})}{\overline{SR}}\right)^2 < \frac{2\ln N}{\overline{SR}^2}
```

- N: number of independent trials.
- $`\overline{SR}`$: the annualised in-sample Sharpe ratio in question.
- If the backtest is shorter than MinBTL, a Sharpe of $`\overline{SR}`$ is the *expected* outcome of luck.

**Paper example** (tested): N = 45, target Sharpe 1 gives 5.0 years. "With five years of data, try no more than 45 independent configurations."

**More values for a target of 1**: N = 100 → 6.4 years, N = 1,000 → 10.6 years, N = 5,000 → 13.6 years. Because the target enters squared, a target of 2 needs a quarter of the time.

**Inverse**: with 8 years and a target Sharpe of 1, about 243 independent trials are allowed.

### Where it lives

`diagnostics/sharpe.py`: `min_backtest_length`, `max_trials_for_length`. Used in `experiments/common.py: diagnose` with both the raw and the effective number of trials; plotted in `reports/figures/e3_cpcv_minbtl.png`.

### Common mistakes and how the code avoids them

- **Reading MinBTL as "enough data for significance".** It only marks where the Sharpe equals the expected luck. Significance needs a margin on top (that is the DSR).
- **Using the raw number of correlated trials.** Both versions are reported.
- **Unit mix-ups.** The function takes an annualised Sharpe and returns years.
- **Ignoring the square.** Halving the target Sharpe quadruples the required length.

### Interview questions

**Q1. What is the Minimum Backtest Length?**
The backtest length below which the best of N skill-less strategies is expected to reach a given in-sample Sharpe ratio. It links three quantities: trials, data, and the Sharpe ratio you can believe.

**Q2. Derive the 2 ln N bound.**
The expected maximum of N standard normals is below $`\sqrt{2\ln N}`$. The annualised Sharpe of a skill-less strategy over y years is roughly normal with standard deviation $`1/\sqrt y`$, so the expected best is below $`\sqrt{2\ln N / y}`$. Requiring that to be below the target gives $`y > 2\ln N/\overline{SR}^2`$.

**Q3. You have 8 years and tried 5,000 variants. What does MinBTL tell you?**
For 5,000 independent trials a Sharpe of 1 would need over 13 years, so a Sharpe of 1 means nothing. With the effective number of trials (a few hundred) the required length is shorter, but the best observed Sharpe is still close to the luck level for the sample length.

**Q4. How should this change a research process?**
Count trials, and decide the search budget before looking: with a fixed amount of data, each extra configuration raises the Sharpe ratio needed to be credible. Prefer few, well-motivated hypotheses to wide grids.

**Q5. What does MinBTL assume?**
Independent trials, normally distributed Sharpe estimates with standard deviation $`1/\sqrt y`$, and zero true Sharpe for every trial. Fat tails and correlated trials change the numbers but not the message.

### Self-quiz

1. Target Sharpe 1.5, N = 45. MinBTL in years?
2. Does doubling N double MinBTL?
3. Which is larger, MinBTL or the bound 2 ln N / SR²?
4. With 2 years of data and a target of 1, roughly how many independent trials are allowed? (Use E[max] ≈ √2 ≈ 1.41.)
5. Why is MinBTL a necessary but not sufficient condition?

<details><summary>Answers</summary>

1. 5.0 / 1.5² ≈ 2.2 years.
2. No. It grows roughly like ln N.
3. The bound.
4. The bracket must be below √2 ≈ 1.41, which corresponds to about 7 trials.
5. Meeting it only means the Sharpe is not below what luck is expected to give; it could still be within luck's normal range.

</details>

---

## 12. E1: the null world

### What it does

Runs every diagnostic on synthetic markets that contain no signal. Since every strategy is worthless by construction, each "discovery" is a false positive, so the experiment measures the error rate of each tool. It is the calibration step: a tool that fails here cannot be trusted on real data.

Two parts: one full-size world with 5,000 variants (to show what a researcher would see), and a Monte Carlo over hundreds of smaller worlds (to measure rates).

### The maths

**Size of a test.** A test with nominal level α should reject a true null with probability at most α. For a family of tests the analogue is the family-wise error rate: the probability that at least one of the N nulls is rejected. In the Monte Carlo, each world is one Bernoulli trial ("did the method make any discovery?"), so the rate is a binomial proportion.

**Wilson interval** for k rejections in n worlds, with $`\hat p = k/n`$ and z = 1.96:

```math
\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
```

Unlike the textbook $`\hat p \pm z\sqrt{\hat p(1-\hat p)/n}`$ it behaves when k is 0 or small. With n = 200 and a true rate of 5%, the standard error is about 1.5 percentage points, so observed rates between roughly 2% and 8% are all compatible with a correctly sized test.

**What should happen.**
- Naive testing: about 5% of variants significant in every world, so almost every world contains "discoveries".
- Bonferroni and Holm: any discovery in at most 5% of worlds (less, because the bound is not tight under dependence).
- Benjamini-Hochberg: under the global null every discovery is false, so FDR equals the probability of any discovery, again at most about 5%.
- Reality Check and SPA: reject in about 5% of worlds. On gross returns all true means are exactly zero, the "least favourable" case where these tests should hit their level.
- DSR above 0.95: rare, because the best Sharpe must beat its own expected value by 1.645 standard errors.
- PBO: about 0.5 on average.

**What actually happened** (numbers in REPORT section 5.1). Three things failed first time and were fixed or demoted:

1. *PSR-based p-values with Bonferroni exceeded 5%.* Bonferroni cannot do that with valid p-values, so the p-values were wrong. The offenders were variants with a handful of active days. For such a variant, with returns $`x_1,\dots,x_m`$ on its m active days and zeros elsewhere, a lucky sample has a high mean and a high sample skewness at once. PSR's variance term $`1 - \hat\gamma_3\widehat{SR} + \dots`$ then becomes small and the z-score explodes. Under the null the correct variance has no skewness term, and the t-statistic of m points cannot exceed $`\sqrt m`$, so the t-test is immune.
2. *SPA studentised by the long-run standard deviation rejected more often than with the sample standard deviation.* The long-run estimate $`\omega_k`$ is far noisier; the statistic is a maximum over thousands of $`\bar d_k/\omega_k`$, and a maximum finds the variants whose $`\omega_k`$ is accidentally small. The bootstrap keeps $`\omega_k`$ fixed, so its null distribution lacks that source of large values.
3. *Cluster counts as the number of trials made the DSR pass far too often.* The silhouette criterion returned a handful of clusters; plugged into the expected-maximum formula that gives a luck benchmark far below the realised best Sharpe.

### Where it lives

`experiments/e1_null.py`: `world_statistics` (one world), `run_worlds` (parallel map), `summarise_worlds` (rates and intervals), `add_tables`, `run`. Wilson interval: `experiments/common.py: wilson_interval`. Regression tests for findings 1 and 2: `tests/test_sharpe.py`, `tests/test_bootstrap.py`.

### Common mistakes and how the code avoids them

- **Validating on net returns only.** With costs every true Sharpe is negative and any test looks safe. The Monte Carlo uses gross returns, where the null sits exactly on the boundary.
- **One world.** A single null world tells you almost nothing about a 5% rate. Hundreds are used and intervals reported.
- **Only a Gaussian null.** Both GBM and fat-tailed GARCH are run; the two failures above showed up differently in each.
- **Reusing random numbers across worlds.** Each world has its own seed; worker processes are started fresh.
- **Explaining a failure with a plausible story.** The first explanation for the SPA result (that mean-reversion rules get a low long-run variance when they are lucky) was tested on a toy example and was wrong. The explanation that survived (noise in the scale under a maximum) is the one with a test.

### Interview questions

**Q1. How did you validate your statistical tools?**
By running them on simulated markets with no predictability, where every strategy's true Sharpe is zero, and measuring how often each method reports a discovery across several hundred independent worlds. A correct method should do so at most about 5% of the time. I also ran a power study with a planted signal to check the tools are not simply always silent.

**Q2. What did that validation find?**
Three problems. Bonferroni applied to PSR-based p-values broke its error bound because in-sample skewness fakes significance for strategies that almost never trade. Hansen's SPA with the long-run standard deviation over-rejected because a noisy scale under a maximum is biased. And a cluster count used as the number of trials made the Deflated Sharpe Ratio pass in most fat-tailed null worlds. I switched to null-variance p-values, the sample standard deviation, and a simulation-based trial count, and kept the originals in the output for comparison.

**Q3. Bonferroni is supposed to be conservative. How can it over-reject?**
Only if the p-values it is fed are not valid, meaning they are smaller than they should be under the null. The procedure was fine; the input was wrong in the extreme tail, which is exactly where Bonferroni operates when N is large.

**Q4. Why is the null tested on gross rather than net returns?**
The null hypothesis is "true Sharpe ≤ 0". Net of costs, every strategy is strictly inside the null, so rejections are even rarer and the check is too easy. Gross returns put every strategy exactly at zero, the boundary case where the error rate is highest.

**Q5. Your SPA rejection rate in one null type was near 5% and the paper's version was higher. How sure are you of the difference?**
Moderately. With 200 worlds a rate has a standard error of one to two percentage points, and the intervals are in the report. The two versions are computed on the same worlds and the same bootstrap draws, so their difference is less noisy than either rate, and the mechanism is reproduced by a unit test. But I would not quote the exact excess.

### Self-quiz

1. In a null world with 5,000 variants, roughly how many are significant at 5% if tested individually?
2. Why does BH control the probability of any discovery under the global null?
3. 200 worlds, 10 rejections. Is that compatible with a 5% test?
4. Why can the t-statistic of a strategy with 6 active days never exceed about 2.45?
5. Why would a DSR built on N = 4 be dangerous?

<details><summary>Answers</summary>

1. About 250.
2. Every discovery is false, so the false discovery proportion is 1 whenever anything is rejected; its expectation (the FDR) equals the probability of rejecting anything.
3. Yes: exactly 5%.
4. With m non-zero observations the self-normalised sum is at most √m; √6 ≈ 2.45.
5. The luck benchmark would be tiny, so almost any best-of-thousands Sharpe would look significant.

</details>

---

## 13. E2: power

### What it does

The mirror image of E1. A real but hidden signal is planted at increasing strengths, and for each strength the experiment records how often each method notices. A tool that never raises an alarm passes E1 trivially; E2 checks the tools can actually detect something.

### The maths

**Power** is the probability of rejecting the null when it is false. It depends on the effect size, the sample length and the strictness of the test. For a single strategy with true annualised Sharpe SR tested over y years with critical value c (in standard errors):

```math
\text{power} \approx \Phi\bigl(SR\sqrt{y} - c\bigr)
```

- For one test at 5%, c = 1.645. For Bonferroni with 1,000 tests, c ≈ 3.9.
- Example: SR = 1, y = 8: $`SR\sqrt y = 2.83`$. Single test: Φ(1.18) = 88%. Bonferroni over 1,000: Φ(-1.07) = 14%. The same strategy that is obvious alone is invisible inside a large search.

**The planted signal.** A hidden AR(1) drift (module 1). Strength is the oracle's annualised Sharpe. The strategies see only prices, so the table in the report also gives an estimate of the true Sharpe of the best price-based variant: the average, over all simulated worlds and assets, of each variant's realised Sharpe, maximised over variants. At zero strength that estimate is slightly positive, which shows its own small selection bias.

**Detection** means: Holm or BH reject at least one variant; the Reality Check or SPA p-value is below 5%; the DSR is above 0.95.

**Thresholds.** For each method, the strength at which the detection rate crosses 50% and 80%, by linear interpolation between tested strengths.

**A result worth understanding.** The DSR as defined in the paper barely responds to the signal. Its luck benchmark is $`SR_0 = \sqrt{V}\times(\text{bracket})`$ with V the variance of Sharpe ratios across variants. A strong signal makes trend rules good and reversal rules bad, which raises V, which raises the benchmark as fast as the best Sharpe rises. Using the theoretical sampling variance $`1/(T-1)`$ for V removes that feedback, and that variant is reported alongside.

### Where it lives

`experiments/e2_power.py`: `run` loops over strengths, reuses `world_statistics` from E1 with `signal_strength` set, and `first_crossing` interpolates thresholds. Figure: `reports/figures/e2_power.png`.

### Common mistakes and how the code avoids them

- **Reporting only size.** A method that never rejects has perfect size and zero use.
- **Measuring power on gross returns and trading on net.** E2 uses net returns.
- **Quoting signal strength in units nobody can interpret.** Strength is an oracle Sharpe, and the best achievable rule's Sharpe is reported next to it.
- **Too few worlds per level.** With a few dozen worlds per level the curves are coarse (steps of a few percentage points); the report says how many.
- **Reading "not detected" as "not there".** At low strengths the signal is real and undetectable with this much data.

### Interview questions

**Q1. What is statistical power and why does it matter here?**
The probability that a test detects a real effect. Corrections for multiple testing protect against false positives by raising the bar, which lowers power. Knowing how large a true edge must be before it can be detected tells you what a negative result means.

**Q2. What true Sharpe ratio could your pipeline detect?**
The report gives the thresholds. In round terms, with about eight years of daily data and a thousand correlated variants, the best rule needs a true net Sharpe somewhat below 1 to be detected half the time by SPA, Holm or BH. Much smaller edges are invisible.

**Q3. Which methods were most powerful?**
SPA, the Reality Check, Holm and BH had similar thresholds in this design, because when the signal is real many variants pick it up at once. The DSR as defined in the paper was by far the least powerful, for the reason above.

**Q4. Why did the paper's DSR fail to detect even a strong signal?**
Its dispersion term is the cross-sectional variance of Sharpe ratios. Under the null that variance is sampling noise. With a real signal it also contains genuine differences between good and bad variants, so the benchmark it sets grows with the signal. In effect it treats real dispersion as evidence of more luck.

**Q5. How would you increase power without increasing false positives?**
Test fewer, better-motivated hypotheses; use more data or more independent assets; combine correlated variants into one composite before testing; and use tests that exploit dependence (bootstrap-based step-down procedures) instead of worst-case corrections.

### Self-quiz

1. Single test at 5%, true annualised Sharpe 0.5, 8 years. Approximate power?
2. Why is detection at strength 0 in E2 lower than the rates in E1?
3. What happens to PBO as the signal gets stronger?
4. Why is the "true Sharpe of the best variant" estimate slightly positive at zero strength?
5. Name one way to make Holm more powerful in this setting.

<details><summary>Answers</summary>

1. 0.5 × √8 = 1.41; Φ(1.41 - 1.645) = Φ(-0.23) ≈ 41%.
2. E2 uses net returns, so at zero signal every true Sharpe is negative; E1 uses gross returns.
3. It falls towards zero: the in-sample winner is a genuinely good variant and stays near the top.
4. It is a maximum over many noisy averages, so it has its own selection bias.
5. Reduce the number of hypotheses (for example test one composite per family).

</details>

---

## 14. E3: real data

### What it does

The full pipeline on real crypto prices: load and validate, build the 5,000-variant matrix net of costs, run every diagnostic, and report how many variants survive each correction, the PBO, and the DSR of the best. It then repeats the key tests on returns in excess of buy-and-hold.

### The maths

Everything from modules 4 to 11, applied to one matrix. The new element is the benchmark.

**Cash benchmark.** The default null is "expected return ≤ 0". A long-only rule on an asset that rose earns positive returns just by being long part of the time.

**Decomposition.** For a long-only rule with position $`w_t \in \{0,1\}`$:

```math
E[w_{t-1} r_t] = \underbrace{E[w_{t-1}]\,E[r_t]}_{\text{exposure}\ \times\ \text{drift}} + \underbrace{\text{Cov}(w_{t-1}, r_t)}_{\text{timing}}
```

The first term is market exposure and requires no skill. Only the covariance term is timing ability.

**Active returns.** To test timing, subtract buy-and-hold of the same asset: $`a_{k,t} = \text{net}_{k,t} - r_t`$. Running the diagnostics on the matrix of $`a`$ asks "does any variant beat holding the coin?" A variant that is flat half the time in a rising market has negative active return even if its Sharpe ratio is higher than the coin's, so this is a strict test.

**Reading the outputs together.**

| Output | Question it answers |
|---|---|
| Naive count | How many would a careless researcher report? |
| BH count | Is there a broad effect, allowing some false positives? |
| Holm count | Is any single variant trustworthy? |
| SPA p-value | Is the best variant better than cash, given the whole search? |
| DSR | Is the best Sharpe above what the best of N lucky draws would show? |
| PBO | Do in-sample rankings persist out of sample? |
| Walk-forward, CPCV | What would selecting the winner actually have earned? |
| MinBTL | Was the sample long enough for this many trials? |
| Active-return column | Is any of it more than holding the asset? |

### Where it lives

`experiments/e3_real.py`: `load_universe`, `buy_and_hold_matrix`, `family_table`, `verdict`, `run`. The diagnostics are in `experiments/common.py: diagnose`. Results: `reports/results/e3.json`; figures `e3_*.png`.

### Common mistakes and how the code avoids them

- **Presenting the best backtest.** The report leads with the counts and the corrections, not the winner's equity curve.
- **Hiding the data source.** The source label is stored with the results and printed in README and REPORT.
- **Testing only against cash.** The active-return analysis is in the same table.
- **Tuning after looking.** Grids, costs and sample start are in the config and were not changed after the first real-data run.
- **Treating BH discoveries as tradeable strategies.** The family table and the active-return column show they are mostly shared market exposure.

### Interview questions

**Q1. Summarise the real-data result.**
Out of 5,000 variants, over a thousand pass a naive 5% test, a few hundred survive false-discovery-rate control, and none survives family-wise control. The best Sharpe is below the level expected from luck given the effective number of trials, SPA does not reject, and nothing is significant relative to buy-and-hold. The exact figures are in the README.

**Q2. BH finds hundreds of survivors. Are those real strategies?**
They have genuinely positive expected returns relative to cash, but the source is long exposure to assets that rose a great deal. Against buy-and-hold of the same asset none is significant even before correction. So they are real exposure and not evidence of timing skill.

**Q3. PBO is well below 0.5. Does that contradict "nothing survives"?**
No. PBO measures whether rankings persist. Trend rules on rising coins rank high in most sub-samples because they share the same exposure, so the in-sample winner tends to stay in the top half. PBO does not test whether the level of performance exceeds luck.

**Q4. The best variant is a long-only moving-average rule on one coin. What does that tell you?**
That the winner is a diluted version of holding the best-performing asset. The selection picked the asset as much as the rule, which is exactly the kind of result selection bias produces.

**Q5. What would you need to see to believe one of these strategies?**
Significance against buy-and-hold after correction, stability across sub-periods and assets, a walk-forward Sharpe close to the in-sample one, and ideally an economic reason it should work, plus true out-of-sample data collected after the research was frozen.

### Self-quiz

1. A long-only rule is in the market 60% of the time on an asset with mean daily return 0.2% and has zero timing skill. Its expected daily return?
2. Why can BH and Holm disagree so sharply?
3. What does a DSR of 0.3 mean in words?
4. Why subtract buy-and-hold of the *same* asset rather than of Bitcoin?
5. Which output is closest to "what would I have earned"?

<details><summary>Answers</summary>

1. 0.6 × 0.2% = 0.12%.
2. BH tolerates a fraction of false discoveries and so picks up a broad weak effect; Holm requires each discovery to be individually beyond doubt.
3. After allowing for the search, there is about a 30% confidence that the best strategy's true Sharpe exceeds what luck would give: no evidence of skill.
4. To remove the drift the variant itself was exposed to; using another asset would mix in the relative performance of the two coins.
5. The walk-forward out-of-sample Sharpe.

</details>

---

## 15. E4: cost sensitivity

### What it does

Repeats the real-data analysis for exchange fees from 0 to 30 basis points per side, to see whether the conclusion depends on the cost assumption.

### The maths

Net returns are linear in the fee f:

```math
\text{net}_{k,t}(f) = g_{k,t} - f\cdot\text{turnover}_{k,t} - \text{slip}_{k,t}
```

so the mean return of variant k falls linearly, $`\overline{\text{net}}_k(f) = \bar g_k - f\,\overline{\text{turnover}}_k - \overline{\text{slip}}_k`$, with slope equal to its average turnover.

- **Annual cost drag** ≈ f × turnover per year. A variant that turns over its capital 30 times a year at 10 bps loses 3% a year.
- **Break-even fee**: $`f^* = (\bar g_k - \overline{\text{slip}}_k)/\overline{\text{turnover}}_k`$, the fee at which the variant's mean net return is zero.
- **Effect on the Sharpe ratio**: approximately $`\Delta SR \approx -f \times \text{annual turnover} / \text{annual volatility}`$. With 30 turns a year, 60% volatility and a 10 bp increase: -0.05.

High-turnover families (short-lookback momentum, tight mean reversion) are hit hardest; slow trend rules barely move.

### Where it lives

`experiments/e4_costs.py: run` calls `Universe.net(fee)` and the light diagnostics for each fee. Turnover per family is in the E3 family table. Figure: `reports/figures/e4_costs.png`.

### Common mistakes and how the code avoids them

- **Re-running the backtest per fee** and risking inconsistent inputs. One backtest, three stored matrices.
- **Varying costs to find a nice headline.** All fee levels are reported in one table; the headline uses the pre-set default of 10 bps.
- **Forgetting slippage.** Only the fee varies; slippage stays on, which the report states.
- **Assuming costs only shift everything down equally.** They reorder the variants because turnover differs.

### Interview questions

**Q1. How sensitive are your conclusions to transaction costs?**
The family-wise conclusion does not change: no variant survives Holm at any fee from 0 to 30 bps. The false-discovery count and the naive count fall as fees rise. The best variants are slow trend rules with low turnover, so their Sharpe ratios hardly move.

**Q2. Why do costs matter more for some strategies than others?**
Cost drag is fee times turnover. A rule that flips position every few days pays the fee dozens of times a year; a slow crossover pays it a handful of times.

**Q3. Is 10 bps per side realistic for crypto?**
It is in the range of taker fees on large exchanges for retail-sized accounts; institutions pay less and small accounts on some venues pay more. Fees are only part of the cost: spread and impact are modelled crudely here.

**Q4. If a strategy looks good gross and bad net, what do you conclude?**
That the edge, if any, is smaller than the cost of capturing it. It may still be exploitable by someone with lower costs or better execution, but not as backtested.

**Q5. How does cost interact with overfitting?**
A search over gross returns tends to select high-turnover rules that fit noise at short horizons. Costs penalise exactly those, so net-of-cost selection is somewhat less prone to picking them, but it does not remove selection bias.

### Self-quiz

1. Turnover of 50 times a year at 20 bps per side: annual cost?
2. Mean gross return 0.05% a day, mean turnover 0.1 a day, no slippage. Break-even fee?
3. Why is net return linear in the fee?
4. Which family in this project has the highest turnover?
5. Why might the best net Sharpe hardly change from 0 to 30 bps?

<details><summary>Answers</summary>

1. 50 × 0.002 = 10% of capital a year.
2. 0.0005 / 0.1 = 0.005, i.e. 50 bps.
3. The fee multiplies a fixed turnover series that does not depend on the fee.
4. Time-series momentum (see the E3 family table).
5. The best variants trade rarely, so their cost drag is small.

</details>

---

## 16. E5: robustness

### What it does

Changes the analysis settings one at a time and checks whether the conclusions move: number of CSCV blocks, number of CPCV groups, embargo length, sample period, bootstrap block length, and one extra bar of execution delay.

### The maths

There is one idea: a conclusion that depends on an arbitrary setting is not a conclusion. Each knob has a trade-off.

| Knob | Trade-off |
|---|---|
| CSCV blocks S | more blocks → more splits ($`\binom{S}{S/2}`$) but shorter blocks (T/S days) with noisier Sharpe ratios |
| CPCV groups N | more groups → more paths ($`\binom{N-1}{k-1}`$) and larger training sets (fraction 1 - k/N) but shorter test groups |
| Embargo | longer → less leakage but less training data |
| Bootstrap block | longer → more dependence preserved but fewer effective blocks (about T / block length) |
| Sample period | shorter → conclusions specific to a regime, and larger standard errors ($`1/\sqrt{\text{years}}`$) |
| Execution delay | one more bar → tests whether results rely on trading at the signal bar's close |

**Sub-periods and standard error.** Cutting the sample into three roughly doubles the standard error of a Sharpe ratio ($`\sqrt 3 \approx 1.7`$). The best Sharpe in a short period is therefore larger, and so is its luck benchmark; the comparison between the two is what matters.

**Persistence check.** For each sub-period's winner, the Sharpe ratio it earned in the rest of the sample is reported. If selection found something real, the winner should do well elsewhere too.

### Where it lives

`experiments/e5_robustness.py: run`, with `purged_kfold_selection` for the k-fold comparison. Settings are under `experiments.e5` in the config. Figure: `reports/figures/e5_robustness.png`.

### Common mistakes and how the code avoids them

- **Reporting only the default setting.** Every setting's result is in a table.
- **Changing several knobs at once.** One at a time, so each effect is attributable.
- **Choosing the setting that gives the preferred answer.** Defaults were fixed in the config before the robustness run.
- **Over-reading sub-period differences.** Each sub-period has a third of the data and much wider error bars.

### Interview questions

**Q1. Which of your conclusions are robust and which are not?**
Robust: no Holm survivors in any sub-period or at any setting, PBO in a narrow band across block counts, SPA p-values above 5% for all block lengths. Less stable: the CPCV median path Sharpe moves noticeably with the number of groups and the embargo, and PBO differs a lot between sub-periods.

**Q2. Why does PBO differ so much between sub-periods?**
In a period with a strong trend, trend rules rank high throughout and PBO is low. In a choppy or falling period there is no persistent ranking, the winner is noise, and PBO rises towards or above 0.5. PBO describes the sample as well as the method.

**Q3. What did the extra bar of delay show?**
That the results do not depend on trading at the same close the signal was computed from. The best variants are slow trend rules, for which one day of delay matters little.

**Q4. How do you choose the embargo length?**
There is no formula. It should cover the horizon over which neighbouring observations share information, which depends on holding periods and indicator lookbacks. Lopez de Prado suggests about 1% of the sample as a starting point; I varied it from 0 to 5%.

**Q5. Is a robustness section itself a multiple-testing problem?**
Yes, if you pick the setting you like. Reporting every setting and fixing the defaults in advance avoids that.

### Self-quiz

1. S = 12 blocks on 2,880 days: block length and number of splits?
2. CPCV with N = 8, k = 2: what fraction of data trains each split (before purging)?
3. Standard error of an annualised Sharpe over 2.6 years?
4. Why can a longer embargo raise or lower the measured out-of-sample Sharpe?
5. What would it mean if results collapsed with one extra bar of delay?

<details><summary>Answers</summary>

1. 240 days; C(12,6) = 924 splits.
2. 75%.
3. 1/√2.6 ≈ 0.62.
4. It removes training data near the test block: less leakage lowers an inflated score, but a different training set can also select a different, better or worse, variant.
5. That the profits came from the first bar after the signal, which is where execution assumptions are least reliable, so they would be suspect.

</details>

---

## One-page summary

- A Sharpe ratio from y years of data has a standard error of about $`1/\sqrt y`$.
- The best of N independent skill-less strategies shows about $`\sqrt{2\ln N}`$ standard errors of Sharpe by luck.
- **PSR**: is one pre-chosen strategy's Sharpe above a benchmark, given sample length and non-normality?
- **DSR**: the same question with the benchmark set to the expected best of N lucky trials.
- **PBO**: how often does the in-sample winner end up in the bottom half out of sample?
- **Purged CV / CPCV**: out-of-sample evaluation that respects time dependence, giving a distribution of backtest paths.
- **Reality Check / SPA**: bootstrap test of whether the best strategy beats the benchmark, preserving correlation between strategies.
- **Bonferroni, Holm**: no false discovery at all (FWER). **BH, BY**: a controlled fraction of false discoveries (FDR).
- **MinBTL**: how long a backtest must be for a given number of trials.
- Always test the tools on data where the answer is known before using them on data where it is not.
