"""Multiple-testing corrections on strategy p-values and the haircut Sharpe ratio.

With N strategies each tested at level alpha, about alpha*N will look significant by
luck. The corrections below turn raw p-values into ADJUSTED p-values; a strategy
"survives" a correction if its adjusted p-value is below alpha.

Family-wise error rate (FWER = probability of at least ONE false discovery):
* Bonferroni:  p_adj = min(1, N p).                              Valid under any dependence.
* Holm (1979): step-down Bonferroni, uniformly more powerful.    Valid under any dependence.

False discovery rate (FDR = expected FRACTION of discoveries that are false):
* Benjamini & Hochberg (1995): valid under independence / positive dependence.
* Benjamini & Yekutieli (2001): valid under any dependence (BH with an extra factor
  c(N) = 1 + 1/2 + ... + 1/N).

References
----------
* Holm, S. (1979). Scandinavian Journal of Statistics 6(2), 65-70.
* Benjamini, Y. and Hochberg, Y. (1995). JRSS B 57(1), 289-300.
* Benjamini, Y. and Yekutieli, D. (2001). Annals of Statistics 29(4), 1165-1188.
* Harvey, C. R. and Liu, Y. (2015). "Backtesting." Journal of Portfolio Management
  42(1), 13-28.  -> haircut Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bonferroni(p: np.ndarray) -> np.ndarray:
    """Bonferroni-adjusted p-values: ``min(1, N p)``."""
    p = np.asarray(p, dtype=float)
    return np.minimum(1.0, p * p.size)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values.

    Sort p_(1) <= ... <= p_(N). ``p_adj_(i) = max_{j<=i} min(1, (N - j + 1) p_(j))``.
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.minimum(1.0, (n - np.arange(n)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(n)
    out[order] = adj
    return out


def _step_up(p: np.ndarray, factor: float) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    scaled = factor * n / np.arange(1, n + 1) * p[order]
    adj = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    out = np.empty(n)
    out[order] = adj
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values).

    ``p_adj_(i) = min_{j>=i} min(1, N p_(j) / j)``.
    """
    return _step_up(np.asarray(p, dtype=float), 1.0)


def benjamini_yekutieli(p: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values: BH multiplied by c(N) = sum_{i<=N} 1/i."""
    p = np.asarray(p, dtype=float)
    return _step_up(p, float(np.sum(1.0 / np.arange(1, p.size + 1))))


ADJUSTMENTS = {
    "bonferroni": bonferroni,
    "holm": holm,
    "bh": benjamini_hochberg,
    "by": benjamini_yekutieli,
}


def survivors(p: np.ndarray, alpha: float = 0.05) -> dict[str, int]:
    """Number of strategies significant at ``alpha``: naive and after each correction."""
    p = np.asarray(p, dtype=float)
    out = {"naive": int(np.sum(p < alpha))}
    for name, func in ADJUSTMENTS.items():
        out[name] = int(np.sum(func(p) < alpha))
    return out


def haircut_sharpe(
    sr_annual: float,
    n_obs: int,
    periods_per_year: float,
    all_sr_annual: np.ndarray | None = None,
    n_tests: float | None = None,
) -> dict[str, dict[str, float]]:
    """Haircut Sharpe ratio (Harvey & Liu 2015).

    Idea: convert the Sharpe ratio to a t-statistic ``t = SR_annual * sqrt(years)`` and a
    two-sided p-value, inflate the p-value for multiple testing, convert the adjusted
    p-value back into a t-statistic and hence a Sharpe ratio. The relative drop is the
    "haircut".

    Methods: ``bonferroni`` uses ``n_tests`` (default: the number of strategies); ``holm``
    and ``bhy`` (Benjamini-Hochberg-Yekutieli) need the p-values of all tests.

    SIMPLIFICATION relative to the paper: Harvey & Liu do not observe the other tests, so
    they simulate their t-statistics from a model fitted to published factor studies.
    Here every trial is observed, so the adjusted p-value of the chosen strategy is
    computed from the ACTUAL p-values of all variants (``all_sr_annual``).
    """
    years = n_obs / periods_per_year

    def p_two_sided(sr):
        return 2.0 * norm.sf(np.abs(np.asarray(sr, dtype=float)) * np.sqrt(years))

    p_single = float(p_two_sided(sr_annual))
    if n_tests is None:
        n_tests = 1.0 if all_sr_annual is None else float(len(all_sr_annual))
    adjusted = {"bonferroni": min(1.0, n_tests * p_single)}
    if all_sr_annual is not None:
        p_all = p_two_sided(all_sr_annual)
        k = int(np.argmin(np.abs(p_all - p_single)))
        adjusted["holm"] = float(holm(p_all)[k])
        adjusted["bhy"] = float(benjamini_yekutieli(p_all)[k])
    out = {}
    for name, p_adj in adjusted.items():
        t_adj = float(norm.isf(p_adj / 2.0)) if p_adj < 1.0 else 0.0
        sr_adj = t_adj / np.sqrt(years)
        out[name] = {
            "p_adjusted": p_adj,
            "sharpe_annual": sr_adj,
            "haircut": 1.0 - sr_adj / sr_annual if sr_annual > 0 else float("nan"),
        }
    out["single"] = {"p_adjusted": p_single, "sharpe_annual": sr_annual, "haircut": 0.0}
    return out
