"""
stats_core.py
=============
Statistical primitives for the analysis of published, aggregate survey rates.

Every function here takes *effective* base sizes rather than respondent counts.
The CSBS applies design weights, which inflates the variance of every estimate;
the published effective base size is the sample size that an unweighted simple
random sample would need in order to achieve the same precision. Using the
respondent count instead would overstate precision by roughly 15-25% on these
data, so all inference in this project uses the effective base.

References for the methods used:
  Wilson, E.B. (1927) score interval for a binomial proportion.
  Newcombe, R.G. (1998) hybrid-score interval for the difference of two
      independent proportions (Method 10).
  Cohen, J. (1988) arcsine effect size h for the difference of two proportions.

Author: Frida Alejandra Ibarra Zavaleta
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

Z_95 = stats.norm.ppf(0.975)


# ---------------------------------------------------------------------------
# Single proportions
# ---------------------------------------------------------------------------

def wilson_ci(p: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the Wald interval because the published rates include values
    close to 0 and close to 1 (e.g. 0.014 for ransomware, 0.891 for large-firm
    policy adoption), where the Wald interval can extend outside [0, 1] and has
    poor coverage.

    Parameters
    ----------
    p : observed proportion in [0, 1]
    n : effective base size
    """
    if not np.isfinite(p) or not np.isfinite(n) or n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def moe(p: float, n: float, conf: float = 0.95) -> float:
    """Half-width of the Wilson interval, in proportion units."""
    lo, hi = wilson_ci(p, n, conf)
    return (hi - lo) / 2


# ---------------------------------------------------------------------------
# Differences between two proportions
# ---------------------------------------------------------------------------

@dataclass
class PropTest:
    """Result of a two-proportion comparison."""
    label: str
    p1: float
    n1: float
    p2: float
    n2: float
    diff_pp: float          # (p1 - p2) in percentage points
    ci_low_pp: float        # Newcombe CI on the difference
    ci_high_pp: float
    z: float
    p_value: float
    cohens_h: float
    h_magnitude: str
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h: the arcsine-transformed difference between two proportions.

    Unlike the raw percentage-point difference, h is stable across the range of
    p: a move from 0.02 to 0.04 and a move from 0.48 to 0.50 are both 2pp but
    are very different in practical terms, and h reflects that.
    """
    if not (np.isfinite(p1) and np.isfinite(p2)):
        return np.nan
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def h_magnitude(h: float) -> str:
    """Conventional Cohen benchmarks. Reported as a guide, not a rule."""
    if not np.isfinite(h):
        return "undefined"
    a = abs(h)
    if a < 0.20:
        return "negligible"
    if a < 0.50:
        return "small"
    if a < 0.80:
        return "medium"
    return "large"


def newcombe_diff_ci(p1: float, n1: float, p2: float, n2: float,
                     conf: float = 0.95) -> tuple[float, float]:
    """Newcombe hybrid-score interval for p1 - p2 (independent samples).

    Built from the two Wilson intervals rather than from a pooled normal
    approximation, so it behaves sensibly when either proportion is small.
    """
    l1, u1 = wilson_ci(p1, n1, conf)
    l2, u2 = wilson_ci(p2, n2, conf)
    lower = (p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lower, upper)


def two_proportion_test(p1: float, n1: float, p2: float, n2: float,
                        label: str = "", alpha: float = 0.05) -> PropTest:
    """Pooled two-proportion z-test plus Newcombe CI and Cohen's h.

    Assumes the two samples are independent. Where that assumption does not
    hold (see `dependent_note` in run_analysis.py) the test is used as a
    conservative screen rather than as an exact test, and this is stated
    wherever such a comparison is reported.
    """
    if not all(np.isfinite(v) for v in (p1, n1, p2, n2)):
        return PropTest(label, p1, n1, p2, n2, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, "undefined", False)

    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    lo, hi = newcombe_diff_ci(p1, n1, p2, n2)
    h = cohens_h(p1, p2)

    return PropTest(
        label=label,
        p1=p1, n1=n1, p2=p2, n2=n2,
        diff_pp=(p1 - p2) * 100,
        ci_low_pp=lo * 100, ci_high_pp=hi * 100,
        z=z, p_value=pval,
        cohens_h=h, h_magnitude=h_magnitude(h),
        significant=bool(np.isfinite(pval) and pval < alpha),
    )


# ---------------------------------------------------------------------------
# Composite indicators
# ---------------------------------------------------------------------------

def composite_mean(rates: list[float]) -> float:
    """Unweighted mean of component rates (the composite C_s)."""
    arr = np.asarray(rates, dtype=float)
    return float(np.nanmean(arr))


def composite_se_bounds(rates: list[float], n: float) -> tuple[float, float]:
    """Bounding standard errors for a composite of correlated proportions.

    The components of C_s (training, board responsibility, policy) are measured
    on the *same* respondents, so their sampling errors are correlated and the
    covariance terms are not recoverable from published tables. Rather than
    assume a value, two bounds are computed:

      lower bound  : components independent  -> SE = sqrt(sum(se_i^2)) / k
      upper bound  : components perfectly correlated -> SE = mean(se_i)

    The truth lies between them. Reporting both is honest about a quantity the
    published tables cannot pin down.
    """
    ses = [np.sqrt(p * (1 - p) / n) for p in rates if np.isfinite(p)]
    if not ses:
        return (np.nan, np.nan)
    k = len(ses)
    indep = float(np.sqrt(np.sum(np.square(ses))) / k)
    perfect = float(np.mean(ses))
    return (indep, perfect)


def differential_ci(v: float, n_v: float, c: float, n_c: float,
                    conf: float = 0.95) -> tuple[float, float]:
    """Conservative CI for the differential D_s = V_s - C_s, in pp.

    V_s is measured on AI-engaged organisations and C_s on all organisations,
    so the two samples overlap. Positive correlation between overlapping
    samples *reduces* the variance of the difference, which means an
    independence assumption yields a CI at least as wide as the truth. The
    interval returned here is therefore conservative in the direction that
    matters: it will not overstate the precision of D_s.
    """
    return tuple(x * 100 for x in newcombe_diff_ci(v, n_v, c, n_c, conf))


# ---------------------------------------------------------------------------
# Rank association at very small n
# ---------------------------------------------------------------------------

def spearman_exact(x: list[float], y: list[float]) -> dict:
    """Spearman rho with an exact permutation p-value.

    With n = 4 there are only 4! = 24 orderings, so the exact null distribution
    is enumerable and the smallest attainable two-sided p-value is 2/24 = 0.083.
    No result at this sample size can reach p < 0.05. The asymptotic p-value
    returned by standard software is invalid here and is deliberately not
    reported.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rho = stats.spearmanr(x, y).statistic

    perms = [stats.spearmanr(x, np.asarray(p)).statistic
             for p in itertools.permutations(y)]
    perms = np.asarray(perms)
    p_exact = float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))

    return {
        "rho": float(rho),
        "n": int(n),
        "p_exact_two_sided": p_exact,
        "min_attainable_p": float(2 / math.factorial(n)) if n <= 8 else np.nan,
        "n_permutations": int(len(perms)),
        "interpretation": (
            "Describes ordering only. At n=4 no permutation test can reach "
            "conventional significance; the coefficient carries no inferential "
            "weight."
        ),
    }


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down adjustment.

    This analysis makes many pairwise size-band comparisons across several
    indicators. At alpha = 0.05 and 18 tests, the probability of at least one
    false positive under a global null is about 60%. Holm controls the
    family-wise error rate without the conservatism of plain Bonferroni.
    """
    idx = np.argsort(pvals)
    m = len(pvals)
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, max(prev, (m - rank) * pvals[i]))
        prev = adj
        out[i] = {
            "p_raw": float(pvals[i]),
            "p_holm": float(adj),
            "reject_holm": bool(adj < alpha),
            "rank": int(rank + 1),
        }
    return out


# ---------------------------------------------------------------------------
# Model selection: competing interval estimators
# ---------------------------------------------------------------------------
#
# Section 3.4 of the dissertation justifies the choice of the Wilson interval
# over the alternatives that a reader might reasonably expect instead. That
# justification is empirical rather than asserted: the functions below implement
# each candidate and evaluate them on the bases and rates that actually occur in
# these data.

def wald_ci(p: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Textbook normal-approximation interval. Included as the baseline."""
    if not np.isfinite(p) or not np.isfinite(n) or n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    half = z * np.sqrt(p * (1 - p) / n)
    return (p - half, p + half)          # deliberately NOT clipped to [0, 1]


def clopper_pearson_ci(p: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Exact interval from the beta distribution. Guaranteed >= nominal coverage."""
    if not np.isfinite(p) or not np.isfinite(n) or n <= 0:
        return (np.nan, np.nan)
    x = int(round(p * n))
    n = int(round(n))
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else stats.beta.ppf(a, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a, x + 1, n - x)
    return (float(lo), float(hi))


def agresti_coull_ci(p: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Add-two-successes-add-two-failures adjustment to the Wald interval."""
    if not np.isfinite(p) or not np.isfinite(n) or n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n_t = n + z**2
    p_t = (p * n + z**2 / 2) / n_t
    half = z * np.sqrt(p_t * (1 - p_t) / n_t)
    return (max(0.0, p_t - half), min(1.0, p_t + half))


def jeffreys_ci(p: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Bayesian interval under the Jeffreys Beta(1/2, 1/2) prior."""
    if not np.isfinite(p) or not np.isfinite(n) or n <= 0:
        return (np.nan, np.nan)
    x = p * n
    a = (1 - conf) / 2
    lo = 0.0 if x <= 0 else stats.beta.ppf(a, x + 0.5, n - x + 0.5)
    hi = 1.0 if x >= n else stats.beta.ppf(1 - a, x + 0.5, n - x + 0.5)
    return (float(lo), float(hi))


INTERVAL_METHODS = {
    "Wald": wald_ci,
    "Wilson": wilson_ci,
    "Clopper-Pearson": clopper_pearson_ci,
    "Agresti-Coull": agresti_coull_ci,
    "Jeffreys": jeffreys_ci,
}


def interval_comparison(p: float, n: float, conf: float = 0.95) -> dict:
    """Width and admissibility of every candidate interval at one (p, n)."""
    out = {}
    for name, fn in INTERVAL_METHODS.items():
        lo, hi = fn(p, n, conf)
        out[name] = {
            "low": lo,
            "high": hi,
            "width_pp": (hi - lo) * 100,
            "admissible": bool(lo >= 0.0 and hi <= 1.0),
        }
    return out


def coverage_simulation(p_true: float, n: float, conf: float = 0.95,
                        n_sim: int = 20000, seed: int = 42) -> dict:
    """Monte Carlo coverage of each candidate interval at a given (p, n).

    An interval that claims 95% confidence should contain the true parameter in
    95% of repeated samples. The Wald interval does not, and the shortfall is
    worst exactly where these data live: rates near 0 or 1 on moderate bases.
    This function measures the shortfall rather than assuming it.
    """
    rng = np.random.default_rng(seed)
    n_int = int(round(n))
    draws = rng.binomial(n_int, p_true, size=n_sim)
    out = {}
    for name, fn in INTERVAL_METHODS.items():
        hits = 0
        widths = np.empty(n_sim)
        # Cache by observed count: only n+1 distinct outcomes are possible.
        cache = {}
        for i, x in enumerate(draws):
            if x not in cache:
                cache[x] = fn(x / n_int, n_int, conf)
            lo, hi = cache[x]
            widths[i] = hi - lo
            if lo <= p_true <= hi:
                hits += 1
        out[name] = {
            "coverage": hits / n_sim,
            "mean_width_pp": float(np.mean(widths) * 100),
            "coverage_error_pp": (hits / n_sim - conf) * 100,
        }
    return out


# ---------------------------------------------------------------------------
# Model selection: competing multiplicity corrections
# ---------------------------------------------------------------------------

def bonferroni(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    """Single-step Bonferroni. Controls FWER but is uniformly more conservative
    than Holm, which is why Holm is preferred here."""
    m = len(pvals)
    return [{"p_raw": float(p), "p_adj": float(min(1.0, m * p)),
             "reject": bool(min(1.0, m * p) < alpha)} for p in pvals]


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    """Step-up FDR control. Reported as a comparator: it controls a different
    and weaker error rate than Holm, so it is not the primary procedure for a
    confirmatory family of contrasts."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * m / (rank + 1))
        adj[i] = val
        prev = val
    return [{"p_raw": float(p[i]), "p_adj": float(adj[i]),
             "reject": bool(adj[i] < alpha)} for i in range(m)]


def multiplicity_comparison(pvals: list[float], alpha: float = 0.05) -> dict:
    """Rejection counts under no correction, Bonferroni, Holm and BH.

    Reported in Section 3.4 so the reader can see what the choice of procedure
    is worth in this application, rather than taking the choice on trust.
    """
    m = len(pvals)
    holm = holm_bonferroni(pvals, alpha)
    bonf = bonferroni(pvals, alpha)
    bh = benjamini_hochberg(pvals, alpha)
    return {
        "n_tests": m,
        "uncorrected": int(sum(p < alpha for p in pvals)),
        "bonferroni": int(sum(x["reject"] for x in bonf)),
        "holm": int(sum(x["reject_holm"] for x in holm)),
        "benjamini_hochberg": int(sum(x["reject"] for x in bh)),
        "fwer_under_global_null": float(1 - (1 - alpha) ** m),
        "detail": {"holm": holm, "bonferroni": bonf, "bh": bh},
    }


# ---------------------------------------------------------------------------
# Hyperparameter selection for the exploratory partition
# ---------------------------------------------------------------------------

def kmeans_sweep(X: np.ndarray, k_values=(2, 3), seed: int = 42,
                 n_init: int = 20) -> list[dict]:
    """Inertia and silhouette across candidate k, on standardised features.

    With four observations, k is bounded above by 3 (silhouette is undefined at
    k = n). The sweep is reported not to select a 'best' k in the usual sense but
    to show how little the criterion discriminates at this sample size.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    out = []
    n = X.shape[0]
    for k in k_values:
        if k >= n or k < 2:
            continue
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        out.append({
            "k": int(k),
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km.labels_)),
            "labels": km.labels_.tolist(),
            "cluster_sizes": np.bincount(km.labels_).tolist(),
        })
    return out


def partition_stability(X: np.ndarray, k: int = 2, seed: int = 42) -> dict:
    """Leave-one-feature-out stability of a k-means partition.

    A partition that survives dropping any single feature is not an artefact of
    that feature. This is the closest available analogue to cross-validation
    when there are four observations and no held-out set is possible.
    """
    from sklearn.cluster import KMeans

    base = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X).labels_
    agree = 0
    detail = []
    for f in range(X.shape[1]):
        lab = KMeans(n_clusters=k, n_init=20,
                     random_state=seed).fit(np.delete(X, f, axis=1)).labels_
        same = bool(np.array_equal(lab, base) or np.array_equal(1 - lab, base))
        agree += same
        detail.append({"feature_dropped": int(f), "same_partition": same})
    return {"n_features": int(X.shape[1]), "n_stable": int(agree),
            "base_labels": base.tolist(), "detail": detail}
