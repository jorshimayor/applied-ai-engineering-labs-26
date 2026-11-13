"""Small-sample statistics for evaluating LLM experiments. Week 4.

Everything here works on PAIRED outcomes: the same test items scored under two
conditions. Pairing removes item difficulty from the comparison, which is the
single biggest gain in statistical power available to small evals.

No scipy: exact tests are written out with math.comb so each one is readable.
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ intervals
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n. Behaves well near 0 and 1,
    unlike the textbook p ± 1.96·sqrt(p(1-p)/n), which gives [0, 0] for 0/30."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) over items, with a percentile bootstrap CI that resamples ITEMS.
    a, b: per-item scores (0/1, or a per-item mean over seeds)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------- tests
def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test. b = items only A got right, c = items only B got right.
    Items both got right (or both wrong) carry no information about the difference."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(b, c), n))


def discordant(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return int((a & ~b).sum()), int((~a & b).sum())


def sign_flip_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20_000, seed: int = 0) -> float:
    """Paired permutation test on per-item differences (works for non-binary per-item means).
    Under H0 each item's difference is equally likely to have either sign."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    row1, row2, col1, n = a + b, c + d, a + c, a + b + c + d

    def pmf(x):
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    p_obs = pmf(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error across many claims)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------- power
def power_mcnemar(n: int, delta: float, discord: float, alpha: float = 0.05,
                  sims: int = 4000, seed: int = 0) -> float:
    """Simulated power of the exact McNemar test.
    delta   = true accuracy difference (A - B)
    discord = fraction of items where A and B disagree (the only items that carry signal)"""
    if discord < abs(delta):
        raise ValueError("discordance must be at least |delta|")
    p_b_only, p_a_only = (discord - delta) / 2, (discord + delta) / 2
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, [p_a_only, p_b_only, 1 - discord], size=sims)
    crit = {}
    hits = 0
    for a_only, b_only, _ in counts:
        key = (int(a_only), int(b_only))
        if key not in crit:
            crit[key] = mcnemar_exact(*key) < alpha
        hits += crit[key]
    return hits / sims


def n_for_power(delta: float, discord: float, target: float = 0.8, alpha: float = 0.05,
                n_max: int = 2000, step: int = 10) -> int | None:
    for n in range(step, n_max + 1, step):
        if power_mcnemar(n, delta, discord, alpha) >= target:
            return n
    return None
