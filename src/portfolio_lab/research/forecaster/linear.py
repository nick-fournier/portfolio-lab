"""Part 1: partial least squares on the stock inputs and their interactions with the market.

The design matrix is ``[x, x·vix, x·dispersion]`` with a leading 1 (``x`` the stock
inputs, missing as 0). Everything is fitted from per-month sums (``X'X``, ``X'y``,
``y'y``), so any set of months is fitted by adding its months' sums.

PLS on standardized inputs: the first component points along the inputs' covariance with
the target, each next one along the covariance matrix times the last, made orthogonal to
the earlier ones (the Krylov form); the coefficients are least squares within the first
``k`` components. ``k`` is chosen from :data:`COMPONENTS` by k-fold over the past: each
calendar year of the training months is held out in turn (fitting on the others, at least
:data:`MIN_FIT_MONTHS` of them) and the ``k`` with the lowest total squared error over the
held-out years wins.
"""

import numpy as np

COMPONENTS = (1, 2, 3, 4, 5, 6, 8, 10)
MIN_FIT_MONTHS = 24

Sums = tuple[np.ndarray, np.ndarray, float]


def design(x: np.ndarray, vix: np.ndarray, dispersion: np.ndarray) -> np.ndarray:
    """``[1, x, x·vix, x·dispersion]`` per row (``vix`` and ``dispersion`` are per row)."""
    x = np.nan_to_num(x)
    return np.hstack([np.ones((len(x), 1)), x, x * vix[:, None], x * dispersion[:, None]])


def month_sums(x: np.ndarray, y: np.ndarray) -> Sums:
    """``X'X``, ``X'y`` and ``y'y`` of one month's design rows and capped targets."""
    return x.T @ x, x.T @ y, float(y @ y)


def total(sums: list[Sums]) -> Sums:
    """The sums of a set of months."""
    return (sum(s[0] for s in sums), sum(s[1] for s in sums), sum(s[2] for s in sums))


def _standardized(g: np.ndarray, c: np.ndarray):
    """Correlation matrix, standardized covariances with y, means, sds and mean y."""
    n = g[0, 0]
    mu = g[0, 1:] / n
    cov = g[1:, 1:] / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    ybar = c[0] / n
    return cov / np.outer(sd, sd), (c[1:] / n - mu * ybar) / sd, mu, sd, ybar


def paths(sums: Sums) -> np.ndarray:
    """Raw coefficients (intercept first) for every count in :data:`COMPONENTS`, as columns."""
    g, c, _ = sums
    s, r, mu, sd, ybar = _standardized(g, c)
    basis = [r / np.linalg.norm(r)]
    for _ in range(max(COMPONENTS) - 1):
        v = s @ basis[-1]
        for u in basis:
            v = v - (u @ v) * u
        basis.append(v / np.linalg.norm(v))
    cols = []
    for k in COMPONENTS:
        b = np.column_stack(basis[:k])
        cols.append(b @ np.linalg.solve(b.T @ s @ b, b.T @ r))
    coef = np.column_stack(cols) / sd[:, None]
    return np.vstack([ybar - mu @ coef, coef])


def errors(coef: np.ndarray, sums: Sums) -> np.ndarray:
    """Total squared error of each coefficient column on the months in ``sums``."""
    g, c, s = sums
    return s - 2 * c @ coef + np.einsum("ik,ij,jk->k", coef, g, coef)


def fit(months: list, sums: dict) -> tuple[np.ndarray, int]:
    """Coefficients fitted on all ``months``, with the component count chosen by k-fold.

    Args:
        months: The training month ends.
        sums: :func:`month_sums` per month end.

    Returns:
        The coefficients (intercept first) and the number of components chosen.
    """
    everything = total([sums[d] for d in months])
    err = np.zeros(len(COMPONENTS))
    for year in sorted({d.year for d in months}):
        held_months = [d for d in months if d.year == year]
        if len(months) - len(held_months) < MIN_FIT_MONTHS:
            continue
        held = total([sums[d] for d in held_months])
        rest = (everything[0] - held[0], everything[1] - held[1], everything[2] - held[2])
        err = err + errors(paths(rest), held)
    j = int(np.argmin(err))
    return paths(everything)[:, j], COMPONENTS[j]
