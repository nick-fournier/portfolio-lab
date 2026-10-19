"""Class 3 optimizer: the trainer on synthetic data with known answers (toy ladder, 2026-10-19).

The real walk (walk.walk) with every extra off, on stocks whose features are drawn uniform in
[-0.5, 0.5] (like ranked inputs) and whose returns are a known linear signal plus noise:
rung 1, a linear model reaches the paper loss's closed-form minimum; rungs 2-3, an MLP and a
one-block transformer learn the signal and rank fresh months by it.
"""

from datetime import date

import numpy as np
import pytest

from portfolio_lab.research.optimizer import data

N, D, T = 30, 4, 36
BETA = np.array([1.0, 0.5, -0.5, 0.0])


def _months(seed=11, s=0.25, fresh=6):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(T + 1 + fresh):
        x = rng.uniform(-0.5, 0.5, (N, D)).astype(np.float32)
        r = (x @ (s * BETA) + rng.normal(0, 0.10, N) + rng.normal(0, 0.04)).astype(np.float32)
        day = date(2000 + k // 12, k % 12 + 1, 28)
        out.append(data.Month(day, [f"S{i}" for i in range(N)], x, r))
    return out


def _settings(model, seed, **kw):
    from portfolio_lab.research.optimizer.walk import Settings  # noqa: PLC0415

    base = dict(window=T, lr=1e-2, schedule="cosine", lr_min=1e-5, clip=None, seed=seed,
                start=date(2000 + T // 12, T % 12 + 1, 28), blocks=1, hidden=16)  # fmt: skip
    base.update(kw)
    return Settings(model, **base)


def test_a_linear_model_reaches_the_paper_losss_closed_form_minimum():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer.walk import walk  # noqa: PLC0415

    months = _months()
    f = np.array([m.x.astype(np.float64).T @ m.r.astype(np.float64) for m in months[:T]])
    lam_star = np.linalg.solve(f.T @ f, f.T @ np.ones(T))  # minimizes mean (1 - lam . f_k)^2
    best = np.mean((1 - f @ lam_star) ** 2)
    for seed in (0, 1):
        _, _, model = walk(months[: T + 1], _settings("linear", seed, lr=0.1, first_epochs=120))
        lam = model.lam.detach().double().numpy()
        assert np.mean((1 - f @ lam) ** 2) == pytest.approx(best, rel=1e-3)
        assert lam @ lam_star / np.linalg.norm(lam) / np.linalg.norm(lam_star) > 0.999


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_one_block_learns_the_planted_signal_and_ranks_fresh_months_by_it(kind):
    torch = pytest.importorskip("torch")
    from scipy.stats import spearmanr  # noqa: PLC0415

    from portfolio_lab.research.optimizer.walk import walk  # noqa: PLC0415

    months = _months()
    books = []
    for seed in (0, 1):
        _, _, model = walk(months[: T + 1], _settings(kind, seed, first_epochs=60))
        with torch.no_grad():
            fresh = [model(torch.from_numpy(m.x)).numpy() for m in months[T + 1 :]]
        rank = np.mean([spearmanr(w, m.x @ BETA).correlation
                        for w, m in zip(fresh, months[T + 1 :], strict=True)])  # fmt: skip
        assert rank > 0.85, (kind, seed, rank)
        books.append(np.concatenate(fresh))
    assert np.corrcoef(books)[0, 1] > 0.9  # the seeds agree
