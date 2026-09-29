"""Plotly figures for a backtest run, returned as JSON for plotly.js in the browser."""

import numpy as np
import plotly.graph_objects as go
import polars as pl

# Titles live in the page (HTML headings), not in the figure, and the legend sits below the
# plot area, so neither can cover the data on narrow (mobile) screens.
_LAYOUT = {
    "margin": {"l": 45, "r": 10, "t": 10, "b": 10},
    "template": "plotly_white",
    "legend": {"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0},
    "hovermode": "x unified",
    "font": {"size": 11},
}


def equity_figure(daily: pl.DataFrame, strategy: str, benchmark: str) -> str:
    """Growth of $1 for the strategy and its benchmark, log scale."""
    fig = go.Figure(layout=_LAYOUT)
    fig.add_scatter(x=daily["date"].to_list(), y=daily["nav"].to_list(), name=strategy)
    if "benchmark_ret" in daily.columns:
        bench = np.cumprod(1 + daily["benchmark_ret"].to_numpy())
        fig.add_scatter(x=daily["date"].to_list(), y=bench.tolist(), name=benchmark)
    fig.update_layout(yaxis_type="log")
    return fig.to_json()


def drawdown_figure(daily: pl.DataFrame) -> str:
    """Percentage below the running peak of NAV."""
    nav = daily["nav"].to_numpy()
    drawdown = nav / np.maximum.accumulate(np.r_[1.0, nav])[1:] - 1
    fig = go.Figure(layout=_LAYOUT)
    fig.add_scatter(x=daily["date"].to_list(), y=drawdown.tolist(), fill="tozeroy", name="drawdown")
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    return fig.to_json()


def weights_figure(weights: pl.DataFrame, top: int = 20) -> str | None:
    """Heatmap of target weights over time for the ``top`` names by average weight."""
    if weights.is_empty():
        return None
    dates = weights["date"].unique().sort()
    avg = (
        weights.group_by("symbol")
        .agg((pl.col("weight").sum() / dates.len()).alias("avg"))
        .sort("avg", descending=True)
        .head(top)
    )
    names = avg["symbol"].to_list()
    grid = (
        pl.DataFrame({"date": dates})
        .join(weights.filter(pl.col("symbol").is_in(names)), on="date", how="left")
        .pivot(on="symbol", index="date", values="weight")
        .sort("date")
    )
    z = [[row.get(n) or 0.0 for row in grid.iter_rows(named=True)] for n in names]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=grid["date"].to_list(),
            y=names,
            colorscale="Blues",
            zmin=0,
            colorbar={"tickformat": ".1%"},
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        ),
        layout=_LAYOUT,
    )
    fig.update_layout(height=80 + 22 * len(names), hovermode="closest")
    return fig.to_json()


def comparison_figure(series: list[tuple[str, pl.DataFrame]], benchmark: str) -> str | None:
    """Growth of $1 for several runs on one log-scale chart, plus the benchmark.

    Args:
        series: ``(label, daily)`` pairs, one per run.
        benchmark: Benchmark label; its curve comes from the run covering the most days.
    """
    if not series:
        return None
    fig = go.Figure(layout=_LAYOUT)
    for label, daily in series:
        fig.add_scatter(x=daily["date"].to_list(), y=daily["nav"].to_list(), name=label)
    longest = max((daily for _, daily in series), key=lambda d: d.height)
    if "benchmark_ret" in longest.columns:
        bench = np.cumprod(1 + longest["benchmark_ret"].to_numpy())
        fig.add_scatter(
            x=longest["date"].to_list(),
            y=bench.tolist(),
            name=f"{benchmark} (benchmark)",
            line={"color": "#6e7781", "dash": "dot"},
        )
    fig.update_layout(yaxis_type="log")
    return fig.to_json()


def cumulative_ic_figure(scores: pl.DataFrame) -> str:
    """Running sum of monthly rank IC per signal: a steady climb means persistent skill."""
    fig = go.Figure(layout=_LAYOUT)
    for (signal,), rows in sorted(scores.sort("date").group_by("signal", maintain_order=True)):
        fig.add_scatter(
            x=rows["date"].to_list(), y=rows["ic"].cum_sum().to_list(), name=signal, mode="lines"
        )
    fig.add_hline(y=0, line={"color": "#6e7781", "width": 1})
    return fig.to_json()
