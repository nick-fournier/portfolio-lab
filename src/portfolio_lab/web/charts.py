"""Plotly figures for a backtest run, returned as JSON for plotly.js in the browser."""

import numpy as np
import plotly.graph_objects as go
import polars as pl

_LAYOUT = {
    "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
    "template": "plotly_white",
    "legend": {"orientation": "h", "y": 1.08},
    "hovermode": "x unified",
}


def equity_figure(daily: pl.DataFrame, strategy: str, benchmark: str) -> str:
    """Growth of $1 for the strategy and its benchmark, log scale."""
    fig = go.Figure(layout=_LAYOUT)
    fig.add_scatter(x=daily["date"].to_list(), y=daily["nav"].to_list(), name=strategy)
    if "benchmark_ret" in daily.columns:
        bench = np.cumprod(1 + daily["benchmark_ret"].to_numpy())
        fig.add_scatter(x=daily["date"].to_list(), y=bench.tolist(), name=benchmark)
    fig.update_layout(title="Growth of $1", yaxis_type="log")
    return fig.to_json()


def drawdown_figure(daily: pl.DataFrame) -> str:
    """Percentage below the running peak of NAV."""
    nav = daily["nav"].to_numpy()
    drawdown = nav / np.maximum.accumulate(np.r_[1.0, nav])[1:] - 1
    fig = go.Figure(layout=_LAYOUT)
    fig.add_scatter(x=daily["date"].to_list(), y=drawdown.tolist(), fill="tozeroy", name="drawdown")
    fig.update_layout(title="Drawdown", yaxis_tickformat=".0%", showlegend=False)
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
    fig.update_layout(title=f"Target weights, top {len(names)} names", height=120 + 22 * len(names))
    return fig.to_json()
