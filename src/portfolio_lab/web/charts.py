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
    # 12 distinct colors (Plotly's default has 10, so an 11th series repeated the first).
    "colorway": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#17becf", "#bcbd22", "#393b79", "#ad494a", "#637939",
    ],
}  # fmt: skip
#: Thin lines, so charts with many series stay readable where they cross.
_LINE = {"width": 1.4}
#: Style of the market reference line (the benchmark, or buy-and-hold of it).
_REFERENCE = {"color": "#6e7781", "dash": "dot", "width": 1.4}


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


def comparison_figure(
    series: list[tuple[str, pl.DataFrame, bool, bool]], benchmark: str | None
) -> str | None:
    """Growth of $1 for several runs on one log-scale chart, plus the benchmark.

    Args:
        series: ``(label, daily, visible, reference)`` per run. Hidden runs can be shown
            from the legend; reference runs (buy-and-hold of the benchmark) are drawn in the
            benchmark's grey dotted style.
        benchmark: Benchmark label, or ``None`` to leave it out (e.g. when a buy-and-hold
            run of it is already drawn); its curve comes from the run covering the most days.
    """
    if not series:
        return None
    fig = go.Figure(layout=_LAYOUT)
    for label, daily, visible, reference in series:
        fig.add_scatter(
            x=daily["date"].to_list(),
            y=daily["nav"].to_list(),
            name=label,
            line=_REFERENCE if reference else _LINE,
            visible=True if visible else "legendonly",
        )
    longest = max((s[1] for s in series), key=lambda d: d.height)
    if benchmark and "benchmark_ret" in longest.columns:
        bench = np.cumprod(1 + longest["benchmark_ret"].to_numpy())
        fig.add_scatter(
            x=longest["date"].to_list(),
            y=bench.tolist(),
            name=f"{benchmark} (benchmark)",
            line=_REFERENCE,
        )
    fig.update_layout(yaxis_type="log")
    return fig.to_json()


def cumulative_ic_figure(scores: pl.DataFrame) -> str:
    """Running sum of monthly rank IC per signal: a steady climb means persistent skill."""
    fig = go.Figure(layout=_LAYOUT)
    for (signal,), rows in sorted(scores.sort("date").group_by("signal", maintain_order=True)):
        fig.add_scatter(
            x=rows["date"].to_list(),
            y=rows["ic"].cum_sum().to_list(),
            name=signal,
            mode="lines",
            line=_LINE,
        )
    fig.add_hline(y=0, line={"color": "#6e7781", "width": 1})
    return fig.to_json()


def calibration_figure(bins: pl.DataFrame) -> str:
    """Stated versus realized chance of beating the median, per confidence decile."""
    fig = go.Figure(layout=_LAYOUT)
    lo = min(bins["predicted"].min(), bins["realized"].min())
    hi = max(bins["predicted"].max(), bins["realized"].max())
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfectly calibrated",
                    line=_REFERENCE)  # fmt: skip
    for (model,), rows in bins.sort("model", "bin").group_by("model", maintain_order=True):
        fig.add_scatter(
            x=rows["predicted"].to_list(), y=rows["realized"].to_list(), name=model,
            mode="lines+markers", line=_LINE,
        )  # fmt: skip
    fig.update_layout(
        xaxis={"title": "stated probability", "tickformat": ".0%"},
        yaxis={"title": "actually beat the median", "tickformat": ".0%"},
        hovermode="closest",
        legend={**_LAYOUT["legend"], "y": -0.3},  # below the axis title
        margin={**_LAYOUT["margin"], "b": 20},
    )
    return fig.to_json()


def importance_figure(table: pl.DataFrame, top: int = 15) -> str:
    """Largest drops in ranking accuracy (AUC) when an input is shuffled."""
    rows = table.sort("auc_drop", descending=True).head(top).reverse()
    fig = go.Figure(
        go.Bar(
            x=rows["auc_drop"].to_list(), y=rows["feature"].to_list(), orientation="h",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ),
        layout=_LAYOUT,
    )  # fmt: skip
    fig.update_layout(height=80 + 24 * rows.height, showlegend=False, hovermode="closest",
                      xaxis={"title": "drop in AUC when shuffled"})  # fmt: skip
    return fig.to_json()
