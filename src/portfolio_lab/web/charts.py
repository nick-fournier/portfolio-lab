"""Plotly figures for a backtest run, returned as JSON for plotly.js in the browser."""

import functools
import json

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

from portfolio_lab.web.series import Series, weekly

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


#: Zoom buttons on long charts: 1, 5 and 10 years back from the end, and everything.
RANGE_BUTTONS = {
    "buttons": [
        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
        {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
        {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
        {"step": "all", "label": "Max"},
    ],
    "x": 0, "y": 1.02, "xanchor": "left", "yanchor": "bottom", "font": {"size": 11},
}  # fmt: skip


def equity_figure(daily: pl.DataFrame, strategy: str, benchmark: str) -> str:
    """Growth of $1 for the strategy and its benchmark, log scale, weekly points."""
    frame = daily.select("date", "nav")
    if "benchmark_ret" in daily.columns:
        frame = frame.with_columns((1 + daily["benchmark_ret"]).cum_prod().alias("bench"))
    frame = weekly(frame)
    fig = go.Figure(layout=_LAYOUT)
    fig.add_scatter(x=frame["date"].to_list(), y=frame["nav"].round(4).to_list(), name=strategy)
    if "bench" in frame.columns:
        fig.add_scatter(
            x=frame["date"].to_list(), y=frame["bench"].round(4).to_list(), name=benchmark
        )
    fig.update_layout(yaxis_type="log", xaxis={"rangeselector": RANGE_BUTTONS})
    return fig.to_json()


def drawdown_figure(daily: pl.DataFrame) -> str:
    """Percentage below the running peak of NAV (weekly points, each week's low)."""
    nav = daily["nav"].to_numpy()
    frame = daily.select("date").with_columns(
        pl.Series("drawdown", nav / np.maximum.accumulate(np.r_[1.0, nav])[1:] - 1)
    )
    frame = frame.group_by_dynamic("date", every="1w").agg(pl.col("drawdown").min())
    fig = go.Figure(layout=_LAYOUT)
    fig.add_scatter(x=frame["date"].to_list(), y=frame["drawdown"].round(4).to_list(),
                    fill="tozeroy",
                    name="drawdown")  # fmt: skip
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
            y=rows["ic"].cum_sum().round(4).to_list(),
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


def growth_figure(
    growth: pl.DataFrame,
    names: dict[str, str],
    visible: set[str],
    groups: dict[str, str] | None = None,
    reference: str = "SPY",
) -> str:
    """Growth of $1 per series (key); ``visible`` ones are drawn, the rest start hidden.

    The legend lists the drawn series first, then the others under their group title
    (``groups`` maps key -> group title, in the order the groups should appear).
    """
    groups = groups or {}
    order = list(dict.fromkeys(groups.values()))

    def rank(key: str) -> tuple:
        group = groups.get(key, "")
        return (key not in visible, order.index(group) if group in order else len(order),
                names.get(key, key))  # fmt: skip

    fig = go.Figure(layout=_LAYOUT)
    by_key = {k: g for (k,), g in growth.sort("date").group_by("key", maintain_order=True)}
    for key in sorted(by_key, key=rank):
        rows, shown = by_key[key], key in visible
        fig.add_scatter(
            x=rows["date"].to_list(), y=rows["growth"].round(4).to_list(), name=names.get(key, key),
            line=_REFERENCE if key == reference else _LINE,
            visible=True if shown else "legendonly",
            legendgroup="shown" if shown else groups.get(key, "other"),
            legendgrouptitle_text="Drawn" if shown else groups.get(key, "Other"),
        )  # fmt: skip
    fig.update_layout(
        yaxis_type="log",
        legend={**_LAYOUT["legend"], "groupclick": "toggleitem"},
        xaxis={"rangeselector": RANGE_BUTTONS},
    )
    return fig.to_json()


@functools.cache
def _white_template() -> dict:
    """The plotly_white theme as plain JSON (built once; it is the slow part of a figure)."""
    return go.layout.Template(pio.templates["plotly_white"]).to_plotly_json()


def series_figure(
    series: list[Series], visible: set[str], groups: dict[str, str] | None = None
) -> str | None:
    """Growth of $1 for ``web.series.Series`` objects: weekly points, log scale, zoom buttons.

    Same look as :func:`growth_figure`, but written as plain JSON: building a plotly Figure
    validates every point, which took about a second for the Compare page.
    """
    if not series:
        return None
    groups = groups or {}
    order = list(dict.fromkeys(groups.values()))

    def rank(s: Series) -> tuple:
        group = groups.get(s.key, "")
        return (s.key not in visible, order.index(group) if group in order else len(order),
                s.name)  # fmt: skip

    traces = []
    for s in sorted(series, key=rank):
        points = weekly(s.daily)
        shown = s.key in visible
        traces.append({
            "type": "scatter", "mode": "lines", "name": s.name,
            "x": [d.isoformat() for d in points["date"].to_list()],
            "y": points["growth"].round(4).to_list(),
            "line": _REFERENCE if s.key == "SPY" else _LINE,
            "visible": True if shown else "legendonly",
            "legendgroup": "shown" if shown else groups.get(s.key, "other"),
            "legendgrouptitle": {"text": "Drawn" if shown else groups.get(s.key, "Other")},
        })  # fmt: skip
    layout = {**_LAYOUT, "yaxis": {"type": "log"}, "xaxis": {"rangeselector": RANGE_BUTTONS},
              "legend": {**_LAYOUT["legend"], "groupclick": "toggleitem"}}  # fmt: skip
    layout["template"] = _white_template()
    return json.dumps({"data": traces, "layout": layout})


def tax_layout() -> str:
    """Layout for the Taxes page, whose script draws the lines: dollars on a log scale."""
    yaxis = {"type": "log", "tickprefix": "$", "tickformat": "~s", "dtick": 1}  # $100k, $1M, ...
    return json.dumps({**_LAYOUT, "yaxis": yaxis, "template": _white_template()})
