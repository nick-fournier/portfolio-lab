"""Class 3 optimizer: the commands' checks and months, the months cache's key, old settings."""

import json
import re
import shlex
from dataclasses import fields
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from typer.testing import CliRunner

from portfolio_lab.research.optimizer import cli
from portfolio_lab.research.optimizer.universe import NO_FLOORS

#: The head run's settings.json, as the walk wrote it (cubevm: class3-runs/ret/run/seed-0).
HEAD_SETTINGS = Path(__file__).resolve().parents[1] / "fixtures" / "class3_head_settings.json"
#: The head run's command line (its start and end aside).
HEAD = shlex.split(
    "--model transformer --blocks 10 --schedule cosine --lr 3e-4 --lr-min 1e-5 "
    "--fast-attention --compile --precision high --clip 40 --average 0.998 --inputs paper "
    "--first-epochs 40 --window 120 --cold --watch --winsor 0.01 --holdout 12 --scatter "
    "--stock-drop 0.2 --no-blank-checks --costs class3-runs/costs6.parquet --holdings-input "
    "--borrow-input --profile alpaca --trade-margin --start-book market"
)


def _text(result) -> str:
    """A command's output with rich's colors, box and line breaks out."""
    plain = re.sub(r"[│╭╮╰╯─]", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    return " ".join(plain.split())


def _walked(tmp_path, monkeypatch, *flags: str) -> tuple:
    """What ``plab optimizer walk`` hands its one seed's worker, with no months built or run.

    (table, run, settings, device, threads, precision, inputs, last, floors), as ``_walk_one``
    takes them.
    """
    calls = []

    class Pool:  # the seed workers' process pool, taking the work without doing it
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args):
            calls.append(args)
            return SimpleNamespace(result=lambda: "walked")

    monkeypatch.setattr(cli, "_months", lambda *args: [])
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", Pool)
    result = CliRunner().invoke(cli.optimizer_app, [
        "walk", "--table", str(tmp_path / "t.parquet"), "--out", str(tmp_path / "w"), *flags,
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    (args,) = calls
    return args


def test_months_cache_key_changes_with_the_inputs_floors_and_drop_list(tmp_path):
    table = tmp_path / "table.parquet"
    shells = frozenset({"101", "202"})
    base = cli.months_cache(table, "paper", (0.0, 0.0, True), shells)
    assert base.parent == tmp_path and base.name.startswith("table.months-paper-p0-v0-nb-")
    assert cli.months_cache(table, "paper", (0, 0, True), shells) == base  # same floors
    assert cli.months_cache(table, "paper", (0.0, 0.0, True), frozenset({"202", "101"})) == base
    others = [
        cli.months_cache(table, "all", (0.0, 0.0, True), shells),  # input set
        cli.months_cache(table, "paper", (5.0, 0.0, True), shells),  # price floor
        cli.months_cache(table, "paper", (0.0, 1e6, True), shells),  # volume floor
        cli.months_cache(table, "paper", (0.0, 0.0, True), shells | {"303"}),  # drop list
        cli.months_cache(table, "paper", NO_FLOORS, frozenset()),  # no floors
    ]
    assert len({base, *others}) == 1 + len(others)
    assert not cli.months_cache(table, "paper", NO_FLOORS, frozenset()).name.startswith(
        "table.months-paper-paper"
    )  # fmt: skip


def test_walk_refuses_loss_options_without_costs(tmp_path):
    base = ["walk", "--table", str(tmp_path / "t.parquet"), "--out", str(tmp_path / "w"),
            "--model", "linear"]  # fmt: skip
    for extra in (
        ["--trade-margin"],
        ["--profile", "alpaca"],
        ["--start-book", "market"],
        ["--holdings-input"],
        ["--gross-limit", "1.25"],
        ["--cash-rate", "0.03"],
        ["--trade-gross", "1"],
    ):
        result = CliRunner().invoke(cli.optimizer_app, base + extra)
        assert result.exit_code == 2, result.output
        text = _text(result)
        assert f"--costs (the per-stock cost table) is needed for {extra[0]}" in text, text
    assert not (tmp_path / "w").exists()  # refused before any work


def test_a_realism_flag_given_beats_the_profile_which_fills_the_rest(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import realism  # noqa: PLC0415

    alpaca = ["--model", "linear", "--costs", "c.parquet", "--profile", "alpaca"]
    for flags, want in (
        (["--gross-limit", "0"], {"gross_limit": 0.0}),  # an explicit 0 is not "not given"
        (["--margin-rate", "0.08"], {"margin_rate": 0.08}),
        (["--cash-rate", "-1"], {"cash_rate": None}),  # the T-bill, given
        (["--no-rebate"], {"rebate": False}),
    ):
        s = _walked(tmp_path, monkeypatch, *alpaca, *flags)[2]
        assert {k: getattr(s, k) for k in realism.ALPACA} == realism.ALPACA | want, flags


def test_no_profile_leaves_the_settings_defaults(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    for extra in ([], ["--profile", "none"]):
        settings = _walked(tmp_path, monkeypatch, "--model", "linear", *extra)[2]
        assert settings == walk.Settings("linear"), extra


def test_the_trade_size_flags_reach_the_settings(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    s = _walked(tmp_path, monkeypatch, "--model", "linear", "--costs", "c.parquet",
                "--trade-gross", "1", "--trade-collateral", "0.4")[2]  # fmt: skip
    assert (s.trade_gross, s.trade_collateral) == (1.0, 0.4)
    priced = _walked(tmp_path, monkeypatch, "--model", "linear", "--costs", "c.parquet",
                     "--price-gross", "1")[2]  # fmt: skip
    assert priced.price_gross == 1.0
    sized = _walked(tmp_path, monkeypatch, "--model", "linear", "--costs", "c.parquet",
                    "--trade-gross", "1", "--loss-gross", "1",
                    "--risk-aversion", "20", "--size-anchor", "0.1")[2]  # fmt: skip
    assert (sized.loss_gross, sized.risk_aversion, sized.size_anchor) == (1.0, 20.0, 0.1)
    plain = walk.Settings("linear", costs="c.parquet")
    assert _walked(tmp_path, monkeypatch, "--model", "linear", "--costs", "c.parquet")[2] == plain
    assert (plain.trade_gross, plain.trade_collateral, plain.price_gross) == (0.0, 0.5, 0.0)


def test_an_unknown_profile_is_refused(tmp_path):
    pytest.importorskip("torch")
    result = CliRunner().invoke(cli.optimizer_app, [
        "walk", "--table", str(tmp_path / "t.parquet"), "--out", str(tmp_path / "w"),
        "--model", "linear", "--costs", "c.parquet", "--profile", "robinhood",
    ])  # fmt: skip
    assert result.exit_code == 2, result.output
    assert "profile must be none or one of ['alpaca'], not 'robinhood'" in _text(result)


def test_the_head_command_line_rebuilds_the_head_runs_settings(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    meta = json.loads(HEAD_SETTINGS.read_text())
    args = _walked(tmp_path, monkeypatch, *HEAD)
    got, want = args[2], walk.settings_from(meta)
    differ = {f.name for f in fields(want) if getattr(got, f.name) != getattr(want, f.name)}
    assert differ == {"start", "costs"}  # both from the command, not the recipe
    assert (got.start, got.costs) == (date(2003, 12, 31), "class3-runs/costs6.parquet")
    assert (args[5], args[6], args[8]) == ("high", meta["inputs"], tuple(meta["floors"]))


def test_ridge_and_signal_build_their_months_like_a_walk(tmp_path, monkeypatch):
    calls = []

    def months(*args):
        calls.append(args)
        raise RuntimeError("stop here")  # only the call matters

    monkeypatch.setattr(cli, "_months", months)
    table = tmp_path / "t.parquet"
    forecasts = tmp_path / "f.parquet"
    pl.DataFrame({"date": [date(2020, 1, 31)], "symbol": ["A"], "forecast": [0.1]}).write_parquet(
        forecasts
    )  # fmt: skip
    for command in (["ridge"], ["signal", "--forecasts", str(forecasts)]):
        result = CliRunner().invoke(cli.optimizer_app, [
            *command, "--table", str(table), "--inputs", "jkp", "--min-price", "5",
            "--no-blank-checks",
        ])  # fmt: skip
        assert isinstance(result.exception, RuntimeError), result.output
    assert calls == [(table, "jkp", (5.0, 0.0, True))] * 2


def test_old_settings_load_when_their_removed_knobs_were_off():
    pytest.importorskip("torch")
    from portfolio_lab.research.optimizer import walk  # noqa: PLC0415

    old = {"model": "transformer", "blocks": 10, "start": "2003-12-31", "clip": 40.0,
           "only": ["2024-07-31"], "stop": False, "patience": 8, "size_bound": False,
           "gross_cap": 2.0, "cost_scale": 1.0, "measure_optimism": True, "vol_column": 3,
           "inputs": "paper", "end": None, "floors": [0.0, 0.0, True]}  # fmt: skip
    s = walk.settings_from(old)
    assert (s.model, s.blocks, s.clip) == ("transformer", 10, 40.0)
    assert s.start == date(2003, 12, 31) and s.only == (date(2024, 7, 31),)
    for knob in ({"size_bound": True}, {"stop": True}, {"margin_spread": 0.01}):
        with pytest.raises(ValueError, match=f"{next(iter(knob))}=.* was removed"):
            walk.settings_from(old | knob)
    with pytest.raises(ValueError, match="not a walk setting"):
        walk.settings_from(old | {"typo": 1})


def test_the_drop_list_is_the_tables_blank_checks_and_needs_the_hive(tmp_path, monkeypatch):
    from portfolio_lab.research.optimizer import universe  # noqa: PLC0415

    table = tmp_path / "t.parquet"
    pl.DataFrame({"symbol": ["101", "303", "101"]}).write_parquet(table)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    with pytest.raises(FileNotFoundError, match="set PORTFOLIO_DATA_DIR"):
        cli._shells(table)  # no security master under the data dir
    monkeypatch.setattr(universe, "blank_checks", lambda _: frozenset({"101", "999"}))
    assert cli._shells(table) == frozenset({"101"})  # a shell not in the table: no say
