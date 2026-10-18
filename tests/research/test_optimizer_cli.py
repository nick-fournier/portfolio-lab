"""Class 3 optimizer: the commands' checks and months, the months cache's key, old settings."""

import re
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest
from typer.testing import CliRunner

from portfolio_lab.research.optimizer import cli
from portfolio_lab.research.optimizer.universe import NO_FLOORS


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
    ):
        result = CliRunner().invoke(cli.optimizer_app, base + extra)
        assert result.exit_code == 2, result.output
        plain = re.sub(r"[│╭╮╰╯─]", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
        text = " ".join(plain.split())  # rich's colors, box and line breaks out
        assert f"--costs (the per-stock cost table) is needed for {extra[0]}" in text, text
    assert not (tmp_path / "w").exists()  # refused before any work


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
