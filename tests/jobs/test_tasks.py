"""Tests for how saved runs are named."""

from portfolio_lab.jobs.tasks import CLASS_2, PRODUCTION, run_label


def _meta(params: dict, label: str) -> dict:
    return {"strategy": "meanvar", "params": params, "label": label}


def test_a_run_saved_before_its_title_takes_the_title():
    untitled = {k: v for k, v in PRODUCTION[1].items() if k != "title"}
    assert run_label(_meta(untitled, "meanvar (objective=kelly, ...)")) == "Piotroski health"
    untitled = {k: v for k, v in CLASS_2[1].items() if k != "title"}
    assert run_label(_meta(untitled, "meanvar (...)")) == "Forecaster"


def test_other_runs_keep_their_saved_label():
    assert run_label(_meta({"min_fscore": 7}, "meanvar (min_fscore=7)")) == "meanvar (min_fscore=7)"
    assert run_label(_meta({"retired": 1}, "meanvar (retired=1)")) == "meanvar (retired=1)"
