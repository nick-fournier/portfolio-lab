"""Tests for how saved runs and paper rebalances are named."""

import json

from portfolio_lab.jobs.models import (
    CLASS_1,
    CLASS_1_KELLY,
    CLASS_2,
    CLASS_2_KELLY,
    model_record,
    run_label,
)


def _meta(params: dict, label: str) -> dict:
    return {"strategy": "meanvar", "params": params, "label": label}


def _untitled(model: tuple) -> dict:
    return {k: v for k, v in model[1].items() if k != "title"}


def test_a_run_saved_before_its_title_takes_the_title():
    assert run_label(_meta(_untitled(CLASS_1), "meanvar (...)")) == "Momentum · health · max-Sharpe"
    assert run_label(_meta(_untitled(CLASS_2), "meanvar (...)")) == "Linear · health · max-Sharpe"


def test_runs_saved_while_kelly_was_production_take_the_kelly_titles():
    # saved under the production titles from 2026-10-06 to 2026-10-18
    saved = _meta(_untitled(CLASS_1_KELLY), "Momentum · health · max-Sharpe")
    assert run_label(saved) == "Momentum · health · Kelly"
    assert (
        run_label(_meta(_untitled(CLASS_2_KELLY), "Linear · health · max-Sharpe"))
        == "Linear · health · Kelly x16"
    )


def test_other_runs_keep_their_saved_label():
    assert run_label(_meta({"min_fscore": 7}, "meanvar (min_fscore=7)")) == "meanvar (min_fscore=7)"
    assert run_label(_meta({"retired": 1}, "meanvar (retired=1)")) == "meanvar (retired=1)"


def test_a_paper_rebalance_record_reads_back_as_its_model():
    assert run_label(json.loads(model_record(*CLASS_2))) == "Linear · health · max-Sharpe"
    assert run_label(json.loads(model_record(*CLASS_1_KELLY))) == "Momentum · health · Kelly"
