import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from portfolio_lab.cli import app

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_file_length.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_file_length", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_length_check_passes_and_fails(tmp_path):
    checker = _load_checker()
    short = tmp_path / "short.py"
    short.write_text("x = 1\n" * 10)
    long = tmp_path / "long.py"
    long.write_text("x = 1\n" * 11)

    assert checker.main(["--max", "10", str(short)]) == 0
    assert checker.main(["--max", "10", str(short), str(long)]) == 1


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
