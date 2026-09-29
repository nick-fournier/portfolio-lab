import pytest

from portfolio_lab.core.config import Settings


@pytest.fixture
def settings(tmp_path):
    """Settings pointed at a temporary data dir, ignoring the real .env (and its keys)."""
    return Settings(_env_file=None, PORTFOLIO_DATA_DIR=tmp_path / "data")
