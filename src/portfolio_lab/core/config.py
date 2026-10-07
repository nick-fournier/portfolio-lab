"""Application settings, read from the environment and the repo's ``.env`` file."""

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: First session of Alpaca's historical SIP data; backfills start here.
HISTORY_START = date(2016, 1, 4)


class Settings(BaseSettings):
    """Runtime configuration.

    Values come from environment variables (case-insensitive) or a ``.env`` file in the
    working directory. Secrets are ``SecretStr`` so they never appear in logs or reprs.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("data"), alias="PORTFOLIO_DATA_DIR")
    #: Trading records (paper account); default ``data_dir/trading``.
    trading_dir: Path | None = Field(default=None, alias="PORTFOLIO_TRADING_DIR")
    #: Where the fetchers (Alpaca, NASDAQ, Tiingo, FRED, EDGAR) keep what they download,
    #: in their own layout, for the conformers to read; default ``data_dir/ingest``.
    ingest_dir: Path | None = Field(default=None, alias="PORTFOLIO_INGEST_DIR")
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None
    alpaca_data_url: str = "https://data.alpaca.markets"
    #: Trading API; only Alpaca's paper (dummy money) endpoint is accepted for now.
    alpaca_paper_url: str = "https://paper-api.alpaca.markets"
    alpaca_requests_per_minute: int = 180
    edgar_user_agent: str = "portfolio-lab nichfournier@gmail.com"
    fred_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    log_level: str = "INFO"

    def for_ingest(self) -> "Settings":
        """These settings with ``data_dir`` pointed at the fetchers' store."""
        return self.model_copy(update={"data_dir": self.ingest_dir or self.data_dir / "ingest"})

    def alpaca_headers(self) -> dict[str, str]:
        """Return the Alpaca authentication headers.

        Raises:
            RuntimeError: If either Alpaca key is missing.
        """
        if not (self.alpaca_api_key_id and self.alpaca_api_secret_key):
            raise RuntimeError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set")
        return {
            "APCA-API-KEY-ID": self.alpaca_api_key_id.get_secret_value(),
            "APCA-API-SECRET-KEY": self.alpaca_api_secret_key.get_secret_value(),
        }


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
