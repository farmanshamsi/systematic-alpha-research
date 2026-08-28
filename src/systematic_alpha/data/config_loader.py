"""Project configuration and credential loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import yaml


class ProjectConfigError(RuntimeError):
    """Raised when project configuration or credentials are invalid."""


@dataclass(frozen=True)
class AlpacaCredentials:
    """Alpaca credentials whose values are suppressed from representations."""

    api_key: str = dataclass_field(repr=False)
    secret_key: str = dataclass_field(repr=False)


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by locating pyproject.toml."""

    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise ProjectConfigError(
        "Could not locate project root containing pyproject.toml."
    )


def load_project_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the project YAML and fail closed on research-safety settings."""

    root = find_project_root()

    path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else root / "config" / "base.yaml"
    )

    if not path.exists():
        raise ProjectConfigError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ProjectConfigError("Configuration must be a YAML mapping.")

    required_sections = {
        "project",
        "safety",
        "broker",
        "market",
        "instruments",
        "data",
        "sample",
        "strategy_design",
        "execution",
        "risk",
        "validation",
    }

    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ProjectConfigError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    for section in sorted(required_sections):
        if not isinstance(config[section], dict):
            raise ProjectConfigError(
                f"Configuration section must be a mapping: {section!r}."
            )

    broker = config["broker"]
    required_broker_keys = {
        "provider",
        "paper",
        "paper_base_url",
        "credentials_file",
        "api_key_env",
        "secret_key_env",
    }

    missing_broker_keys = required_broker_keys.difference(broker)
    if missing_broker_keys:
        raise ProjectConfigError(
            f"Missing broker keys: {sorted(missing_broker_keys)}"
        )

    data = config["data"]
    required_data_keys = {
        "raw_storage_dir",
        "processed_storage_dir",
        "metadata_storage_dir",
        "storage_format",
        "allow_raw_overwrite",
    }

    missing_data_keys = required_data_keys.difference(data)
    if missing_data_keys:
        raise ProjectConfigError(
            f"Missing data keys: {sorted(missing_data_keys)}"
        )

    safety = config["safety"]
    sample = config["sample"]
    execution = config["execution"]
    project = config["project"]

    if project.get("environment") != "paper":
        raise ProjectConfigError(
            "Project environment must remain 'paper'."
        )

    if safety.get("allow_live_trading") is not False:
        raise ProjectConfigError(
            "Live trading must remain disabled during project development."
        )

    if safety.get("require_paper_mode") is not True:
        raise ProjectConfigError(
            "Paper-mode enforcement must remain enabled."
        )

    if broker["paper"] is not True:
        raise ProjectConfigError("Broker must be configured for paper mode.")

    if str(broker["paper_base_url"]).rstrip("/") != (
        "https://paper-api.alpaca.markets"
    ):
        raise ProjectConfigError(
            "Broker base URL must be the Alpaca paper endpoint."
        )

    if sample.get("final_test_locked") is not True:
        raise ProjectConfigError(
            "The final-test interval must remain locked."
        )

    if execution.get("order_submission_enabled") is not False:
        raise ProjectConfigError(
            "Order submission must remain disabled by default."
        )

    return config


def load_alpaca_credentials(
    config: dict[str, Any] | None = None,
) -> AlpacaCredentials:
    """Load credentials from process environment or the local .env file."""

    project_config = config or load_project_config()
    root = find_project_root()

    try:
        from decouple import Config, RepositoryEnv
    except ImportError as exc:
        raise ProjectConfigError(
            "python-decouple is required only when Alpaca credentials are loaded."
        ) from exc

    broker = project_config["broker"]
    api_key_name = broker["api_key_env"]
    secret_key_name = broker["secret_key_env"]
    process_api_key = os.environ.get(api_key_name)
    process_secret_key = os.environ.get(secret_key_name)

    if process_api_key is not None or process_secret_key is not None:
        if (
            process_api_key is None
            or process_secret_key is None
            or not process_api_key.strip()
            or not process_secret_key.strip()
        ):
            raise ProjectConfigError(
                "Required Alpaca process-environment credentials are incomplete."
            )
        return AlpacaCredentials(
            api_key=process_api_key.strip(),
            secret_key=process_secret_key.strip(),
        )

    env_path = root / broker["credentials_file"]
    if not env_path.exists():
        raise ProjectConfigError("Required local credentials file is missing.")
    env = Config(RepositoryEnv(str(env_path)))

    try:
        api_key = env(api_key_name)
        secret_key = env(secret_key_name)
    except Exception as exc:
        raise ProjectConfigError(
            "Required Alpaca credentials are missing from the local environment."
        ) from exc

    if not api_key.strip() or not secret_key.strip():
        raise ProjectConfigError("Alpaca credentials cannot be empty.")

    return AlpacaCredentials(
        api_key=api_key.strip(),
        secret_key=secret_key.strip(),
    )


def resolve_data_directories(
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Resolve configured data directories relative to the repository."""

    project_config = config or load_project_config()
    root = find_project_root()
    data = project_config["data"]

    return {
        "raw": root / data["raw_storage_dir"],
        "processed": root / data["processed_storage_dir"],
        "metadata": root / data["metadata_storage_dir"],
    }
