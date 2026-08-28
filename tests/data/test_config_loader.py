from pathlib import Path

import pytest
import yaml

from systematic_alpha.data.config_loader import (
    ProjectConfigError,
    find_project_root,
    load_project_config,
    resolve_data_directories,
)


def test_project_root_contains_pyproject() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").exists()


def test_project_config_loads() -> None:
    config = load_project_config()

    assert config["project"]["name"] == "systematic-alpha-research"
    assert config["broker"]["paper"] is True
    assert config["safety"]["allow_live_trading"] is False
    assert config["sample"]["final_test_locked"] is True
    assert config["execution"]["order_submission_enabled"] is False
    assert (
        config["strategy_design"]["trend_baseline_positioning"]
        == "long_short_neutral"
    )


def test_data_directories_are_inside_project() -> None:
    root = find_project_root()
    directories = resolve_data_directories()

    for path in directories.values():
        assert isinstance(path, Path)
        assert path.is_relative_to(root)


@pytest.mark.parametrize(
    ("section", "key", "unsafe_value", "message"),
    [
        ("project", "environment", "live", "environment must remain 'paper'"),
        ("safety", "allow_live_trading", True, "Live trading must remain disabled"),
        ("safety", "require_paper_mode", False, "Paper-mode enforcement"),
        ("broker", "paper", False, "configured for paper mode"),
        ("broker", "paper_base_url", "https://api.alpaca.markets", "paper endpoint"),
        ("sample", "final_test_locked", False, "must remain locked"),
        ("execution", "order_submission_enabled", True, "disabled by default"),
    ],
)
def test_project_config_rejects_unsafe_research_boundaries(
    tmp_path: Path,
    section: str,
    key: str,
    unsafe_value: object,
    message: str,
) -> None:
    config = load_project_config()
    config[section][key] = unsafe_value
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ProjectConfigError, match=message):
        load_project_config(path)
