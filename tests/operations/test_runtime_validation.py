from pathlib import Path
import shutil

import yaml

from systematic_alpha.operations.runtime_validation import (
    EXPECTED_JOB_IDS,
    HEALTH_CHECK_IDS,
    parse_dependency_lock,
    run_operational_validation,
    validate_container_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_exact_lock_and_offline_health_contract(tmp_path: Path) -> None:
    locked = parse_dependency_lock(PROJECT_ROOT / "requirements.lock")
    assert locked["alpaca-py"].version == "0.43.5"
    assert locked["exchange-calendars"].version == "4.13.2"
    assert locked["pytest"].hash_count > 0
    assert locked["pytest-cov"].hash_count > 0
    assert locked["wheel"].hash_count > 0

    result = run_operational_validation(
        PROJECT_ROOT,
        probe_parent=tmp_path,
        clean_environment_validated=True,
    )
    assert tuple(check.check_id for check in result.health_checks) == HEALTH_CHECK_IDS
    assert all(check.passed for check in result.health_checks)
    assert tuple(item.job_id for item in result.schedule_entries) == EXPECTED_JOB_IDS
    assert result.evaluation_complete is True
    assert result.clean_environment_validated is True
    assert result.container_static_validation_passed is True
    assert result.broker_network_accessed is False
    assert result.credentials_accessed is False
    assert result.orders_submitted is False
    assert result.locked_final_test_data_accessed is False


def test_container_contract_fails_if_default_service_gains_environment(
    tmp_path: Path,
) -> None:
    for relative in (
        "Dockerfile",
        ".dockerignore",
        "compose.yaml",
        "requirements.lock",
    ):
        shutil.copy2(PROJECT_ROOT / relative, tmp_path / relative)
    workflow = tmp_path / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / ".github/workflows/ci.yml", workflow)
    assert validate_container_contract(tmp_path) is True

    compose_path = tmp_path / "compose.yaml"
    compose = yaml.safe_load(compose_path.read_text("utf-8"))
    compose["services"]["axiom-smoke"]["environment"] = {
        "ORDER_SUBMISSION_ENABLED": "true"
    }
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False), "utf-8")
    assert validate_container_contract(tmp_path) is False

