"""Deterministic, offline Day 23 operational health validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module, metadata
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tomllib
from typing import Final, Mapping

import yaml

from systematic_alpha.data.config_loader import find_project_root


DAY23_SCHEMA_VERSION: Final[str] = "day23_reproducible_operations_v1"
REFERENCE_PYTHON_VERSION: Final[str] = "3.11.15"
PAPER_BASE_URL: Final[str] = "https://paper-api.alpaca.markets"
LOCK_FILENAME: Final[str] = "requirements.lock"
BASE_CONFIG_FILENAME: Final[str] = "config/base.yaml"
OPERATIONS_CONFIG_FILENAME: Final[str] = "config/operations.yaml"
EXPECTED_JOB_IDS: Final[tuple[str, ...]] = (
    "health-smoke",
    "day22-campaign-once",
    "day21-strategy-once",
)
HEALTH_CHECK_IDS: Final[tuple[str, ...]] = (
    "supported_python_version",
    "required_package_imports",
    "project_identity_and_paper_environment",
    "real_money_disabled",
    "exact_paper_endpoint",
    "order_submission_disabled_by_default",
    "manual_confirmation_and_kill_switch",
    "locked_final_test_boundary",
    "timezone_contract",
    "dependency_lock_integrity",
    "scheduled_entrypoint_integrity",
    "required_repository_files",
    "temporary_persistence_round_trip",
    "clean_shutdown_without_external_access",
)
REQUIRED_REPOSITORY_FILES: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "requirements.lock",
    "config/base.yaml",
    "config/operations.yaml",
    "Dockerfile",
    ".dockerignore",
    "compose.yaml",
    ".github/workflows/ci.yml",
    "docs/DAY23_REPRODUCIBLE_OPERATIONS_SPECIFICATION.md",
    "docs/DAY23_OPERATIONS_RUNBOOK.md",
    "scripts/run_day23_operational_validation.py",
    "scripts/run_scheduled_job.py",
)
IMPORT_NAMES: Final[Mapping[str, str]] = {
    "alpaca-py": "alpaca",
    "arch": "arch",
    "cvxpy": "cvxpy",
    "exchange-calendars": "exchange_calendars",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "python-decouple": "decouple",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "yfinance": "yfinance",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
}
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^([A-Za-z0-9_.-]+)")
_LOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)"
)


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str
    hash_count: int
    exact: bool


@dataclass(frozen=True)
class DependencyAudit:
    dependency_order: int
    dependency_name: str
    pyproject_specifier: str
    locked_version: str
    hash_count: int
    lock_exact: bool
    installed_version: str
    installed_matches_lock: bool

    def row(self) -> dict[str, object]:
        return {
            "dependency_order": self.dependency_order,
            "dependency_name": self.dependency_name,
            "pyproject_specifier": self.pyproject_specifier,
            "locked_version": self.locked_version,
            "hash_count": self.hash_count,
            "lock_exact": self.lock_exact,
            "installed_version": self.installed_version,
            "installed_matches_lock": self.installed_matches_lock,
        }


@dataclass(frozen=True)
class HealthCheck:
    check_order: int
    check_id: str
    passed: bool
    detail: str

    def row(self) -> dict[str, object]:
        return {
            "check_order": self.check_order,
            "check_id": self.check_id,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScheduleEntry:
    job_order: int
    job_id: str
    entrypoint: str
    authorization_flag: str
    schedule_policy: str
    automatic: bool
    order_capable: bool

    def row(self) -> dict[str, object]:
        return {
            "job_order": self.job_order,
            "job_id": self.job_id,
            "entrypoint": self.entrypoint,
            "authorization_flag": self.authorization_flag,
            "schedule_policy": self.schedule_policy,
            "automatic": self.automatic,
            "order_capable": self.order_capable,
        }


@dataclass(frozen=True)
class Day23ValidationResult:
    dependency_audit: tuple[DependencyAudit, ...]
    health_checks: tuple[HealthCheck, ...]
    schedule_entries: tuple[ScheduleEntry, ...]
    lock_sha256: str
    python_version: str
    clean_environment_validated: bool
    container_runtime_available: bool
    container_static_validation_passed: bool
    broker_network_accessed: bool = False
    credentials_accessed: bool = False
    orders_submitted: bool = False
    locked_final_test_data_accessed: bool = False

    @property
    def evaluation_complete(self) -> bool:
        return (
            tuple(check.check_id for check in self.health_checks)
            == HEALTH_CHECK_IDS
            and all(check.passed for check in self.health_checks)
            and all(
                item.lock_exact and item.hash_count > 0
                for item in self.dependency_audit
            )
            and not self.broker_network_accessed
            and not self.credentials_accessed
            and not self.orders_submitted
            and not self.locked_final_test_data_accessed
        )


def parse_dependency_lock(path: Path) -> dict[str, LockedRequirement]:
    """Parse exact pip-compile requirement blocks without executing pip."""

    text = path.read_text("utf-8")
    lines = text.splitlines()
    requirements: dict[str, LockedRequirement] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line and not line[0].isspace() and not line.startswith("#"):
            match = _LOCK_PATTERN.match(line)
            if match is None:
                raise ValueError(f"Non-exact requirement line in lock: {line}")
            name, version = match.groups()
            block = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if (
                    next_line
                    and not next_line[0].isspace()
                    and not next_line.startswith("#")
                ):
                    break
                block.append(next_line)
                index += 1
            normalized = _normalize_name(name)
            if normalized in requirements:
                raise ValueError(f"Duplicate locked requirement: {normalized}")
            requirements[normalized] = LockedRequirement(
                name=normalized,
                version=version,
                hash_count=sum(
                    item.strip().startswith("--hash=sha256:") for item in block
                ),
                exact=True,
            )
            continue
        index += 1
    if not requirements:
        raise ValueError("Dependency lock contains no requirements.")
    return requirements


def _direct_dependencies(project_root: Path) -> tuple[tuple[str, str], ...]:
    document = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))
    project = document["project"]
    values = tuple(project["dependencies"]) + tuple(
        project["optional-dependencies"]["dev"]
    )
    dependencies: list[tuple[str, str]] = []
    for value in values:
        match = _NAME_PATTERN.match(value)
        if match is None:
            raise ValueError(f"Invalid direct dependency: {value}")
        dependencies.append((_normalize_name(match.group(1)), value))
    return tuple(dependencies)


def _installed_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def build_dependency_audit(
    project_root: Path, locked: Mapping[str, LockedRequirement]
) -> tuple[DependencyAudit, ...]:
    rows: list[DependencyAudit] = []
    for order, (name, specifier) in enumerate(
        _direct_dependencies(project_root), start=1
    ):
        item = locked.get(name)
        installed = _installed_version(name)
        rows.append(
            DependencyAudit(
                dependency_order=order,
                dependency_name=name,
                pyproject_specifier=specifier,
                locked_version=item.version if item else "missing",
                hash_count=item.hash_count if item else 0,
                lock_exact=bool(item and item.exact),
                installed_version=installed,
                installed_matches_lock=bool(item and installed == item.version),
            )
        )
    return tuple(rows)


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _schedule_entries(operations: Mapping[str, object]) -> tuple[ScheduleEntry, ...]:
    raw_jobs = operations.get("jobs")
    if not isinstance(raw_jobs, list):
        return ()
    rows: list[ScheduleEntry] = []
    for order, raw in enumerate(raw_jobs, start=1):
        if not isinstance(raw, dict):
            return ()
        rows.append(
            ScheduleEntry(
                job_order=order,
                job_id=str(raw.get("job_id", "")),
                entrypoint=str(raw.get("entrypoint", "")),
                authorization_flag=(
                    ""
                    if raw.get("authorization_flag") is None
                    else str(raw["authorization_flag"])
                ),
                schedule_policy=str(raw.get("schedule_policy", "")),
                automatic=raw.get("automatic") is True,
                order_capable=raw.get("order_capable") is True,
            )
        )
    return tuple(rows)


def _persistence_probe(parent: Path | None) -> tuple[bool, str]:
    parent_path = parent.resolve() if parent is not None else None
    if parent_path is not None:
        parent_path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="axiom-day23-probe-", dir=parent_path
        ) as directory:
            probe = Path(directory) / "probe.bin"
            payload = b"axiom-day23-persistence-probe-v1\n"
            expected = hashlib.sha256(payload).hexdigest()
            probe.write_bytes(payload)
            actual = hashlib.sha256(probe.read_bytes()).hexdigest()
            probe.unlink()
            if actual != expected or probe.exists():
                return False, "write/read/hash/remove probe failed"
        return True, "temporary write/read/hash/remove probe passed"
    except OSError as exc:
        return False, f"persistence probe failed: {type(exc).__name__}"


def _import_check() -> tuple[bool, str]:
    failed: list[str] = []
    for package, module in IMPORT_NAMES.items():
        try:
            import_module(module)
        except Exception:
            failed.append(package)
    if failed:
        return False, "failed imports: " + ",".join(failed)
    return True, f"{len(IMPORT_NAMES)} required imports passed"


def container_runtime_available() -> bool:
    """Return whether Docker or Podman is installed; do not start either."""

    return shutil.which("docker") is not None or shutil.which("podman") is not None


def validate_container_contract(project_root: str | Path) -> bool:
    """Statically verify the frozen container and CI safety contract."""

    root = Path(project_root).resolve()
    try:
        dockerfile = (root / "Dockerfile").read_text("utf-8")
        dockerignore = (root / ".dockerignore").read_text("utf-8").splitlines()
        compose = _load_yaml_mapping(root / "compose.yaml")
        workflow = (root / ".github/workflows/ci.yml").read_text("utf-8")
        services = compose["services"]
        smoke = services["axiom-smoke"]
        ignored = {line.strip() for line in dockerignore if line.strip()}
    except (OSError, KeyError, TypeError, ValueError):
        return False
    install_position = dockerfile.find(
        "pip install --require-hashes -r requirements.lock"
    )
    source_position = dockerfile.find("COPY src ./src")
    return (
        dockerfile.startswith("FROM python:3.11.15-slim-bookworm\n")
        and 0 <= install_position < source_position
        and "USER axiom" in dockerfile
        and "HEALTHCHECK" in dockerfile
        and 'CMD ["python", "scripts/run_day23_operational_validation.py", "--check-only"]' in dockerfile
        and {
            ".env",
            ".git",
            "data/raw",
            "data/processed",
            "outputs",
            "artifacts",
            "logs",
            "backups",
        }.issubset(ignored)
        and isinstance(services, dict)
        and isinstance(smoke, dict)
        and smoke.get("environment") is None
        and smoke.get("secrets") is None
        and smoke.get("ports") is None
        and smoke.get("command")
        == [
            "python",
            "scripts/run_day23_operational_validation.py",
            "--check-only",
            "--probe-parent",
            "/tmp",
        ]
        and "requirements.lock" in workflow
        and "pip install --require-hashes -r requirements.lock" in workflow
        and "pip check" in workflow
        and "run_day23_operational_validation.py --check-only" in workflow
        and "pytest" in workflow
        and "ALPACA_API_KEY" not in workflow
        and "ALPACA_SECRET_KEY" not in workflow
    )


def run_operational_validation(
    project_root: str | Path | None = None,
    *,
    probe_parent: str | Path | None = None,
    clean_environment_validated: bool = False,
    container_runtime_is_available: bool | None = None,
    container_static_validation_passed: bool | None = None,
) -> Day23ValidationResult:
    """Run all 14 frozen checks without credentials, network, data, or orders."""

    if project_root is None:
        root = find_project_root(Path(__file__).resolve())
    else:
        root = Path(project_root).resolve()
    lock_path = root / LOCK_FILENAME
    locked = parse_dependency_lock(lock_path)
    dependency_audit = build_dependency_audit(root, locked)
    base = _load_yaml_mapping(root / BASE_CONFIG_FILENAME)
    operations = _load_yaml_mapping(root / OPERATIONS_CONFIG_FILENAME)
    schedules = _schedule_entries(operations)
    import_passed, import_detail = _import_check()
    persistence_passed, persistence_detail = _persistence_probe(
        Path(probe_parent) if probe_parent is not None else None
    )

    project = base.get("project", {})
    safety = base.get("safety", {})
    broker = base.get("broker", {})
    market = base.get("market", {})
    sample = base.get("sample", {})
    execution = base.get("execution", {})
    runtime = operations.get("runtime", {})
    if not all(
        isinstance(value, dict)
        for value in (project, safety, broker, market, sample, execution, runtime)
    ):
        raise ValueError("Operational configuration sections must be mappings.")

    lock_integrity = (
        bool(dependency_audit)
        and all(row.lock_exact and row.hash_count > 0 for row in dependency_audit)
        and locked.get("alpaca-py") is not None
        and locked["alpaca-py"].version == "0.43.5"
        and locked.get("exchange-calendars") is not None
        and locked["exchange-calendars"].version == "4.13.2"
        and locked.get("pytest") is not None
        and locked.get("pytest-cov") is not None
    )
    schedule_integrity = (
        operations.get("schema_version") == "day23_operations_v1"
        and runtime
        == {
            "default_mode": "smoke",
            "broker_environment": "paper",
            "scheduler_timezone": "UTC",
            "exchange_timezone": "America/New_York",
            "credentials_required_in_smoke": False,
            "network_allowed_in_smoke": False,
            "order_submission_allowed_in_smoke": False,
        }
        and tuple(row.job_id for row in schedules) == EXPECTED_JOB_IDS
        and tuple(row.entrypoint for row in schedules)
        == (
            "scripts/run_day23_operational_validation.py",
            "scripts/run_day22_live_campaign.py",
            "scripts/run_day21_controlled_paper_execution.py",
        )
        and tuple(row.authorization_flag for row in schedules)
        == ("", "--authorized-paper-campaign", "--authorized-paper-order")
        and tuple(row.schedule_policy for row in schedules)
        == (
            "external_utc_scheduler",
            "frozen_day22_slots_only",
            "manual_only",
        )
        and tuple(row.automatic for row in schedules) == (True, True, False)
        and tuple(row.order_capable for row in schedules) == (False, True, True)
    )
    required_files_present = all(
        (root / relative).is_file() for relative in REQUIRED_REPOSITORY_FILES
    )
    python_version = ".".join(str(value) for value in sys.version_info[:3])
    conditions: tuple[tuple[bool, str], ...] = (
        (
            sys.version_info[:2] in ((3, 11), (3, 12)),
            f"runtime Python {python_version}; reference {REFERENCE_PYTHON_VERSION}",
        ),
        (import_passed, import_detail),
        (
            project.get("name") == "systematic-alpha-research"
            and project.get("environment") == "paper"
            and broker.get("paper") is True,
            "project identity and paper environment verified",
        ),
        (
            safety.get("allow_live_trading") is False,
            "allow_live_trading is false",
        ),
        (
            broker.get("paper_base_url") == PAPER_BASE_URL,
            "exact Alpaca paper endpoint verified",
        ),
        (
            execution.get("order_submission_enabled") is False,
            "default order submission is disabled",
        ),
        (
            safety.get("require_paper_mode") is True
            and safety.get("require_manual_order_confirmation") is True
            and safety.get("kill_switch_enabled") is True,
            "paper, manual confirmation, and kill switch gates enabled",
        ),
        (
            sample.get("final_test_locked") is True
            and sample.get("final_test_start") == "2026-01-02"
            and sample.get("final_test_end") == "2026-06-30",
            "locked final-test boundary is unchanged",
        ),
        (
            market.get("exchange_timezone") == "America/New_York"
            and market.get("storage_timezone") == "UTC"
            and runtime.get("scheduler_timezone") == "UTC",
            "New York exchange and UTC storage/scheduler verified",
        ),
        (lock_integrity, f"exact hashed lock contains {len(locked)} packages"),
        (schedule_integrity, "three exact fail-closed jobs verified"),
        (
            required_files_present,
            f"{len(REQUIRED_REPOSITORY_FILES)} required repository files present",
        ),
        (persistence_passed, persistence_detail),
        (
            runtime.get("credentials_required_in_smoke") is False
            and runtime.get("network_allowed_in_smoke") is False
            and runtime.get("order_submission_allowed_in_smoke") is False,
            "clean shutdown with zero credential/network/order access",
        ),
    )
    checks = tuple(
        HealthCheck(order, check_id, passed, detail)
        for order, (check_id, (passed, detail)) in enumerate(
            zip(HEALTH_CHECK_IDS, conditions, strict=True), start=1
        )
    )
    return Day23ValidationResult(
        dependency_audit=dependency_audit,
        health_checks=checks,
        schedule_entries=schedules,
        lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        python_version=python_version,
        clean_environment_validated=clean_environment_validated,
        container_runtime_available=(
            container_runtime_available()
            if container_runtime_is_available is None
            else container_runtime_is_available
        ),
        container_static_validation_passed=(
            validate_container_contract(root)
            if container_static_validation_passed is None
            else container_static_validation_passed
        ),
    )
