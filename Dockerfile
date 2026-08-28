FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/axiom

RUN groupadd --system axiom \
    && useradd --system --gid axiom --home-dir /opt/axiom axiom

COPY requirements.lock pyproject.toml README.md ./
RUN pip install --require-hashes -r requirements.lock

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY docs/DAY23_REPRODUCIBLE_OPERATIONS_SPECIFICATION.md ./docs/DAY23_REPRODUCIBLE_OPERATIONS_SPECIFICATION.md
COPY docs/DAY23_OPERATIONS_RUNBOOK.md ./docs/DAY23_OPERATIONS_RUNBOOK.md
COPY Dockerfile .dockerignore compose.yaml ./
COPY .github/workflows/ci.yml ./.github/workflows/ci.yml

RUN pip install --no-deps --no-build-isolation . \
    && mkdir -p /opt/axiom/runtime \
    && chown -R axiom:axiom /opt/axiom/runtime

USER axiom

HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD ["python", "scripts/run_day23_operational_validation.py", "--check-only", "--probe-parent", "/tmp"]

CMD ["python", "scripts/run_day23_operational_validation.py", "--check-only"]

