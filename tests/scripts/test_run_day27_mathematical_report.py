"""Focused offline-rendering tests for the Day 27 OU derivation."""

from __future__ import annotations

import pytest

from systematic_alpha.analysis.day27_mathematical_report import (
    OU_DERIVATION_MATH_GROUP_ID,
    REPORT_MATH_GROUPS,
    report_math_placeholder,
)
from scripts.run_day27_mathematical_report import _latex_svg, enhance_typeset_math


def _ou_derivation_equations() -> tuple[str, ...]:
    return next(
        equations
        for group_id, _columns, equations in REPORT_MATH_GROUPS
        if group_id == OU_DERIVATION_MATH_GROUP_ID
    )


@pytest.mark.parametrize("latex", _ou_derivation_equations())
def test_every_new_ou_equation_renders_as_accessible_svg(latex: str) -> None:
    rendered = _latex_svg(latex)
    assert rendered.startswith("<svg ")
    assert 'role="img" aria-label="LaTeX equation:' in rendered
    assert "<metadata>" not in rendered


def test_math_enhancement_replaces_every_group_once() -> None:
    paragraphs = "".join(
        f"<p>{report_math_placeholder(group_id)}</p>"
        for group_id, _columns, _equations in REPORT_MATH_GROUPS
    )
    rendered, receipt = enhance_typeset_math(
        f"<!doctype html><html><head></head><body>{paragraphs}</body></html>"
    )

    assert receipt["equation_groups"] == len(REPORT_MATH_GROUPS)
    assert receipt["equations"] == sum(
        len(equations) for _group_id, _columns, equations in REPORT_MATH_GROUPS
    )
    assert receipt["static_replacements"] == len(REPORT_MATH_GROUPS)
    assert all(
        f"<p>{report_math_placeholder(group_id)}</p>" not in rendered
        for group_id, _columns, _equations in REPORT_MATH_GROUPS
    )
    assert (
        f'data-axiom-math-group="{OU_DERIVATION_MATH_GROUP_ID}"' in rendered
    )
