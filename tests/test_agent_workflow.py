"""Structural tests for repository agent coordination and gate parity."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKFLOW = _load_script("check_agent_workflow")


def _handoff() -> str:
    return """\
### Handoff

#### Objective
Exercise the workflow fixture.
#### Reviewed state
fixture digest
#### Changes
Fixture only.
#### Evidence
Focused structural test.
#### Assumptions
None.
#### Uncertainties
None.
#### Review Focus
State-machine behavior.
#### Protected actions not taken
No external actions.
#### Recommended Next Action
Review the fixture.
"""


def _review(*, verdict: str, self_reviewed: str, suffix: str = "") -> str:
    return f"""\
### Review{suffix}

#### Verdict
{verdict}
#### Self-reviewed
{self_reviewed}
#### Correctness
The fixture is structurally correct.
#### Evidence Quality
Focused structural evidence.
#### Simplicity
No unnecessary mechanism.
#### Missing Cases
None for this fixture.
#### Required Changes
None.
#### Optional Improvements
None.
"""


def _task_text(
    *,
    status: str,
    turn: str,
    driver: str = "driver",
    reviewer: str = "reviewer",
    handoff_log: str,
) -> str:
    return f"""\
# Current Task

## Title
Workflow fixture
## Task ID
RTGS-900
## Role Assignment
- Driver: {driver}
- Reviewer: {reviewer}
- Turn: {turn}
## Mode
Validate
## Risk
Standard
## Maturity
- Target: CPU-contracted
- Reached: CPU-contracted
## Goal
Exercise the task state machine.
## Motivation
Prevent false completion.
## Success Criteria
The fixture is accepted only with a valid review.
## Constraints
Remain local.
## Non-Goals
No runtime experiment.
## Selected Skills
- `rtgs-task-workflow`
## Experiment Contract
None
## Current Evidence
Focused structural tests.
## Minimal Plan
Validate the record.
## Status
{status}
## Human Decisions
None.
## Handoff Log
{handoff_log}
"""


def _problems(text: str) -> list[str]:
    problems: list[str] = []
    task = WORKFLOW.parse_task_document(text, "fixture.md", problems)
    WORKFLOW.validate_task_record(
        task,
        "fixture.md",
        problems,
        root=REPO,
        archived=False,
    )
    return problems


def test_repository_agent_workflow_is_valid() -> None:
    problems, _skill_count, _archive_count = WORKFLOW.validate(REPO)
    assert problems == []


def test_provisional_self_review_accepts_only_same_normalized_identity() -> None:
    valid = _task_text(
        status="Provisionally accepted (self-reviewed)",
        turn="driver",
        driver="Codex",
        reviewer="codex",
        handoff_log=_handoff() + _review(verdict="Accepted", self_reviewed="Yes"),
    )
    assert _problems(valid) == []

    invalid = _task_text(
        status="Accepted",
        turn="driver",
        driver="Codex",
        reviewer="codex",
        handoff_log=_handoff() + _review(verdict="Accepted", self_reviewed="No"),
    )
    assert any("independent review requires distinct" in item for item in _problems(invalid))


def test_review_requires_the_complete_structured_shape() -> None:
    text = _task_text(
        status="Accepted",
        turn="driver",
        handoff_log=(_handoff() + "### Review\n\n#### Verdict\nAccepted\n#### Self-reviewed\nNo\n"),
    )
    problems = _problems(text)
    assert any("Review 1 has no `#### Correctness` content" in item for item in problems)
    assert any("Review 1 has no `#### Evidence Quality` content" in item for item in problems)


def test_every_follow_up_verdict_names_a_distinct_task() -> None:
    text = _task_text(
        status="Accepted with follow-up",
        turn="driver",
        handoff_log=_handoff() + _review(verdict="Accepted with follow-up", self_reviewed="No"),
    )
    assert any(
        "Accepted with follow-up requires a distinct RTGS-NNN" in item for item in _problems(text)
    )


def test_two_revision_verdicts_escalate_to_a_human() -> None:
    text = _task_text(
        status="Revision required",
        turn="driver",
        handoff_log=(
            _handoff()
            + _review(
                verdict="Revision required",
                self_reviewed="No",
                suffix=" (first)",
            )
            + _review(
                verdict="Revision required",
                self_reviewed="No",
                suffix=" (second)",
            )
        ),
    )
    assert any("two consecutive revision verdicts" in item for item in _problems(text))
