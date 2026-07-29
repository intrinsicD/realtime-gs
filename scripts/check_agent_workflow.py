#!/usr/bin/env python3
"""Validate realtime-gs agent coordination, discovery, and gate parity.

The repository already has strong domain gates for experiments, result bundles, and claims.
This checker covers the coordination layer around them:

1. The one active task is either the unchanged template or a complete valid record.
2. Archived task records have unique IDs, terminal state, and the same schema.
3. Status, turn, review verdict, self-review, and Driver/Reviewer identity agree.
4. Maturity shortfalls remain explicit follow-ups rather than false completion.
5. A linked experiment contract exists and makes the task Protected.
6. ``.agents/skills`` exactly mirrors canonical ``.claude/skills`` entries.
7. ``AGENTS.md`` remains a redirect to canonical ``CLAUDE.md``.
8. The PR template and local tool permissions expose the same review and gate surfaces.
9. Hosted CI invokes ``./scripts/verify.sh`` so local and hosted gates cannot drift.

Run: python scripts/check_agent_workflow.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_TASK_SECTIONS = (
    "Title",
    "Task ID",
    "Role Assignment",
    "Mode",
    "Risk",
    "Maturity",
    "Goal",
    "Motivation",
    "Success Criteria",
    "Constraints",
    "Non-Goals",
    "Selected Skills",
    "Experiment Contract",
    "Current Evidence",
    "Minimal Plan",
    "Status",
    "Human Decisions",
    "Handoff Log",
)

REQUIRED_FILLED_TASK_SECTIONS = (
    "Title",
    "Task ID",
    "Role Assignment",
    "Mode",
    "Risk",
    "Maturity",
    "Goal",
    "Motivation",
    "Success Criteria",
    "Constraints",
    "Non-Goals",
    "Selected Skills",
    "Experiment Contract",
    "Current Evidence",
    "Minimal Plan",
    "Status",
)

MODES = frozenset({"Explore", "Decide", "Implement", "Validate", "Stabilize", "Productize"})
RISKS = frozenset({"Standard", "Protected"})
MATURITY_LEVELS = (
    "Not applicable",
    "Scaffolded",
    "CPU-contracted",
    "Pipeline-integrated",
    "Calibrated",
    "Claim-ready",
    "Retired",
)
MATURITY_RANK = {name: index for index, name in enumerate(MATURITY_LEVELS[1:])}

STATUSES = frozenset(
    {
        "Not started",
        "In progress",
        "In review",
        "Revision required",
        "Blocked on human decision",
        "Accepted",
        "Accepted with follow-up",
        "Provisionally accepted (self-reviewed)",
        "Rejected",
        "Inconclusive",
        "Superseded",
    }
)
STATUS_TURNS = {
    "Not started": "driver",
    "In progress": "driver",
    "In review": "reviewer",
    "Revision required": "driver",
    "Blocked on human decision": "human",
    "Accepted": "driver",
    "Accepted with follow-up": "driver",
    "Provisionally accepted (self-reviewed)": "driver",
    "Rejected": "driver",
    "Inconclusive": "driver",
    "Superseded": "driver",
}
TERMINAL_STATUSES = frozenset(
    {"Accepted", "Accepted with follow-up", "Rejected", "Inconclusive", "Superseded"}
)
REVIEW_VERDICTS = frozenset(
    {"Accepted", "Accepted with follow-up", "Revision required", "Rejected", "Inconclusive"}
)
REQUIRED_HANDOFF_FIELDS = (
    "Objective",
    "Reviewed state",
    "Changes",
    "Evidence",
    "Assumptions",
    "Uncertainties",
    "Review Focus",
    "Protected actions not taken",
    "Recommended Next Action",
)
REQUIRED_REVIEW_FIELDS = (
    "Verdict",
    "Self-reviewed",
    "Correctness",
    "Evidence Quality",
    "Simplicity",
    "Missing Cases",
    "Required Changes",
    "Optional Improvements",
)
STATUS_REVIEW_VERDICTS = {
    "Accepted": {"Accepted"},
    "Accepted with follow-up": {"Accepted with follow-up"},
    "Provisionally accepted (self-reviewed)": {"Accepted", "Accepted with follow-up"},
    "Revision required": {"Revision required"},
    "Rejected": {"Rejected"},
    "Inconclusive": {"Inconclusive"},
}

TASK_ID_TEMPLATE = "Next unused `RTGS-NNN` identifier in `docs/tasks/` (for example `RTGS-001`)."
ROLE_TEMPLATE = """\
- Driver:
- Reviewer:
- Turn: driver / reviewer / human"""
MODE_TEMPLATE = "Explore / Decide / Implement / Validate / Stabilize / Productize"
RISK_TEMPLATE = "Standard / Protected"
MATURITY_TEMPLATE = (
    "- Target: Not applicable / Scaffolded / CPU-contracted / Pipeline-integrated / "
    "Calibrated / Claim-ready / Retired\n"
    "- Reached: Not applicable / Scaffolded / CPU-contracted / Pipeline-integrated / "
    "Calibrated / Claim-ready / Retired"
)
EXPERIMENT_TEMPLATE = "None, or one repository-relative `experiments/tasks/<task_id>.json` path."
STATUS_TEMPLATE = """\
Not started / In progress / In review / Revision required /
Blocked on human decision / Accepted / Accepted with follow-up /
Provisionally accepted (self-reviewed) / Rejected / Inconclusive / Superseded"""
HUMAN_DECISIONS_TEMPLATE = """\
Record escalated questions and dated answers here. An answer that exists only in chat is not
durable task state. Use one block per decision:

```markdown
### Question
### Options
### Recommendation
### Decision
### Date
```"""
HANDOFF_TEMPLATE = """\
Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template."""

TASK_TEMPLATE_SECTIONS = {
    "Title": "",
    "Task ID": TASK_ID_TEMPLATE,
    "Role Assignment": ROLE_TEMPLATE,
    "Mode": MODE_TEMPLATE,
    "Risk": RISK_TEMPLATE,
    "Maturity": MATURITY_TEMPLATE,
    "Goal": "",
    "Motivation": "",
    "Success Criteria": "",
    "Constraints": "",
    "Non-Goals": "",
    "Selected Skills": "-",
    "Experiment Contract": EXPERIMENT_TEMPLATE,
    "Current Evidence": "",
    "Minimal Plan": "",
    "Status": STATUS_TEMPLATE,
    "Human Decisions": HUMAN_DECISIONS_TEMPLATE,
    "Handoff Log": HANDOFF_TEMPLATE,
}

FENCE_OPEN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,}).*$")
REVIEW_BLOCK = re.compile(
    r"^### Review(?:[^\S\n]+\([^()\n]*\))?[^\S\n]*$\n?"
    r"(.*?)(?=^### |\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
HANDOFF_BLOCK = re.compile(
    r"^### Handoff(?:[^\S\n]+\([^()\n]*\))?[^\S\n]*$\n?"
    r"(.*?)(?=^### |\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
ARCHIVE_NAME = re.compile(r"^(RTGS-\d{3,})-[a-z0-9-]+\.md$")
TASK_ID = re.compile(r"^RTGS-\d{3,}$")
SKILL_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def markdown_lines(text: str):
    """Yield ``(line, inside_fence)`` while respecting Markdown code fences."""

    fence_character: str | None = None
    fence_length = 0
    for line in text.splitlines():
        if fence_character is not None:
            yield line, True
            closing = re.fullmatch(
                rf"[ ]{{0,3}}{re.escape(fence_character)}{{{fence_length},}}[^\S\n]*",
                line,
            )
            if closing:
                fence_character = None
                fence_length = 0
            continue

        opening = FENCE_OPEN.match(line)
        if opening:
            marker = opening.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            yield line, True
        else:
            yield line, False


def without_fenced_code(text: str) -> str:
    """Blank fenced code while retaining line boundaries for structural scans."""

    return "\n".join("" if fenced else line for line, fenced in markdown_lines(text))


def sections(
    text: str,
    level: int = 2,
    *,
    with_duplicates: bool = False,
) -> dict[str, str] | tuple[dict[str, str], set[str]]:
    """Split Markdown into a map at one heading level."""

    prefix = "#" * level + " "
    found: dict[str, str] = {}
    duplicates: set[str] = set()
    heading: str | None = None
    body: list[str] = []
    for line, fenced in markdown_lines(text):
        if not fenced and line.startswith(prefix):
            if heading is not None:
                if heading in found:
                    duplicates.add(heading)
                found[heading] = "\n".join(body).strip()
            heading = line[len(prefix) :].strip()
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        if heading in found:
            duplicates.add(heading)
        found[heading] = "\n".join(body).strip()
    if with_duplicates:
        return found, duplicates
    return found


def field_values(body: str, name: str) -> list[str]:
    """Read visible ``- Name: value`` fields from a section."""

    return [
        match.group(1).strip()
        for match in re.finditer(
            rf"^-[^\S\n]*{re.escape(name)}:[^\S\n]*(.*)$",
            without_fenced_code(body),
            re.MULTILINE,
        )
    ]


def structured_blocks(
    handoff_log: str,
    pattern: re.Pattern[str],
) -> list[tuple[dict[str, str], set[str]]]:
    """Return level-four fields for visible Handoff or Review entries."""

    parsed: list[tuple[dict[str, str], set[str]]] = []
    for match in pattern.finditer(without_fenced_code(handoff_log)):
        nested, duplicates = sections(match.group(1), level=4, with_duplicates=True)
        parsed.append((nested, duplicates))
    return parsed


def parse_task_document(text: str, source: str, problems: list[str]) -> dict[str, str]:
    """Parse one active or archived task and validate its heading skeleton."""

    task, duplicates = sections(text, with_duplicates=True)
    nonblank = [line for line in text.splitlines() if line.strip()]
    root_headings = [
        line[2:].strip()
        for line, fenced in markdown_lines(text)
        if not fenced and line.startswith("# ") and not line.startswith("## ")
    ]
    if not nonblank or nonblank[0] != "# Current Task" or root_headings != ["Current Task"]:
        problems.append(f"{source}: root heading must be exactly '# Current Task'")
    for heading in sorted(duplicates):
        problems.append(f"{source}: duplicate `## {heading}` section")
    for heading in sorted(set(task) - set(REQUIRED_TASK_SECTIONS)):
        problems.append(f"{source}: unexpected `## {heading}` section")
    return task


def is_unfilled_template(task: dict[str, str]) -> bool:
    """Recognize only the complete unchanged active-task template."""

    return set(task) == set(TASK_TEMPLATE_SECTIONS) and all(
        task.get(name) == value for name, value in TASK_TEMPLATE_SECTIONS.items()
    )


def _single_field(
    values: list[str],
    name: str,
    source: str,
    problems: list[str],
) -> str:
    if len(values) > 1:
        problems.append(f"{source}: duplicate field {name!r}")
    return values[0] if values else ""


def _validate_reviews(
    task: dict[str, str],
    source: str,
    status: str,
    driver: str,
    reviewer: str,
    problems: list[str],
) -> None:
    handoff_log = task.get("Handoff Log", "")
    handoffs = structured_blocks(handoff_log, HANDOFF_BLOCK)
    reviews = structured_blocks(handoff_log, REVIEW_BLOCK)
    for index, (handoff, duplicates) in enumerate(handoffs, start=1):
        for name in sorted(duplicates & set(REQUIRED_HANDOFF_FIELDS)):
            problems.append(f"{source}: Handoff {index} has duplicate `#### {name}` field")
        for name in REQUIRED_HANDOFF_FIELDS:
            if not handoff.get(name, "").strip():
                problems.append(f"{source}: Handoff {index} has no `#### {name}` content")

    requires_handoff = status in {
        "In review",
        "Revision required",
        "Accepted",
        "Accepted with follow-up",
        "Provisionally accepted (self-reviewed)",
        "Rejected",
        "Inconclusive",
    }
    if requires_handoff and not handoffs:
        problems.append(f"{source}: status {status!r} requires a structured Handoff block")

    for index, (review, duplicates) in enumerate(reviews, start=1):
        verdict = review.get("Verdict", "").strip()
        self_reviewed = review.get("Self-reviewed", "").strip().removesuffix(".")
        for name in sorted(duplicates & set(REQUIRED_REVIEW_FIELDS)):
            problems.append(f"{source}: Review {index} has duplicate `#### {name}` field")
        for name in REQUIRED_REVIEW_FIELDS:
            if not review.get(name, "").strip():
                problems.append(f"{source}: Review {index} has no `#### {name}` content")
        if verdict not in REVIEW_VERDICTS:
            problems.append(
                f"{source}: Review {index} Verdict must be one of "
                f"{sorted(REVIEW_VERDICTS)}, found {verdict!r}"
            )
        if self_reviewed not in {"Yes", "No"}:
            problems.append(
                f"{source}: Review {index} Self-reviewed must be 'Yes' or 'No', "
                f"found {self_reviewed!r}"
            )

    expected = STATUS_REVIEW_VERDICTS.get(status)
    if expected is None:
        return
    if not reviews:
        problems.append(f"{source}: status {status!r} requires a structured Review block")
        return

    latest, _duplicates = reviews[-1]
    verdict = latest.get("Verdict", "").strip()
    self_reviewed = latest.get("Self-reviewed", "").strip().removesuffix(".")
    if verdict not in expected:
        problems.append(
            f"{source}: status {status!r} requires latest Review Verdict in "
            f"{sorted(expected)}, found {verdict!r}"
        )

    provisional = status == "Provisionally accepted (self-reviewed)"
    if provisional:
        if self_reviewed != "Yes":
            problems.append(f"{source}: provisional acceptance requires Self-reviewed 'Yes'")
        if driver and reviewer and driver.strip().casefold() != reviewer.strip().casefold():
            problems.append(
                f"{source}: self-review requires the Driver and Reviewer labels to match"
            )
    else:
        if self_reviewed == "Yes":
            problems.append(f"{source}: Self-reviewed 'Yes' permits only provisional acceptance")
        if (
            self_reviewed == "No"
            and driver
            and reviewer
            and driver.strip().casefold() == reviewer.strip().casefold()
        ):
            problems.append(
                f"{source}: independent review requires distinct Driver and Reviewer labels"
            )

    if (
        status == "Revision required"
        and len(reviews) >= 2
        and all(
            review.get("Verdict", "").strip() == "Revision required"
            for review, _duplicates in reviews[-2:]
        )
    ):
        problems.append(
            f"{source}: two consecutive revision verdicts require human escalation, "
            "not another Revision required state"
        )


def _validate_maturity(
    task: dict[str, str],
    source: str,
    task_id: str,
    status: str,
    problems: list[str],
) -> None:
    body = task.get("Maturity", "")
    target = _single_field(field_values(body, "Target"), "Maturity Target", source, problems)
    reached = _single_field(field_values(body, "Reached"), "Maturity Reached", source, problems)
    for name, value in (("Target", target), ("Reached", reached)):
        if value not in MATURITY_LEVELS:
            problems.append(
                f"{source}: Maturity {name} must be one of {list(MATURITY_LEVELS)}, found {value!r}"
            )
    if target not in MATURITY_LEVELS or reached not in MATURITY_LEVELS:
        return
    if (target == "Not applicable") != (reached == "Not applicable"):
        problems.append(f"{source}: Maturity Target and Reached must agree on Not applicable")
        return
    if target == "Not applicable":
        return

    below_target = MATURITY_RANK[reached] < MATURITY_RANK[target]
    if status == "Accepted" and below_target:
        problems.append(
            f"{source}: Accepted task reached {reached!r} below target {target!r}; "
            "use Accepted with follow-up"
        )
    if status == "Accepted with follow-up":
        linked = set(re.findall(r"\bRTGS-\d{3,}\b", task.get("Handoff Log", "")))
        linked.discard(task_id)
        if not linked:
            problems.append(
                f"{source}: Accepted with follow-up requires a distinct RTGS-NNN in the Handoff Log"
            )
    if status in {"Accepted", "Accepted with follow-up"} and reached == "Scaffolded":
        closure_text = f"{task.get('Non-Goals', '')}\n{task.get('Handoff Log', '')}".casefold()
        if "intended endpoint" not in closure_text and "rtgs-" not in closure_text:
            problems.append(
                f"{source}: accepted Scaffolded work must name a follow-up or state that "
                "Scaffolded is the intended endpoint"
            )


def validate_task_record(
    task: dict[str, str],
    source: str,
    problems: list[str],
    *,
    root: Path,
    archived: bool,
    filename_task_id: str | None = None,
) -> None:
    """Validate the shared active/archive task schema."""

    for heading in REQUIRED_TASK_SECTIONS:
        if heading not in task:
            problems.append(f"{source}: missing `## {heading}` section")
    for heading in REQUIRED_FILLED_TASK_SECTIONS:
        if not task.get(heading, ""):
            problems.append(f"{source}: task has no {heading}")

    roles = task.get("Role Assignment", "")
    driver = _single_field(field_values(roles, "Driver"), "Driver", source, problems)
    reviewer = _single_field(field_values(roles, "Reviewer"), "Reviewer", source, problems)
    turn = _single_field(field_values(roles, "Turn"), "Turn", source, problems).lower()
    if not driver:
        problems.append(f"{source}: task has no Driver")
    if not reviewer:
        problems.append(f"{source}: task has no Reviewer")

    task_id = task.get("Task ID", "")
    if TASK_ID.fullmatch(task_id) is None:
        problems.append(f"{source}: Task ID must be RTGS-NNN, found {task_id!r}")
    elif archived and filename_task_id != task_id:
        problems.append(
            f"{source}: Task ID {task_id!r} does not match filename ID {filename_task_id!r}"
        )

    mode = task.get("Mode", "")
    if mode not in MODES:
        problems.append(f"{source}: Mode must be one of {sorted(MODES)}, found {mode!r}")
    risk = task.get("Risk", "")
    if risk not in RISKS:
        problems.append(f"{source}: Risk must be one of {sorted(RISKS)}, found {risk!r}")
    if task.get("Selected Skills", "").strip() == "-":
        problems.append(f"{source}: Selected Skills must replace the '-' placeholder")

    experiment = task.get("Experiment Contract", "").strip()
    if experiment != "None":
        experiment_path = Path(experiment)
        safe = (
            not experiment_path.is_absolute()
            and ".." not in experiment_path.parts
            and experiment_path.parts[:2] == ("experiments", "tasks")
            and experiment_path.suffix == ".json"
        )
        if not safe:
            problems.append(
                f"{source}: Experiment Contract must be None or experiments/tasks/<task_id>.json"
            )
        elif not (root / experiment_path).is_file():
            problems.append(f"{source}: Experiment Contract does not exist: {experiment}")
        if risk != "Protected":
            problems.append(f"{source}: a linked experiment contract requires Risk Protected")

    status = task.get("Status", "")
    if status not in STATUSES:
        problems.append(f"{source}: Status must be one of {sorted(STATUSES)}, found {status!r}")
    elif archived:
        if status not in TERMINAL_STATUSES:
            problems.append(f"{source}: archived task must have terminal Status, found {status!r}")
        if turn != "none":
            problems.append(f"{source}: archived task requires Turn 'none', found {turn!r}")
    else:
        expected_turn = STATUS_TURNS[status]
        if turn != expected_turn:
            problems.append(
                f"{source}: status {status!r} requires Turn {expected_turn!r}, found {turn!r}"
            )

    if status == "Blocked on human decision":
        decision_text = task.get("Human Decisions", "").strip()
        if not decision_text or decision_text == HUMAN_DECISIONS_TEMPLATE:
            problems.append(f"{source}: blocked task requires a recorded Human Decision")
    if status == "Superseded":
        handoff = task.get("Handoff Log", "").strip()
        if not handoff or handoff == HANDOFF_TEMPLATE:
            problems.append(f"{source}: Superseded task requires a reason in the Handoff Log")

    if status in STATUSES:
        _validate_reviews(task, source, status, driver, reviewer, problems)
        _validate_maturity(task, source, task_id, status, problems)


def _task_problems(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    archive_ids: dict[str, str] = {}
    archive_dir = root / "docs" / "tasks"
    if not archive_dir.is_dir():
        problems.append("missing directory: docs/tasks")
    else:
        for path in sorted(archive_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            source = path.relative_to(root).as_posix()
            match = ARCHIVE_NAME.fullmatch(path.name)
            filename_task_id = match.group(1) if match else None
            if match is None:
                problems.append(f"{source}: name must be RTGS-NNN-<lowercase-slug>.md")
            elif filename_task_id in archive_ids:
                problems.append(
                    f"{source}: task ID {filename_task_id} already used by "
                    f"{archive_ids[filename_task_id]}"
                )
            else:
                archive_ids[filename_task_id] = source
            task = parse_task_document(path.read_text(encoding="utf-8"), source, problems)
            validate_task_record(
                task,
                source,
                problems,
                root=root,
                archived=True,
                filename_task_id=filename_task_id,
            )

    active_path = root / ".agents" / "state" / "current-task.md"
    if not active_path.is_file():
        problems.append("missing file: .agents/state/current-task.md")
    else:
        task = parse_task_document(
            active_path.read_text(encoding="utf-8"),
            ".agents/state/current-task.md",
            problems,
        )
        if not is_unfilled_template(task):
            validate_task_record(
                task,
                ".agents/state/current-task.md",
                problems,
                root=root,
                archived=False,
            )
            task_id = task.get("Task ID", "")
            if task_id in archive_ids:
                problems.append(
                    f".agents/state/current-task.md: {task_id} is already archived as "
                    f"{archive_ids[task_id]}"
                )
    return problems, len(archive_ids)


def _skill_problems(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    canonical_dir = root / ".claude" / "skills"
    mirror_dir = root / ".agents" / "skills"
    if not canonical_dir.is_dir():
        return ["missing directory: .claude/skills"], 0
    if not mirror_dir.is_dir():
        return ["missing directory: .agents/skills"], 0

    canonical = {
        path.name: path
        for path in canonical_dir.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    mirrored = {
        path.name: path
        for path in mirror_dir.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    for name in sorted(set(canonical) - set(mirrored)):
        problems.append(f".agents/skills is missing canonical skill {name!r}")
    for name in sorted(set(mirrored) - set(canonical)):
        problems.append(f".agents/skills contains stale skill {name!r}")

    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    for name, path in sorted(canonical.items()):
        text = (path / "SKILL.md").read_text(encoding="utf-8")
        frontmatter = SKILL_FRONTMATTER.match(text)
        if frontmatter is None:
            problems.append(f"{path.relative_to(root)}/SKILL.md has no frontmatter")
            continue
        name_match = re.search(r"^name:\s*(\S+)\s*$", frontmatter.group(1), re.MULTILINE)
        description = re.search(r"^description:\s*(.+)$", frontmatter.group(1), re.MULTILINE)
        if name_match is None or name_match.group(1) != name:
            problems.append(f"skill {name!r} frontmatter name must match its directory")
        if description is None or not description.group(1).strip():
            problems.append(f"skill {name!r} frontmatter description is missing")
        if not name.startswith(("rtgs-", "realtime-gs-")):
            problems.append(f"skill {name!r} is not repository-prefixed")
        if name not in claude:
            problems.append(f"skill {name!r} is not routed from CLAUDE.md")

        mirror = mirrored.get(name)
        if mirror is not None:
            if not mirror.is_symlink():
                problems.append(f".agents/skills/{name} must be a symlink to the canonical skill")
            else:
                try:
                    target = mirror.resolve(strict=True)
                    expected = path.resolve(strict=True)
                except FileNotFoundError:
                    problems.append(f".agents/skills/{name} is a broken symlink")
                else:
                    if target != expected:
                        problems.append(
                            f".agents/skills/{name} resolves outside .claude/skills/{name}"
                        )
    return problems, len(canonical)


def _authority_problems(root: Path) -> list[str]:
    problems: list[str] = []
    agents_path = root / "AGENTS.md"
    if not agents_path.is_file():
        problems.append("missing file: AGENTS.md")
    else:
        agents = agents_path.read_text(encoding="utf-8").casefold()
        if "claude.md" not in agents or "canonical" not in agents:
            problems.append("AGENTS.md must remain a canonical redirect to CLAUDE.md")

    ci_path = root / ".github" / "workflows" / "ci.yml"
    if not ci_path.is_file():
        problems.append("missing file: .github/workflows/ci.yml")
    else:
        ci = ci_path.read_text(encoding="utf-8")
        if re.search(r"run:[^\S\n]+(?:\|[^\S\n]*\n[^\S\n]*)?\./scripts/verify\.sh", ci) is None:
            problems.append(".github/workflows/ci.yml must invoke ./scripts/verify.sh")

    pr_template_path = root / ".github" / "pull_request_template.md"
    if not pr_template_path.is_file():
        problems.append("missing file: .github/pull_request_template.md")
    else:
        pr_template = pr_template_path.read_text(encoding="utf-8")
        for fragment in (
            "Task ID",
            "Driver",
            "Reviewer",
            "Risk",
            "Target maturity",
            "Reached maturity",
            "Self-reviewed",
            "Protected actions not taken",
            "./scripts/verify.sh",
        ):
            if fragment not in pr_template:
                problems.append(f".github/pull_request_template.md is missing {fragment!r}")

    verify_path = root / "scripts" / "verify.sh"
    if not verify_path.is_file():
        problems.append("missing file: scripts/verify.sh")
    else:
        verify = verify_path.read_text(encoding="utf-8")
        for command in (
            'CUDA_VISIBLE_DEVICES=""',
            "scripts/check_agent_workflow.py",
            "scripts/experiment_contract.py validate",
        ):
            if command not in verify:
                problems.append(f"scripts/verify.sh does not run {command}")

    settings_path = root / ".claude" / "settings.json"
    if not settings_path.is_file():
        problems.append("missing file: .claude/settings.json")
    else:
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f".claude/settings.json is invalid JSON: {error}")
        else:
            allowed = settings.get("permissions", {}).get("allow", [])
            for fragment in ("check_agent_workflow.py", "experiment_contract.py"):
                if not any(fragment in str(item) for item in allowed):
                    problems.append(f".claude/settings.json permissions do not expose {fragment}")
    return problems


def validate(root: Path = ROOT) -> tuple[list[str], int, int]:
    """Return ``(problems, skill_count, archived_task_count)``."""

    task_errors, archive_count = _task_problems(root)
    skill_errors, skill_count = _skill_problems(root)
    return task_errors + skill_errors + _authority_problems(root), skill_count, archive_count


def main() -> int:
    problems, skill_count, archive_count = validate()
    if problems:
        print(f"check_agent_workflow: {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"check_agent_workflow: OK ({skill_count} skills, "
        f"{archive_count} archived tasks, active task/template valid)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
