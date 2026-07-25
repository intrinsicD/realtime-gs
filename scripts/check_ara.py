#!/usr/bin/env python3
"""Structural validator for the Agent-Native Research Artifact under ``ara/``.

``ara/`` is the repository's claim and evidence ledger: ``ara/logic/claims.md`` is where a
quantitative or capability statement becomes a tracked claim with a falsification criterion
and a proof binding. Nothing enforced that ledger before this checker, so a claim could enter
``README.md`` or ``docs/`` with no row, or a row could cite a proof path that had been moved
or never existed.

Checks are structural on purpose — they catch drift that misleads agents, never prose style:

  1. Required ``ara/`` layer files exist.
  2. Every layer-index path referenced in ``ara/PAPER.md`` exists on disk.
  3. Every claim heading matches ``## C<NN>: <title>`` and every claim ID is unique.
  4. Every claim carries the nine required fields.
  5. Every ``Status`` starts with a known disposition word.
  6. Every ``Dependencies`` entry names a claim ID defined in this file.
  7. Every ``Proof`` entry that looks like a repository path exists on disk.
  8. Every ``From staging`` observation ID is defined in ``ara/staging/observations.yaml``.
  9. ``ara/`` is described in CLAUDE.md so agents discover the ledger at all.

Run: python scripts/check_ara.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARA = ROOT / "ara"

REQUIRED_FILES = (
    "ara/PAPER.md",
    "ara/logic/problem.md",
    "ara/logic/claims.md",
    "ara/logic/concepts.md",
    "ara/staging/observations.yaml",
    "ara/trace/exploration_tree.yaml",
    "ara/evidence/README.md",
)

REQUIRED_CLAIM_FIELDS = (
    "Statement",
    "Status",
    "Provenance",
    "Crystallized via",
    "Falsification criteria",
    "Proof",
    "Dependencies",
    "Tags",
    "From staging",
)

# First word of a Status line. Free-form qualifiers may follow (structsplat-style
# "refuted development actual-rate claim"), but the disposition itself must be one of these.
STATUS_WORDS = frozenset(
    {
        "supported",
        "refuted",
        "untested",
        "unavailable",
        "hypothesis",
        "superseded",
        "withdrawn",
    }
)

# A Proof entry is treated as a repository path when it starts with one of these roots.
# ``runs/`` is deliberately absent: it is gitignored local run output, so a fresh clone cannot
# resolve it. A claim may cite a runs/ artifact for context, but its durable binding must be a
# tracked path.
PATH_ROOTS = ("ara/", "benchmarks/", "docs/", "src/", "tests/", "scripts/", "dataset/")

CLAIM_HEADING = re.compile(r"^## (C\d+):\s*(\S.*)$")
FIELD = re.compile(r"^- \*\*([A-Za-z ]+)\*\*:\s*(.*)$")

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        err(f"missing file: {path.relative_to(ROOT)}")
        return ""


def check_required_files() -> None:
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            err(f"required ara file missing: {rel}")


def check_paper_layer_index() -> None:
    """Every backticked path in ara/PAPER.md must resolve, relative to ara/ or to the root."""
    paper = read(ARA / "PAPER.md")
    for token in re.findall(r"`([\w./-]+\.(?:md|yaml|json))`", paper):
        if (ARA / token).exists() or (ROOT / token).exists():
            continue
        err(f"ara/PAPER.md references '{token}' which does not exist")


def parse_claims() -> dict[str, dict[str, str]]:
    """Return {claim_id: {field: value}} parsed from ara/logic/claims.md.

    Field values may wrap across several indented continuation lines; they are joined with a
    single space so a ``Proof`` list split over four lines parses the same as a one-line list.
    """
    text = read(ARA / "logic" / "claims.md")
    claims: dict[str, dict[str, str]] = {}
    current: str | None = None
    field_name: str | None = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        heading = CLAIM_HEADING.match(line)
        if heading:
            claim_id = heading.group(1)
            if claim_id in claims:
                err(f"ara/logic/claims.md:{lineno}: duplicate claim id '{claim_id}'")
            claims.setdefault(claim_id, {})
            current, field_name = claim_id, None
            continue
        if line.startswith("## "):
            err(
                f"ara/logic/claims.md:{lineno}: heading is not a well-formed claim "
                f"('## C<NN>: <title>'): {line.strip()!r}"
            )
            current, field_name = None, None
            continue
        if current is None:
            continue
        field = FIELD.match(line)
        if field:
            field_name = field.group(1)
            claims[current][field_name] = field.group(2).strip()
            continue
        # Indented continuation of the field opened above.
        if field_name and line.startswith("  ") and line.strip():
            claims[current][field_name] = f"{claims[current][field_name]} {line.strip()}".strip()
            continue
        if not line.strip():
            field_name = None
    return claims


def check_claim_fields(claims: dict[str, dict[str, str]]) -> None:
    for claim_id, fields in claims.items():
        for required in REQUIRED_CLAIM_FIELDS:
            if required not in fields:
                err(f"claim {claim_id} is missing required field '{required}'")


def check_claim_statuses(claims: dict[str, dict[str, str]]) -> None:
    for claim_id, fields in claims.items():
        status = fields.get("Status", "")
        if not status:
            continue
        first = status.split()[0].strip(".,;:").lower()
        if first not in STATUS_WORDS:
            err(
                f"claim {claim_id} has status '{status}' whose disposition '{first}' is not one "
                f"of {sorted(STATUS_WORDS)}"
            )


def check_claim_dependencies(claims: dict[str, dict[str, str]]) -> None:
    for claim_id, fields in claims.items():
        for dep in re.findall(r"C\d+", fields.get("Dependencies", "")):
            if dep not in claims:
                err(f"claim {claim_id} depends on '{dep}' which is not defined in claims.md")
            if dep == claim_id:
                err(f"claim {claim_id} depends on itself")


def check_claim_proofs(claims: dict[str, dict[str, str]]) -> None:
    for claim_id, fields in claims.items():
        proof = fields.get("Proof", "")
        if not proof:
            continue
        entries = re.findall(r"`([^`]+)`", proof) or [
            part.strip() for part in proof.strip("[]").split(",")
        ]
        cited_path = False
        for entry in entries:
            token = entry.strip().strip("`\"'")
            if not token.startswith(PATH_ROOTS):
                continue
            cited_path = True
            # A claim may cite a specific test as "path::test_name"; resolve the file part.
            path_part = token.split("::", 1)[0]
            if not (ROOT / path_part).exists():
                err(f"claim {claim_id} cites proof path '{path_part}' which does not exist")
        status = fields.get("Status", "").split()
        disposition = status[0].lower() if status else ""
        if disposition in {"supported", "refuted"} and not cited_path:
            err(
                f"claim {claim_id} is '{disposition}' but its Proof cites no repository "
                "artifact path (a disposed claim must be bound to evidence on disk)"
            )


def check_staging_ids(claims: dict[str, dict[str, str]]) -> None:
    observations = read(ARA / "staging" / "observations.yaml")
    known = set(re.findall(r"^\s*-\s*id:\s*(O\d+)\s*$", observations, flags=re.MULTILINE))
    for claim_id, fields in claims.items():
        for obs in re.findall(r"O\d+", fields.get("From staging", "")):
            if obs not in known:
                err(
                    f"claim {claim_id} came 'From staging: {obs}' but that observation is not "
                    "defined in ara/staging/observations.yaml"
                )


def check_ara_is_documented() -> None:
    """The ledger is useless if agents never learn it exists."""
    claude = read(ROOT / "CLAUDE.md")
    if "ara/logic/claims.md" not in claude:
        err("CLAUDE.md does not mention ara/logic/claims.md (agents cannot discover the ledger)")


def main() -> int:
    check_required_files()
    check_paper_layer_index()
    claims = parse_claims()
    if not claims:
        err("ara/logic/claims.md defines no claims (expected at least one '## C<NN>:' heading)")
    check_claim_fields(claims)
    check_claim_statuses(claims)
    check_claim_dependencies(claims)
    check_claim_proofs(claims)
    check_staging_ids(claims)
    check_ara_is_documented()
    if errors:
        print(f"check_ara: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"check_ara: OK ({len(claims)} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
