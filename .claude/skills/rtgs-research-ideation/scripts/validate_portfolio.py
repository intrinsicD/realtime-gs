#!/usr/bin/env python3
"""Check a realtime-gs research portfolio's structure and calibrated language."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN = (
    "definitely novel",
    "no one has ever done this",
    "guaranteed publication",
    "guarantees a publication",
)
SECTIONS = (
    "frontier map",
    "functional problem signature",
    "anti-library",
    "productive recombinations",
    "exploratory candidates",
    "transformational candidates",
    "cross-domain transfers",
    "new-evidence discovery programs",
    "pareto set",
    "recommended first experiment",
    "audit limitations",
)
IDEA_FIELDS = (
    "central claim",
    "novelty class",
    "known foundation",
    "irreducible delta",
    "new prediction",
    "cheapest killing test",
    "null hypothesis",
    "prior-art threats",
    "novelty confidence",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate(text: str) -> list[str]:
    """Return structural/calibration problems; do not infer scientific novelty."""

    problems: list[str] = []
    lowered = normalize(text)
    for phrase in FORBIDDEN:
        if phrase in lowered:
            problems.append(f"forbidden certainty claim: {phrase!r}")
    for section in SECTIONS:
        if section not in lowered:
            problems.append(f"missing portfolio section: {section}")
    for field in IDEA_FIELDS:
        if field not in lowered:
            problems.append(f"missing idea-card field: {field}")
    if re.search(r"\bN[0-4](?:-T)?\b", text, re.IGNORECASE) is None:
        problems.append("no novelty class (N0-N4, optionally -T) found")
    if "cutoff" not in lowered:
        problems.append("missing literature/search cutoff")
    if "broken correspondence" not in lowered:
        problems.append("missing broken-correspondence analysis")
    if "adoption barrier" not in lowered:
        problems.append("missing adoption barrier")
    candidate_headings = re.findall(
        r"^#{3,4}\s+.*(?:candidate|transfer|program)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if len(candidate_headings) < 6:
        problems.append("fewer than six candidate/transfer/program headings")
    return problems


def self_test() -> int:
    valid = """
# Research portfolio
Literature cutoff: 2026-07-29. Sources searched: papers and repositories.
## Frontier map
## Functional problem signature
## Anti-library
## Productive recombinations
### Candidate: one
Central claim: test. Novelty class: N2. Known foundation: A. Irreducible delta: D.
New prediction: P. Cheapest killing test: E. Null hypothesis: H0. Prior-art threats: W.
Novelty confidence: provisional.
### Candidate: two
## Exploratory candidates
### Candidate: three
## Transformational candidates
### Candidate: four
## Cross-domain transfers
### Transfer: five
Broken correspondence: B. Adoption barrier: C.
## New-evidence discovery programs
### Program: six
## Pareto set
## Recommended first experiment
## Audit limitations
"""
    invalid = "This is definitely novel and guaranteed publication."
    ok = not validate(valid) and bool(validate(invalid))
    print("self-test: PASS" if ok else "self-test: FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.portfolio is None:
        parser.error("provide a Markdown portfolio or --self-test")
    if not args.portfolio.is_file():
        print(f"error: file not found: {args.portfolio}", file=sys.stderr)
        return 2
    problems = validate(args.portfolio.read_text(encoding="utf-8"))
    if problems:
        print("portfolio validation: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("portfolio validation: PASS")
    print("Note: structural validation does not establish novelty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
