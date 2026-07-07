#!/usr/bin/env python3
"""Docs <-> code consistency checker.

Fails (exit 1) when documentation has drifted from the code. Checks are structural on
purpose — they catch the drift that actually misleads agents (missing modules, phantom CLI
commands, stale file references), not prose style.

Checks:
  1. Required docs exist (ARCHITECTURE, RESEARCH, ROADMAP, BENCHMARKS, EXPERIMENTS,
     CLAUDE.md, AGENTS.md, README.md).
  2. Every subpackage of ``rtgs`` is mentioned in docs/ARCHITECTURE.md and CLAUDE.md.
  3. Every CLI subcommand defined in ``rtgs.cli`` is documented in docs/ARCHITECTURE.md,
     and no documented command is missing from the CLI.
  4. Every registered lifter name appears in docs/ARCHITECTURE.md.
  5. Every ``.claude/skills/*/SKILL.md`` skill is listed in CLAUDE.md.
  6. Every path-like reference in CLAUDE.md's repository map exists on disk.
  7. Every public module in src/rtgs has a module docstring.
  8. docs/BENCHMARKS.md contains the auto-generated markers benchmarks/run.py rewrites.

Run: python scripts/docs_sync.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "rtgs"

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        err(f"missing file: {path.relative_to(ROOT)}")
        return ""


def check_required_docs() -> None:
    for rel in [
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "docs/ARCHITECTURE.md",
        "docs/RESEARCH.md",
        "docs/ROADMAP.md",
        "docs/BENCHMARKS.md",
        "docs/EXPERIMENTS.md",
    ]:
        if not (ROOT / rel).is_file():
            err(f"required doc missing: {rel}")


def check_subpackages_documented() -> None:
    arch = read(ROOT / "docs" / "ARCHITECTURE.md")
    claude = read(ROOT / "CLAUDE.md")
    for pkg in sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()):
        if pkg not in arch:
            err(f"subpackage 'rtgs/{pkg}' not mentioned in docs/ARCHITECTURE.md")
        if pkg not in claude:
            err(f"subpackage 'rtgs/{pkg}' not mentioned in CLAUDE.md repository map")


def cli_commands_from_source() -> set[str]:
    """Parse rtgs/cli.py for add_parser("<cmd>") calls without importing torch."""
    tree = ast.parse(read(SRC / "cli.py"))
    cmds = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            cmds.add(node.args[0].value)
    return cmds


def check_cli_documented() -> None:
    arch = read(ROOT / "docs" / "ARCHITECTURE.md")
    in_code = cli_commands_from_source()
    documented = set(re.findall(r"`rtgs ([a-z0-9-]+)", arch))
    for cmd in sorted(in_code - documented):
        err(
            f"CLI command 'rtgs {cmd}' exists in rtgs/cli.py but is not documented "
            "in docs/ARCHITECTURE.md (use the form `rtgs <cmd> ...`)"
        )
    for cmd in sorted(documented - in_code):
        err(f"docs/ARCHITECTURE.md documents 'rtgs {cmd}' but rtgs/cli.py does not define it")


def lifters_from_source() -> set[str]:
    """Parse rtgs/lift/__init__.py for the _LIFTERS registry keys."""
    tree = ast.parse(read(SRC / "lift" / "__init__.py"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        is_registry = any(isinstance(t, ast.Name) and t.id == "_LIFTERS" for t in node.targets)
        if is_registry:
            names.update(key.value for key in node.value.keys if isinstance(key, ast.Constant))
    return names


def check_lifters_documented() -> None:
    arch = read(ROOT / "docs" / "ARCHITECTURE.md")
    for name in sorted(lifters_from_source()):
        if f"`{name}`" not in arch:
            err(f"lifter '{name}' is registered but not documented in docs/ARCHITECTURE.md")


def check_skills_listed() -> None:
    skills_dir = ROOT / ".claude" / "skills"
    if not skills_dir.is_dir():
        err("missing .claude/skills directory")
        return
    claude = read(ROOT / "CLAUDE.md")
    for skill in sorted(p.name for p in skills_dir.iterdir() if (p / "SKILL.md").is_file()):
        if skill not in claude:
            err(f"skill '.claude/skills/{skill}' not mentioned in CLAUDE.md")


def check_claude_md_paths() -> None:
    claude = read(ROOT / "CLAUDE.md")
    # Check path-like tokens that clearly refer to repo files/dirs.
    for token in re.findall(
        r"(?:^|[\s`(])((?:src|tests|benchmarks|docs|scripts|\.claude|\.github)/[\w./-]*)", claude
    ):
        rel = token.rstrip(".,;:")
        if not (ROOT / rel).exists():
            err(f"CLAUDE.md references '{rel}' which does not exist")


def check_module_docstrings() -> None:
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            err(f"module missing docstring: {path.relative_to(ROOT)}")


def check_benchmark_markers() -> None:
    bench = read(ROOT / "docs" / "BENCHMARKS.md")
    for marker in ("<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->"):
        if marker not in bench:
            err(
                f"docs/BENCHMARKS.md missing marker {marker} (benchmarks/run.py --update-docs "
                "rewrites the block between them)"
            )


def main() -> int:
    check_required_docs()
    check_subpackages_documented()
    check_cli_documented()
    check_lifters_documented()
    check_skills_listed()
    check_claude_md_paths()
    check_module_docstrings()
    check_benchmark_markers()
    if errors:
        print(f"docs_sync: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("docs_sync: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
