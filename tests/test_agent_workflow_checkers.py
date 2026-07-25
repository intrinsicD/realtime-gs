"""Tests for the agent-workflow structural checkers under scripts/.

These checkers gate agent behavior (claim-ledger hygiene, results-bundle completeness,
scripts/ layout), so a regression in them silently removes a guardrail. Each test drives the
checker's ``main``/entry function over a temporary fixture rather than the real repository, so
the tests stay fast and do not depend on current repository content — except the three
"live repository passes" tests, which assert the committed tree is actually clean.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name: str) -> ModuleType:
    """Import a scripts/*.py checker as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_checker_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# check_results_bundle
# --------------------------------------------------------------------------------------

BUNDLE = _load("check_results_bundle")

INDEX_HTML = """<html><body>
<h1>Run summary</h1><p>PSNR 24.50 dB, SSIM 0.812</p>
<a href="gaussians.ply">final gaussians</a>
<a href="gaussians_init.ply">initial gaussians</a>
<a href="metrics.json">metrics.json</a>
<img src="reconstruction_contact_sheet.png">
</body></html>
"""

RECEIPT = """# Smoke receipt
- served index.html from repo root, HTTP 200 for page and every local target
- rtgs view --gaussians runs/x/gaussians.ply --scene dataset/s/frame_00008 --downscale 16
"""


def _write_bundle(root: Path, *, previews: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE.REQUIRED_ARTIFACTS:
        (root / name).write_text("placeholder", encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps({"metrics": {"psnr": 24.5, "ssim": 0.812}, "timings": {"fit": 1.0}}),
        encoding="utf-8",
    )
    if previews:
        for name in BUNDLE.REQUIRED_PREVIEWS:
            (root / name).write_text("placeholder", encoding="utf-8")
    (root / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (root / "smoke_receipt.md").write_text(RECEIPT, encoding="utf-8")
    return root


def test_complete_bundle_passes(tmp_path: Path) -> None:
    assert BUNDLE.check_bundle(_write_bundle(tmp_path / "run"), previews=True) == []


def test_missing_results_page_fails(tmp_path: Path) -> None:
    run = _write_bundle(tmp_path / "run")
    (run / "index.html").unlink()
    problems = BUNDLE.check_bundle(run, previews=True)
    assert any("missing index.html" in p for p in problems)


def test_broken_relative_link_fails(tmp_path: Path) -> None:
    run = _write_bundle(tmp_path / "run")
    (run / "reconstruction_contact_sheet.png").unlink()
    problems = BUNDLE.check_bundle(run, previews=False)
    assert any("reconstruction_contact_sheet.png" in p and "does not exist" in p for p in problems)


def test_absolute_link_fails(tmp_path: Path) -> None:
    run = _write_bundle(tmp_path / "run")
    (run / "index.html").write_text(
        INDEX_HTML.replace('href="gaussians.ply"', 'href="/abs/gaussians.ply"'), encoding="utf-8"
    )
    problems = BUNDLE.check_bundle(run, previews=True)
    assert any("absolute link" in p for p in problems)


def test_receipt_without_viewer_command_fails(tmp_path: Path) -> None:
    run = _write_bundle(tmp_path / "run")
    (run / "smoke_receipt.md").write_text(
        "# Smoke receipt\n- index.html HTTP 200\n", encoding="utf-8"
    )
    problems = BUNDLE.check_bundle(run, previews=True)
    assert any("rtgs view" in p for p in problems)


def test_page_without_numbers_fails(tmp_path: Path) -> None:
    """A page that only links metrics.json is not summary-bound."""
    run = _write_bundle(tmp_path / "run")
    (run / "index.html").write_text(
        '<html><body><a href="gaussians.ply">g</a><a href="metrics.json">m</a></body></html>',
        encoding="utf-8",
    )
    problems = BUNDLE.check_bundle(run, previews=True)
    assert any("no numeric value" in p for p in problems)


def test_non_finite_metric_fails(tmp_path: Path) -> None:
    run = _write_bundle(tmp_path / "run")
    (run / "metrics.json").write_text('{"metrics": {"psnr": NaN}}', encoding="utf-8")
    problems = BUNDLE.check_bundle(run, previews=True)
    assert any("non-finite" in p for p in problems)


def test_no_previews_flag_skips_preview_requirement(tmp_path: Path) -> None:
    run = _write_bundle(tmp_path / "run", previews=False)
    (run / "index.html").write_text(
        INDEX_HTML.replace('<img src="reconstruction_contact_sheet.png">', ""), encoding="utf-8"
    )
    assert BUNDLE.check_bundle(run, previews=False) == []
    assert BUNDLE.check_bundle(run, previews=True) != []


# --------------------------------------------------------------------------------------
# Live repository state
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("checker", ["docs_sync", "check_ara", "check_script_layout"])
def test_repository_passes_checker(checker: str, capsys: pytest.CaptureFixture[str]) -> None:
    """The committed tree must satisfy every structural checker verify.sh runs."""
    module = _load(checker)
    assert module.main() == 0, capsys.readouterr().err


def test_every_skill_name_is_prefixed() -> None:
    """Skill names must be repo-prefixed so they cannot collide across open repositories."""
    skills = sorted((REPO / ".claude" / "skills").iterdir())
    assert skills, "no skills found"
    for skill in skills:
        skill_md = skill / "SKILL.md"
        if not skill_md.is_file():
            continue
        name_lines = [
            line
            for line in skill_md.read_text(encoding="utf-8").splitlines()
            if line.startswith("name:")
        ]
        assert name_lines, f"{skill.name}/SKILL.md has no 'name:' frontmatter field"
        name = name_lines[0].removeprefix("name:").strip()
        assert name == skill.name, f"{skill.name}/SKILL.md declares mismatched name {name!r}"
        assert name.startswith(("rtgs-", "realtime-gs-")), (
            f"skill {name!r} is not repo-prefixed; an unprefixed name collides with a sibling "
            "repository's skill of the same name"
        )
