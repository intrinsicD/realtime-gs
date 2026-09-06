# Work handoff — 7 September 2026

The code and tracked research evidence belong to the `main` branches of
[intrinsicD/structsplat](https://github.com/intrinsicD/structsplat) and
[intrinsicD/realtime-gs](https://github.com/intrinsicD/realtime-gs).

The complete current reports and calibrated inputs are available as checksummed assets in the
[private work-handoff release](https://github.com/intrinsicD/structsplat-research-artifacts/releases/tag/work-handoff-2026-09-07).
Sign in with the `intrinsicD` GitHub account or another account granted access to that repository.
The artifact repository is private because the inputs and previews include dome captures.

## Restore at work

Clone both code repositories into the same parent directory, or run `git pull --ff-only` on
`main` in existing clean checkouts. The example below uses `~/Documents` as that parent.
The archives restore paths under `structsplat/` and `realtime-gs/`; extract into that parent,
not inside either repository. Use fresh checkouts if the work computer has local changes in
its datasets or results: extraction restores the archived files at their original relative paths.

```bash
mkdir -p "$HOME/Documents/work-handoff-20260907"
gh release download work-handoff-2026-09-07 \
  --repo intrinsicD/structsplat-research-artifacts \
  --dir "$HOME/Documents/work-handoff-20260907"

cd "$HOME/Documents/work-handoff-20260907"
sha256sum --check SHA256SUMS

for name in structsplat-reports realtime-gs-reports realtime-gs-data; do
  cat "$name".tar.gz.part-* | tar -xz -C "$HOME/Documents"
done
```

Each archive's `.manifest.json` lists every restored file, its byte size and SHA-256. Identical
files share tar hard links to reduce transfer size; copy a restored file before editing it if
its preserved evidence bytes must remain independent. The archives preserve historical reports,
source snapshots and review receipts as they were recorded, including original machine paths.
They do not rerun, relabel or revise frozen experiments.

## Open the tomography report

```bash
cd "$HOME/Documents/realtime-gs"
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory runs/20260906_tomography_source_constraints_haelyn_dome
```

Open <http://127.0.0.1:8765/index.html>. After setting up the normal repository environment, the
saved representative reconstruction can be inspected with:

```bash
.venv/bin/rtgs view \
  --gaussians runs/20260906_tomography_source_constraints_haelyn_dome/gaussians.ply \
  --initial runs/20260906_tomography_source_constraints_haelyn_dome/gaussians_init.ply \
  --no-open
```

Use the canonical `RESULT` and `AUDIT` records under `benchmarks/results/` and the archived
RTGS-016 task for interpretation. Frozen execution source is preserved in the run's
`source_snapshot.tar.gz`; installing current dependencies alone does not recreate the original
producer environment. The run's environment and source receipts describe that environment.

## Archive scope

- `realtime-gs-data`: the complete local `dataset/`, including the downloaded Haelyn reference,
  calibrated dome inputs and compact teachers.
- `realtime-gs-reports`: the complete local `runs/`, including the final tomography report,
  models, previews, histories, source snapshot, audit provenance and browser receipts.
- `structsplat-reports`: the complete local `runs/` and the 5 September code-driven,
  HIER-033, HIER-034, HIER-035 and HIER-036 report directories. The manifests enumerate the roots.
  Their audited evidence archives are also tracked in the code repository.

Older unrelated StructSplat `results/` directories, dependency environments, build caches,
local CLI settings and temporary scratch files are outside this handoff. Virtual environments
must be installed on the work machine using each repository's setup instructions.
