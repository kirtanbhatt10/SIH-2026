# Evidence folder — populate from your own recordings

This folder is intentionally **empty in git** until you run the builder against
your real recordings. It is where measured, committable proof lives.

## Why this exists

`.gitignore` excludes `*.wav` and `*.png`, so every physical experiment we ran
left no trace in the repository. The AI charter requires the opposite:

- §3  — *"Report experimentally measured limits."*
- §10 — AI 1 deliverable: *"Hardware frequency capability measurement."*
- §13 — *Do not present unmeasured values as results.*

`docs/evidence/` is allow-listed in `.gitignore`, so WAV/PNG placed here **are**
committed while stray recordings elsewhere stay ignored.

## How to populate it

From the repo root, pointing at wherever your recordings actually are:

```bash
# preview only, writes nothing
python scripts/make_evidence.py --curated --input-dir /path/to/recordings --dry-run

# generate the evidence set
python scripts/make_evidence.py --curated --input-dir /path/to/recordings

# commit
git add docs/evidence
git commit -m "docs: add measured hardware feasibility evidence"
```

Drop `--curated` to process every known recording instead of the decisive six.

## What gets produced

```
docs/evidence/
├── audio/            2 s mono 16-bit excerpts   (~188 KB each)
├── spectra/          spectrum + spectrogram PNG (~250-350 KB each)
├── measurements.json machine-readable measured values + methodology
├── measurements.csv  same, for the PPT lead
└── MANIFEST.md       results table + SHA-256 provenance of each source file
```

## Budget

Keep the whole folder under ~15 MB. The `--curated` set is about 4 MB.
Only commit artifacts that defend a specific claim in the report or PPT.

## Warning

Do not commit output generated from synthetic/test audio and present it as
measured hardware results. The builder records a SHA-256 of every source file
so provenance is checkable — keep it honest.
