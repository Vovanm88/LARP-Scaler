# LARP-Scaler paper

This directory contains the source of the LARP-Scaler preprint and all
reproducible figures.

## Layout

- `main.tex` — paper source;
- `references.bib` — bibliography;
- `data/benchmarks.json` — single machine-readable source for reported
  aggregate measurements;
- `data/revision/` — per-image reconstruction metrics and bootstrap intervals;
- `data/training_run.json` — progressive curriculum, recovered final-stage
  config, Trackio provenance, checkpoint hash, runtime, and selected loss
  milestones;
- `scripts/compute_revision_metrics.py` — validates saved pair IDs and sizes,
  computes Bicubic/Lanczos rows, and produces seeded bootstrap intervals;
- `scripts/make_figures.py` — deterministic SVG/PDF/PNG figure generator;
- `scripts/make_qualitative.py` — real-image qualitative montage builder;
- `figures/` — generated vector figures;
- `qualitative/` — real per-case benchmark outputs and the montage manifest.

## Regenerate figures

The scripts require Python 3.10+, Pillow, and ReportLab. They do not require
Matplotlib.

```bash
pip install -e ".[paper]"
python paper/scripts/make_figures.py
```

The metric script is intentionally path-driven because the source images and
saved learned-baseline runs are not committed:

```bash
python paper/scripts/compute_revision_metrics.py \
  --release-root /path/to/saved-release-runs \
  --input-root /path/to/benchmark-inputs \
  --photo-pool /path/to/native-photo-pool
```

## Build the paper

From this directory:

```bash
latexmk -pdf -jobname=larpscaler_draft main.tex
```

This produces `paper/larpscaler_draft.pdf`. Rename a reviewed release copy only
after the arXiv record and final release metadata exist.

No institutional affiliations are listed. The documented 13.6-hour compute
figure covers only the final direct-refinement stage initialized from the
selected SFT checkpoint, not the full upstream training lineage.

## Remaining release work

- add no-reference IQA and/or human-preference evaluation;
- add arXiv and Hugging Face Paper Page URLs;
- review dataset and upstream-model attributions and licenses;
- replace the temporary software citation in the repository README.
