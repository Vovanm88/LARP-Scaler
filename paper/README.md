# LARP-Scaler paper

This directory contains the source of the LARP-Scaler preprint and all
reproducible figures.

## Layout

- `main.tex` — paper source;
- `references.bib` — bibliography;
- `data/benchmarks.json` — single machine-readable source for reported
  aggregate measurements;
- `data/training_run.json` — recovered final-stage config, Trackio provenance,
  checkpoint hash, runtime, and selected loss milestones;
- `scripts/make_figures.py` — deterministic SVG/PDF figure generator;
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

## Build the paper

From this directory:

```bash
latexmk -pdf main.tex
```

The release PDF should be copied or renamed to `paper/LARP-Scaler.pdf` only
after the draft has been reviewed. The PDF is intentionally absent from the
current repository state.

No institutional affiliations are listed. The documented 13.6-hour compute
figure covers only the final direct-refinement stage initialized from the
selected SFT checkpoint, not the full upstream training lineage.

## Release TODO

- add LPIPS, DISTS, no-reference IQA, and/or human-preference evaluation;
- add Gradio, arXiv, and Hugging Face Paper Page URLs;
- review dataset and upstream-model attributions and licenses;
- replace the temporary software citation in the repository README.
