# Qualitative comparison inputs

These are real outputs from the native-resolution 12-photo benchmark; no
synthetic placeholders are used as experimental evidence. Each directory
contains a 512×512 LR input, 2048×2048 outputs, and a 2048×2048 ground truth.
PiD was run with the official `PiD_res2k_sr4x` four-step checkpoint in its
intended 512→2048 regime. Exact source IDs and protocol metadata are recorded
in `manifest.json`.

Then run:

```bash
python paper/scripts/make_qualitative.py paper/qualitative/manifest.json \
  --output paper/figures/qualitative_comparison.jpg
```

The script writes the JPEG used by the paper, the SVG used by GitHub, and a PDF
copy for publication workflows.
