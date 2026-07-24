# Qualitative comparison inputs

No synthetic or generated placeholder images are used as experimental evidence.
Add real benchmark outputs here and create a manifest such as:

```json
{
  "columns": [
    {"key": "input", "label": "LR input"},
    {"key": "real_esrgan", "label": "Real-ESRGAN"},
    {"key": "lua", "label": "LUA"},
    {"key": "pid", "label": "PiD"},
    {"key": "larp_scaler", "label": "LARP-Scaler"},
    {"key": "ground_truth", "label": "Ground truth"}
  ],
  "cases": [
    {
      "label": "Photo 01",
      "images": {
        "input": "photo-01/input.png",
        "real_esrgan": "photo-01/real-esrgan.png",
        "lua": "photo-01/lua.png",
        "pid": "photo-01/pid.png",
        "larp_scaler": "photo-01/larp-scaler.png",
        "ground_truth": "photo-01/ground-truth.png"
      }
    }
  ]
}
```

Then run:

```bash
python paper/scripts/make_qualitative.py paper/qualitative/manifest.json \
  --output paper/figures/qualitative_comparison.pdf
```

The paper automatically replaces its explicit placeholder when the PDF exists.
