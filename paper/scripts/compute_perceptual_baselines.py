"""Compute LPIPS/DISTS for deterministic interpolation baselines.

The inputs and targets are the same saved 128->512 pairs used by the
controlled x4 benchmark. LPIPS uses the AlexNet backbone and DISTS is provided
by piq. Per-image values and seeded bootstrap intervals are written separately
from the distortion-metric CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
import piq
import torch
from PIL import Image


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 1234
METHODS = {
    "Bicubic": Image.Resampling.BICUBIC,
    "Lanczos": Image.Resampling.LANCZOS,
}


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
        .to(device)
    )


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    dists_metric = piq.DISTS().to(device).eval()
    rows: list[dict[str, str | float]] = []

    roots = {
        "photo": args.input_root / "photos_512_to_128",
        "anime": args.input_root / "anime_512_to_128",
    }
    for domain, root in roots.items():
        low_paths = sorted(root.glob("*_low.png"))
        if len(low_paths) != 12:
            raise RuntimeError(f"Expected 12 {domain} pairs, found {len(low_paths)}")
        for low_path in low_paths:
            target_path = low_path.with_name(low_path.name.replace("_low.png", "_target.png"))
            if not target_path.is_file():
                raise FileNotFoundError(target_path)
            with Image.open(low_path) as opened:
                low = opened.convert("RGB")
            with Image.open(target_path) as opened:
                target = opened.convert("RGB")
            if low.size != (128, 128) or target.size != (512, 512):
                raise ValueError(f"Unexpected dimensions for {low_path}: {low.size}, {target.size}")
            truth = image_tensor(target, device)
            for method, resampling in METHODS.items():
                prediction = image_tensor(low.resize(target.size, resampling), device)
                with torch.inference_mode():
                    lpips_value = float(
                        lpips_metric(prediction * 2 - 1, truth * 2 - 1).item()
                    )
                    dists_value = float(dists_metric(prediction, truth).item())
                rows.append(
                    {
                        "domain": domain,
                        "id": low_path.stem,
                        "method": method,
                        "lpips": lpips_value,
                        "dists": dists_value,
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for domain in roots:
        results[domain] = {}
        for method in METHODS:
            selected = [
                row for row in rows if row["domain"] == domain and row["method"] == method
            ]
            summary: dict[str, float | int] = {"n": len(selected)}
            for metric in ("lpips", "dists"):
                mean, low, high = bootstrap([float(row[metric]) for row in selected])
                summary[metric] = mean
                summary[f"{metric}_ci95_low"] = low
                summary[f"{metric}_ci95_high"] = high
            results[domain][method] = summary

    payload = {
        "protocol": {
            "input_size": [128, 128],
            "target_size": [512, 512],
            "scale": 4,
            "images_per_domain": 12,
            "lpips_backbone": "alex",
            "dists_implementation": "piq",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "results": results,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
