from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 1234
METRICS = ("psnr", "ssim", "mae")


def canonical_pair_id(value: str) -> str:
    """Normalize historical low/target CSV naming to the underlying pair ID."""
    value = value.removesuffix("_low").removesuffix("_target")
    prefix, separator, remainder = value.partition("_")
    if separator and prefix.isdigit() and "-" in remainder:
        return remainder
    return value


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


def score(prediction: Image.Image, target: Image.Image, device: torch.device) -> dict[str, float]:
    if prediction.size != target.size:
        raise ValueError(f"Shape mismatch: {prediction.size} != {target.size}")
    pred = image_tensor(prediction, device)
    truth = image_tensor(target, device)
    error = pred - truth
    mse = error.square().mean()
    mae = error.abs().mean()
    window = torch.ones(3, 1, 11, 11, device=device) / 121
    mu_p = F.conv2d(pred, window, padding=5, groups=3)
    mu_t = F.conv2d(truth, window, padding=5, groups=3)
    var_p = F.conv2d(pred.square(), window, padding=5, groups=3) - mu_p.square()
    var_t = F.conv2d(truth.square(), window, padding=5, groups=3) - mu_t.square()
    covariance = F.conv2d(pred * truth, window, padding=5, groups=3) - mu_p * mu_t
    ssim = (
        (2 * mu_p * mu_t + 0.01**2)
        * (2 * covariance + 0.03**2)
        / (
            (mu_p.square() + mu_t.square() + 0.01**2)
            * (var_p + var_t + 0.03**2)
        )
    ).mean()
    return {
        "psnr": float((-10 * torch.log10(mse.clamp_min(1e-12))).item()),
        "ssim": float(ssim.item()),
        "mae": float(mae.item()),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_summary(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key in sorted(groups):
        group = groups[key]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, len(group), size=(BOOTSTRAP_SAMPLES, len(group)))
        summary = dict(zip(group_fields, key, strict=True))
        summary["n"] = len(group)
        for metric_name in METRICS:
            values = np.asarray([float(row[metric_name]) for row in group], dtype=np.float64)
            bootstrap_means = values[indices].mean(axis=1)
            summary[metric_name] = float(values.mean())
            summary[f"{metric_name}_ci95_low"] = float(np.quantile(bootstrap_means, 0.025))
            summary[f"{metric_name}_ci95_high"] = float(np.quantile(bootstrap_means, 0.975))
        summaries.append(summary)
    return summaries


def existing_rows(path: Path, domain: str, method: str, *, scale: int | None = None) -> list[dict]:
    rows = read_csv(path)
    if scale is not None:
        rows = [row for row in rows if int(row["scale"]) == scale]
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 rows in {path}, got {len(rows)}")
    return [
        {
            "domain": domain,
            "id": row["id"],
            "method": method,
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
            "mae": float(row["mae"]),
        }
        for row in rows
    ]


def current_12(
    device: torch.device, release: Path, inputs: Path
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    existing = {
        "photo": (
            release / "benchmark_supups_photos_x4" / "metrics.csv",
            release / "benchmark_realesrgan_photos_x4" / "metrics.csv",
            release / "benchmark_lua_photos" / "per_image.csv",
        ),
        "anime": (
            release / "benchmark_supups_anime_x4" / "metrics.csv",
            release / "benchmark_realesrgan_anime_x4" / "metrics.csv",
            release / "benchmark_lua_anime" / "per_image.csv",
        ),
    }
    for domain, paths in existing.items():
        domain_start = len(rows)
        rows.extend(existing_rows(paths[0], domain, "LARP-Scaler"))
        rows.extend(existing_rows(paths[1], domain, "Real-ESRGAN"))
        rows.extend(existing_rows(paths[2], domain, "LUA", scale=4))
        pair_root = (
            inputs / "photos_512_to_128"
            if domain == "photo"
            else inputs / "anime_512_to_128"
        )
        low_paths = sorted(pair_root.glob("*_low.png"))
        if len(low_paths) != 12:
            raise RuntimeError(f"Expected 12 low-resolution {domain} inputs, got {len(low_paths)}")
        expected_ids = {canonical_pair_id(path.stem) for path in low_paths}
        for low_path in low_paths:
            target_path = low_path.with_name(low_path.name.replace("_low.png", "_target.png"))
            if not target_path.is_file():
                raise FileNotFoundError(target_path)
            with Image.open(low_path) as opened:
                low = opened.convert("RGB")
            with Image.open(target_path) as opened:
                target = opened.convert("RGB")
            if low.size != (128, 128) or target.size != (512, 512):
                raise ValueError(f"Unexpected pair dimensions: {low_path} {low.size}, {target.size}")
            for method, resampling in (
                ("Bicubic", Image.Resampling.BICUBIC),
                ("Lanczos", Image.Resampling.LANCZOS),
            ):
                measured = score(low.resize(target.size, resampling), target, device)
                rows.append(
                    {
                        "domain": domain,
                        "id": low_path.stem,
                        "method": method,
                        **measured,
                    }
                )
        domain_rows = rows[domain_start:]
        for method in ("LARP-Scaler", "Real-ESRGAN", "LUA", "Bicubic", "Lanczos"):
            method_ids = {
                canonical_pair_id(row["id"])
                for row in domain_rows
                if row["method"] == method
            }
            if method_ids != expected_ids:
                raise RuntimeError(
                    f"{domain}/{method} IDs do not match the saved low/target pairs"
                )
    summaries = bootstrap_summary(rows, ("domain", "method"))
    return rows, summaries


def multi_scale_48(
    device: torch.device, release: Path, photo_pool: Path
) -> tuple[list[dict], list[dict]]:
    selection = json.loads(
        (release / "photo_matrix_native_sana" / "selection.json").read_text(encoding="utf-8")
    )
    if len(selection) != 48:
        raise RuntimeError(f"Expected 48 selected photos, got {len(selection)}")
    selected_ids = {row["id"] for row in selection}
    matrix = read_csv(release / "photo_matrix_native_sana" / "per_image_metrics.csv")
    larp_rows = [
        row
        for row in matrix
        if row["method"] == "supups_adapter"
        and float(row["sigma"]) == 0.35
        and int(row["steps"]) == 1
        and int(row["scale"]) in (2, 4, 8)
    ]
    if len(larp_rows) != 48 * 3:
        raise RuntimeError(f"Expected 144 LARP-Scaler rows, got {len(larp_rows)}")
    rows = [
        {
            "id": row["id"],
            "scale": int(row["scale"]),
            "method": "LARP-Scaler",
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
            "mae": float(row["mae"]),
        }
        for row in larp_rows
    ]
    if {row["id"] for row in rows} != selected_ids:
        raise RuntimeError("LARP-Scaler matrix IDs do not match the 48-photo selection")
    for item in selection:
        source = photo_pool / f"{item['id']}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if min(image.size) < 2048:
            raise ValueError(f"Source is smaller than 2048 px: {source} {image.size}")
        left = (image.width - 2048) // 2
        top = (image.height - 2048) // 2
        target = image.crop((left, top, left + 2048, top + 2048))
        for scale in (2, 4, 8):
            low = target.resize((2048 // scale, 2048 // scale), Image.Resampling.LANCZOS)
            for method, resampling in (
                ("Bicubic", Image.Resampling.BICUBIC),
                ("Lanczos", Image.Resampling.LANCZOS),
            ):
                measured = score(low.resize(target.size, resampling), target, device)
                rows.append(
                    {
                        "id": item["id"],
                        "scale": scale,
                        "method": method,
                        **measured,
                    }
                )
    for scale in (2, 4, 8):
        for method in ("LARP-Scaler", "Bicubic", "Lanczos"):
            method_ids = {
                row["id"]
                for row in rows
                if row["scale"] == scale and row["method"] == method
            }
            if method_ids != selected_ids:
                raise RuntimeError(
                    f"x{scale}/{method} IDs do not match the 48-photo selection"
                )
    summaries = bootstrap_summary(rows, ("scale", "method"))
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute interpolation baselines and bootstrap intervals."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--photo-pool", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "revision",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    current_rows, current_summary = current_12(
        device, args.release_root, args.input_root
    )
    matrix_rows, matrix_summary = multi_scale_48(
        device, args.release_root, args.photo_pool
    )
    write_csv(args.output / "controlled_x4_12_per_image.csv", current_rows)
    write_csv(args.output / "multiscale_48_photo_per_image.csv", matrix_rows)
    result = {
        "metadata": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "metrics": "RGB [0,1], full frame; SSIM uses an 11x11 uniform window",
            "validation": {
                "controlled_x4": {
                    "domains": 2,
                    "images_per_domain": 12,
                    "low_size": [128, 128],
                    "target_size": [512, 512],
                    "id_sets_equal_across_methods": True,
                },
                "multiscale_48_photo": {
                    "images": 48,
                    "target_size": [2048, 2048],
                    "low_sizes": {"2": [1024, 1024], "4": [512, 512], "8": [256, 256]},
                    "id_sets_equal_across_methods": True,
                },
            },
        },
        "controlled_x4_12": current_summary,
        "multiscale_48_photo": matrix_summary,
    }
    (args.output / "revision_benchmarks.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
