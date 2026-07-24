"""Steady-state PiD latency at the checkpoint's native 512->2048 setting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--images", type=int, default=3)
    cli = parser.parse_args()

    inputs = sorted(cli.input_root.glob("*_low.png"))[: cli.images]
    if len(inputs) != cli.images:
        raise RuntimeError(f"Expected {cli.images} inputs, found {len(inputs)}")

    sys.argv = [
        "pid",
        "--backbone",
        "flux",
        "--input_path",
        str(inputs[0]),
        "--prompt",
        "high quality photo",
        "--degrade_sigmas",
        "0.0",
        "--cfg_scale",
        "1",
        "--pid_inference_steps",
        "4",
        "--scale",
        "4",
        "--load_ema_to_reg",
    ]
    from pid._src.inference.cli_utils import parse_clean_args
    from pid._src.inference.decoder import load_our_decoder
    from pid._src.inference.inference_utils import load_input_image

    args = parse_clean_args()
    model = load_our_decoder(args, [], True)

    @torch.inference_mode()
    def run(path: Path) -> Image.Image:
        image = load_input_image(str(path)).to(dtype=torch.bfloat16, device="cuda")
        if tuple(image.shape[-2:]) != (512, 512):
            raise ValueError(f"Expected a 512x512 PiD input, got {tuple(image.shape[-2:])}")
        latent = model.encode_lq_latent(image)
        height, width = latent.shape[-2] * 8, latent.shape[-1] * 8
        result = model.generate_samples_from_batch(
            {
                model.config.input_caption_key: [args.prompt],
                "LQ_latent": latent,
                "degrade_sigma": torch.zeros(1, device="cuda"),
            },
            cfg_scale=1,
            num_steps=4,
            seed=1234,
            shift=None,
            image_size=(height * 4, width * 4),
        )[0]
        if result.ndim == 4:
            result = result[:, 0]
        array = (
            result.float()
            .clamp(-1, 1)
            .add(1)
            .div(2)
            .mul(255)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return Image.fromarray(np.ascontiguousarray(array), "RGB")

    rows = []
    torch.cuda.reset_peak_memory_stats()
    for path in inputs:
        for _ in range(cli.warmups):
            run(path)
            torch.cuda.synchronize()
        for repeat in range(cli.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = run(path)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if output.size != (2048, 2048):
                raise ValueError(f"Unexpected output size: {output.size}")
            rows.append(
                {
                    "id": path.stem,
                    "repeat": repeat,
                    "seconds": seconds,
                    "output_size": list(output.size),
                }
            )

    times = np.asarray([row["seconds"] for row in rows], dtype=np.float64)
    payload = {
        "protocol": {
            "method": "PiD",
            "checkpoint": "PiD_res2k_sr4x_official_flux_distill_4step",
            "input_size": [512, 512],
            "output_size": [2048, 2048],
            "scale": 4,
            "steps": 4,
            "cfg_scale": 1,
            "seed": 1234,
            "warmups_per_image": cli.warmups,
            "timed_runs_per_image": cli.runs,
            "images": len(inputs),
            "model_loading_included": False,
            "file_writing_included": False,
            "input_conversion_and_cpu_output_included": True,
            "gpu": torch.cuda.get_device_name(0),
        },
        "summary": {
            "n": len(rows),
            "median_seconds": float(np.median(times)),
            "mean_seconds": float(times.mean()),
            "p10_seconds": float(np.quantile(times, 0.1)),
            "p90_seconds": float(np.quantile(times, 0.9)),
            "peak_memory_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        },
        "rows": rows,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
