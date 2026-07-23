from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .pipeline import LarpScaler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale an image with LarpScaler.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--scale", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dtypes = {
        "auto": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    upscaler = LarpScaler.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtypes[args.dtype],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    result = upscaler.upscale(
        args.input,
        scale=args.scale,
        steps=args.steps,
        noise_level=args.noise_level,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved {result.width}x{result.height} image to {args.output}")


if __name__ == "__main__":
    main()
