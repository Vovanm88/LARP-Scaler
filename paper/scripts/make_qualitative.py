#!/usr/bin/env python3
"""Build JPEG and SVG qualitative montages from real benchmark images."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CELL = 420
HEADER = 60
LABEL = 48


def fitted(image: Image.Image, size: int) -> Image.Image:
    """Scale an image to fill the montage cell.

    Low-resolution inputs are smaller than the cell and must be enlarged so that
    every column is displayed at the same physical size. Enlargement uses
    nearest-neighbour so the montage shows the actual low-resolution pixels
    instead of an implicitly interpolated version of the input.
    """
    output = Image.new("RGB", (size, size), "white")
    copy = image.copy()
    scale = min(size / copy.width, size / copy.height)
    target = (max(1, round(copy.width * scale)), max(1, round(copy.height * scale)))
    resample = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST
    copy = copy.resize(target, resample)
    output.paste(copy, ((size - copy.width) // 2, (size - copy.height) // 2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("qualitative_comparison.jpg"))
    args = parser.parse_args()

    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    columns = spec["columns"]
    cases = spec["cases"]
    if not columns or not cases:
        raise ValueError("The manifest must contain non-empty 'columns' and 'cases' lists")

    width = LABEL + CELL * len(columns)
    height = HEADER + CELL * len(cases)
    images: list[list[Image.Image]] = []
    for case in cases:
        row = []
        for column in columns:
            path = (args.manifest.parent / case["images"][column["key"]]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing qualitative image: {path}")
            with Image.open(path) as opened:
                row.append(fitted(opened.convert("RGB"), CELL))
        images.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)
    header_font = ImageFont.load_default()
    row_font = ImageFont.load_default()
    for index, column in enumerate(columns):
        text = column["label"]
        box = draw.textbbox((0, 0), text, font=header_font)
        x = LABEL + index * CELL + (CELL - (box[2] - box[0])) // 2
        draw.text((x, 20), text, fill="black", font=header_font)
    for row_index, case in enumerate(cases):
        draw.text((8, HEADER + row_index * CELL + 12), case["label"], fill="black", font=row_font)
        for column_index, image in enumerate(images[row_index]):
            montage.paste(image, (LABEL + column_index * CELL, HEADER + row_index * CELL))
    montage.save(args.output, "JPEG", quality=92, optimize=True, progressive=True)

    svg_path = args.output.with_suffix(".svg")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {width} {height}">',
        "<title>LARP-Scaler qualitative comparison</title>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for index, column in enumerate(columns):
        x = LABEL + index * CELL + CELL / 2
        svg.append(
            f'<text x="{x}" y="38" text-anchor="middle" font-family="Arial" font-size="22" '
            f'font-weight="700">{html.escape(column["label"])}</text>'
        )
    for row_index, case in enumerate(cases):
        cy = HEADER + row_index * CELL + CELL / 2
        svg.append(
            f'<text x="26" y="{cy}" text-anchor="middle" transform="rotate(-90 26 {cy})" '
            f'font-family="Arial" font-size="18" font-weight="700">{html.escape(case["label"])}</text>'
        )
        for column_index, image in enumerate(images[row_index]):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=93)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            x = LABEL + column_index * CELL
            y = HEADER + row_index * CELL
            svg.append(
                f'<image x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'href="data:image/jpeg;base64,{encoded}"/>'
            )
    svg.append("</svg>")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} and {svg_path}")


if __name__ == "__main__":
    main()
