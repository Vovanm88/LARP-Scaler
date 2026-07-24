#!/usr/bin/env python3
"""Generate matching SVG and PDF figures without optional plotting packages."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.pdfgen import canvas as pdf_canvas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "benchmarks.json"
DEFAULT_TRAINING_DATA = ROOT / "data" / "training_run.json"
DEFAULT_OUTPUT = ROOT / "figures"

COLORS = {
    "LARP-Scaler": "#2563eb",
    "Real-ESRGAN": "#f59e0b",
    "PiD": "#db2777",
    "LUA": "#10b981",
    "Lanczos": "#64748b",
    "Bicubic": "#94a3b8",
    "neutral": "#64748b",
    "light": "#e2e8f0",
    "ink": "#0f172a",
    "muted": "#475569",
    "white": "#ffffff",
}


def hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))


class Figure:
    """Small top-left-coordinate vector surface backed by SVG and ReportLab."""

    def __init__(self, output: Path, name: str, width: int, height: int, title: str, desc: str):
        output.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.svg_path = output / f"{name}.svg"
        self.pdf_path = output / f"{name}.pdf"
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" '
            f'viewBox="0 0 {width} {height}">',
            f"<title id=\"title\">{html.escape(title)}</title>",
            f"<desc id=\"desc\">{html.escape(desc)}</desc>",
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        ]
        self.pdf = pdf_canvas.Canvas(
            str(self.pdf_path),
            pagesize=(width, height),
            pageCompression=1,
            invariant=1,
        )
        self.pdf.setTitle(title)

    def _pdf_y(self, y: float) -> float:
        return self.height - y

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = COLORS["white"],
        stroke: str = COLORS["light"],
        stroke_width: float = 1,
        radius: float = 0,
    ) -> None:
        self.svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'rx="{radius:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )
        self.pdf.setFillColorRGB(*hex_rgb(fill))
        self.pdf.setStrokeColorRGB(*hex_rgb(stroke))
        self.pdf.setLineWidth(stroke_width)
        self.pdf.roundRect(x, self.height - y - height, width, height, radius, fill=1, stroke=1)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = COLORS["neutral"],
        width: float = 1,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}"{dash_attr}/>'
        )
        self.pdf.setStrokeColorRGB(*hex_rgb(color))
        self.pdf.setLineWidth(width)
        if dash:
            self.pdf.setDash(*[float(part) for part in dash.split()])
        else:
            self.pdf.setDash()
        self.pdf.line(x1, self._pdf_y(y1), x2, self._pdf_y(y2))

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 12,
        color: str = COLORS["ink"],
        anchor: str = "start",
        bold: bool = False,
    ) -> None:
        weight = "700" if bold else "400"
        self.svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size:.2f}" font-weight="{weight}" text-anchor="{anchor}">'
            f"{html.escape(value)}</text>"
        )
        self.pdf.setFillColorRGB(*hex_rgb(color))
        self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        if anchor == "middle":
            self.pdf.drawCentredString(x, self._pdf_y(y), value)
        elif anchor == "end":
            self.pdf.drawRightString(x, self._pdf_y(y), value)
        else:
            self.pdf.drawString(x, self._pdf_y(y), value)

    def rotated_text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        angle: float,
        size: float = 12,
        color: str = COLORS["ink"],
        bold: bool = False,
    ) -> None:
        """Draw centered text rotated in the top-left SVG coordinate system."""
        weight = "700" if bold else "400"
        self.svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size:.2f}" '
            f'font-weight="{weight}" text-anchor="middle" dominant-baseline="middle" '
            f'transform="rotate({angle:.2f} {x:.2f} {y:.2f})">'
            f"{html.escape(value)}</text>"
        )
        self.pdf.saveState()
        self.pdf.translate(x, self._pdf_y(y))
        self.pdf.rotate(-angle)
        self.pdf.setFillColorRGB(*hex_rgb(color))
        self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.pdf.drawCentredString(0, -size * 0.35, value)
        self.pdf.restoreState()

    def circle(self, x: float, y: float, radius: float, *, fill: str, stroke: str = COLORS["white"]) -> None:
        self.svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}"/>'
        )
        self.pdf.setFillColorRGB(*hex_rgb(fill))
        self.pdf.setStrokeColorRGB(*hex_rgb(stroke))
        self.pdf.circle(x, self._pdf_y(y), radius, fill=1, stroke=1)

    def polyline(self, points: list[tuple[float, float]], *, color: str, width: float = 2) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        path = self.pdf.beginPath()
        path.moveTo(points[0][0], self._pdf_y(points[0][1]))
        for x, y in points[1:]:
            path.lineTo(x, self._pdf_y(y))
        self.pdf.setStrokeColorRGB(*hex_rgb(color))
        self.pdf.setLineWidth(width)
        self.pdf.drawPath(path, stroke=1, fill=0)

    def polygon(self, points: list[tuple[float, float]], *, fill: str) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(f'<polygon points="{encoded}" fill="{fill}"/>')
        path = self.pdf.beginPath()
        path.moveTo(points[0][0], self._pdf_y(points[0][1]))
        for x, y in points[1:]:
            path.lineTo(x, self._pdf_y(y))
        path.close()
        self.pdf.setFillColorRGB(*hex_rgb(fill))
        self.pdf.drawPath(path, stroke=0, fill=1)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, *, color: str = COLORS["neutral"]) -> None:
        self.line(x1, y1, x2, y2, color=color, width=1.5)
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 9
        spread = 0.55
        points = [
            (x2, y2),
            (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread)),
            (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread)),
        ]
        self.polygon(points, fill=color)

    def save(self) -> None:
        self.svg.append("</svg>")
        self.svg_path.write_text("\n".join(self.svg) + "\n", encoding="utf-8")
        self.pdf.showPage()
        self.pdf.save()
        document = pdfium.PdfDocument(str(self.pdf_path))
        page = document[0]
        bitmap = page.render(scale=2)
        bitmap.to_pil().save(self.pdf_path.with_suffix(".png"))
        bitmap.close()
        page.close()
        document.close()


def node(fig: Figure, x: float, y: float, width: float, height: float, title: str, lines: list[str], color: str) -> None:
    fig.rect(x, y, width, height, fill=COLORS["white"], stroke=color, stroke_width=1.6, radius=10)
    fig.text(x + width / 2, y + 30, title, size=13, anchor="middle", bold=True)
    for index, line in enumerate(lines):
        fig.text(x + width / 2, y + 52 + index * 15, line, size=10, color=COLORS["muted"], anchor="middle")


def architecture(output: Path) -> None:
    fig = Figure(
        output,
        "architecture",
        1120,
        330,
        "LARP-Scaler architecture",
        "Low-resolution input is resized and encoded for direct SANA latent refinement; an independent guidance image enters through a gated adapter.",
    )
    fig.text(24, 30, "Direct latent refinement with independent image guidance", size=18, bold=True)
    node(fig, 25, 80, 135, 90, "LR image", ["PIL or file"], COLORS["neutral"])
    node(fig, 205, 80, 145, 90, "Lanczos resize", ["target geometry"], COLORS["neutral"])
    node(fig, 395, 80, 145, 90, "AutoencoderDC", ["32-channel latent"], COLORS["neutral"])
    node(fig, 600, 65, 190, 120, "SANA DiT refiner", ["flow-matching steps", "+ cached text condition"], COLORS["LARP-Scaler"])
    node(fig, 845, 80, 125, 90, "VAE decode", ["HR image"], COLORS["neutral"])
    node(fig, 205, 220, 145, 80, "Guidance image", ["original or custom"], COLORS["Real-ESRGAN"])
    node(fig, 395, 220, 145, 80, "VAE encode", ["low-res latent"], COLORS["Real-ESRGAN"])
    node(fig, 600, 215, 190, 90, "Image adapter", ["3-stage upsampler", "+ gated cross-attention"], COLORS["Real-ESRGAN"])
    for start, end in [
        ((160, 125), (205, 125)),
        ((350, 125), (395, 125)),
        ((540, 125), (600, 125)),
        ((790, 125), (845, 125)),
        ((350, 260), (395, 260)),
        ((540, 260), (600, 260)),
    ]:
        fig.arrow(*start, *end)
    fig.arrow(695, 215, 695, 185, color=COLORS["Real-ESRGAN"])
    fig.save()


def data_pipeline(output: Path) -> None:
    fig = Figure(
        output,
        "data_pipeline",
        1120,
        340,
        "LARP-Scaler data pipeline",
        "Six-stage pipeline from diverse image sources through native-resolution quality profiling, distilled scoring, captioning, and assembly.",
    )
    fig.text(24, 32, "Training-corpus preparation", size=18, bold=True)
    stages = [
        ("Sources", ["photos, art,", "anime"], COLORS["neutral"]),
        ("Native IQA", ["sharpness,", "artifacts, gates"], COLORS["LARP-Scaler"]),
        ("DINOv2", ["1024-D visual", "embeddings"], COLORS["LARP-Scaler"]),
        ("VLM distillation", ["photo + general", "quality students"], COLORS["Real-ESRGAN"]),
        ("Captioning", ["source +", "Qwen2.5-VL"], COLORS["PiD"]),
        ("Assembly", ["clean pool +", "defect bad-pool"], COLORS["LUA"]),
    ]
    xs = [24, 207, 390, 573, 756, 939]
    for index, (title, lines, color) in enumerate(stages):
        node(fig, xs[index], 85, 145, 115, title, lines, color)
        if index < len(stages) - 1:
            fig.arrow(xs[index] + 145, 142, xs[index + 1], 142)
    fig.text(
        560,
        260,
        "288,423 captioned records • dual quality scores • 2,500 defect-annotated negatives • 18,873 anime images",
        size=12,
        anchor="middle",
        color=COLORS["muted"],
    )
    fig.save()


def axes(fig: Figure, x: float, y: float, width: float, height: float, *, y_ticks: list[float], y_min: float, y_max: float) -> None:
    for value in y_ticks:
        py = y + height - (value - y_min) / (y_max - y_min) * height
        fig.line(x, py, x + width, py, color=COLORS["light"], width=1)
        fig.text(x - 8, py + 4, f"{value:g}", size=9, color=COLORS["muted"], anchor="end")
    fig.line(x, y + height, x + width, y + height, color=COLORS["neutral"], width=1)
    fig.line(x, y, x, y + height, color=COLORS["neutral"], width=1)


def quality(data: dict, output: Path) -> None:
    fig = Figure(
        output,
        "quality_comparison",
        1120,
        650,
        "Controlled reconstruction quality",
        "PSNR and SSIM comparisons on twelve photo and twelve anime images at four-times enlargement.",
    )
    fig.text(560, 30, "Controlled ×4 reference reconstruction (12 images per domain)", size=18, anchor="middle", bold=True)
    panels = [
        ("photo", "psnr_db", "Photo — PSNR (dB) ↑", 14, 34, [15, 20, 25, 30]),
        ("anime", "psnr_db", "Anime — PSNR (dB) ↑", 14, 31, [15, 20, 25, 30]),
        ("photo", "ssim", "Photo — SSIM ↑", 0.1, 0.9, [0.2, 0.4, 0.6, 0.8]),
        ("anime", "ssim", "Anime — SSIM ↑", 0.1, 0.95, [0.2, 0.4, 0.6, 0.8]),
    ]
    origins = [(70, 85), (620, 85), (70, 365), (620, 365)]
    for (domain, metric, title, y_min, y_max, ticks), (px, py) in zip(panels, origins):
        rows = data["quality_x4_512"][domain]
        fig.text(px + 210, py, title, size=14, anchor="middle", bold=True)
        chart_y = py + 25
        axes(fig, px, chart_y, 420, 195, y_ticks=ticks, y_min=y_min, y_max=y_max)
        slot = 420 / len(rows)
        for index, row in enumerate(rows):
            value = row[metric]
            bar_width = min(70, slot * 0.58)
            x = px + slot * index + (slot - bar_width) / 2
            bar_height = (value - y_min) / (y_max - y_min) * 195
            y = chart_y + 195 - bar_height
            fig.rect(x, y, bar_width, bar_height, fill=COLORS[row["method"]], stroke=COLORS[row["method"]])
            precision = 2 if metric == "psnr_db" else 3
            fig.text(x + bar_width / 2, y - 7, f"{value:.{precision}f}", size=9, anchor="middle", bold=True)
            fig.text(x + bar_width / 2, chart_y + 214, row["method"], size=8, anchor="middle")
    fig.save()


def latency(data: dict, output: Path) -> None:
    fig = Figure(
        output,
        "latency_1024",
        1080,
        500,
        "End-to-end latency comparison",
        "Median latency at two-times, four-times, and eight-times enlargement on twenty-four real photographs.",
    )
    fig.text(540, 30, "1024px output on 24 real photographs (240 timed calls per cell)", size=17, anchor="middle", bold=True)
    rows = data["latency_1024_24_photos"]["rows"]
    order = ["LARP-Scaler", "LUA", "PiD", "Real-ESRGAN"]
    present = {row["method"] for row in rows}
    methods = [m for m in order if m in present]
    scales = [2, 4, 8]
    x0, y0, width, height = 90, 75, 900, 325
    axes(fig, x0, y0, width, height, y_ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], y_min=0, y_max=3)
    group_width = width / len(scales)
    bar_width = 48
    offset = (len(methods) - 1) / 2
    for scale_index, scale in enumerate(scales):
        center = x0 + group_width * (scale_index + 0.5)
        for method_index, method in enumerate(methods):
            value = next(row["median_s"] for row in rows if row["method"] == method and row["scale"] == scale)
            x = center + (method_index - offset) * (bar_width + 5) - bar_width / 2
            bar_height = value / 3 * height
            y = y0 + height - bar_height
            fig.rect(x, y, bar_width, bar_height, fill=COLORS[method], stroke=COLORS[method])
            fig.text(x + bar_width / 2, y - 7, f"{value:.2f}", size=8, anchor="middle")
        fig.text(center, y0 + height + 27, f"×{scale}", size=12, anchor="middle", bold=True)
    fig.rotated_text(
        28,
        y0 + height / 2,
        "Median latency (s)",
        angle=-90,
        size=11,
        bold=True,
    )
    legend_x = 230
    for index, method in enumerate(methods):
        x = legend_x + index * 190
        fig.rect(x, 452, 15, 15, fill=COLORS[method], stroke=COLORS[method])
        fig.text(x + 22, 464, method, size=10)
    fig.save()


def ablation(data: dict, output: Path) -> None:
    fig = Figure(
        output,
        "prompt_adapter_ablation",
        900,
        480,
        "Prompt and image-adapter ablation",
        "Prompt changes provide small gains while matched guidance enters through the trained adapter branch.",
    )
    fig.text(450, 30, "Prompt × image-adapter interaction on 12 photographs", size=17, anchor="middle", bold=True)
    fig.text(
        75,
        66,
        "Branch-on gain with matched guidance: +0.514 dB",
        size=12,
        color=COLORS["LARP-Scaler"],
        bold=True,
    )
    rows = data["ablation_prompt_adapter_12_photos"]["rows"]
    prompts = ["Empty", "Detailed", "Detailed + photo"]
    x_positions = [190, 450, 710]
    y0, height, y_min, y_max = 100, 270, 31.2, 31.9
    axes(fig, 100, y0, 700, height, y_ticks=[31.2, 31.4, 31.6, 31.8], y_min=y_min, y_max=y_max)
    for adapter, color, shape in (
        ("Off", COLORS["neutral"], "circle"),
        ("Matched guidance", COLORS["LARP-Scaler"], "square"),
    ):
        values = [
            next(row["psnr_db"] for row in rows if row["prompt"] == prompt and row["adapter"] == adapter)
            for prompt in prompts
        ]
        points = [(x, y0 + height - (value - y_min) / (y_max - y_min) * height) for x, value in zip(x_positions, values)]
        fig.polyline(points, color=color, width=2.5)
        for (x, y), value in zip(points, values):
            if shape == "circle":
                fig.circle(x, y, 6, fill=color)
            else:
                fig.rect(x - 6, y - 6, 12, 12, fill=color, stroke=color)
            fig.text(x, y - 12, f"{value:.3f}", size=9, anchor="middle", bold=True)
    for x, prompt in zip(x_positions, prompts):
        fig.text(x, 400, prompt, size=11, anchor="middle")
    fig.circle(275, 445, 5, fill=COLORS["neutral"])
    fig.text(288, 449, "Adapter off", size=10)
    fig.rect(505, 440, 10, 10, fill=COLORS["LARP-Scaler"], stroke=COLORS["LARP-Scaler"])
    fig.text(522, 449, "Matched guidance", size=10)
    fig.save()


def dataset_composition(data: dict, output: Path) -> None:
    fig = Figure(
        output,
        "dataset_composition",
        900,
        510,
        "Dataset composition",
        "Horizontal bars show the number of records contributed by each of nine source datasets.",
    )
    fig.text(450, 30, "Training corpus composition — 288,423 records", size=17, anchor="middle", bold=True)
    rows = sorted(data["dataset"]["sources"], key=lambda row: row["rows"], reverse=True)
    max_value = max(row["rows"] for row in rows)
    x0, y0, max_width, row_height = 190, 70, 580, 43
    for index, row in enumerate(rows):
        y = y0 + index * row_height
        width = row["rows"] / max_value * max_width
        color = COLORS["LARP-Scaler"] if row["rows"] >= 20000 else "#93c5fd"
        fig.text(x0 - 14, y + 20, row["source"], size=10, anchor="end")
        fig.rect(x0, y + 5, width, 24, fill=color, stroke=color, radius=3)
        fig.text(x0 + width + 10, y + 22, f"{row['rows']:,}", size=10)
    fig.save()


def training_curve(data: dict, output: Path) -> None:
    fig = Figure(
        output,
        "training_curve",
        900,
        500,
        "Final direct-refinement training trace",
        "Logged training and validation losses over the 4000-update direct-refinement stage.",
    )
    fig.text(450, 30, "Final direct-refinement stage — Trackio milestone trace", size=17, anchor="middle", bold=True)
    rows = [row for row in data["milestones"] if row["validation_loss"] is not None]
    x0, y0, width, height = 95, 80, 720, 300
    y_min, y_max = 0.12, 0.20
    axes(
        fig,
        x0,
        y0,
        width,
        height,
        y_ticks=[0.12, 0.14, 0.16, 0.18, 0.20],
        y_min=y_min,
        y_max=y_max,
    )

    def point(row: dict, key: str) -> tuple[float, float]:
        x = x0 + (row["step"] - 100) / 3900 * width
        y = y0 + height - (row[key] - y_min) / (y_max - y_min) * height
        return x, y

    series = [
        ("train_loss", "Training loss", COLORS["neutral"], "circle"),
        ("validation_loss", "Validation loss", COLORS["LARP-Scaler"], "square"),
    ]
    for key, _, color, shape in series:
        points = [point(row, key) for row in rows]
        fig.polyline(points, color=color, width=2.5)
        for index, (x, y) in enumerate(points):
            if shape == "circle":
                fig.circle(x, y, 4, fill=color)
            else:
                fig.rect(x - 4, y - 4, 8, 8, fill=color, stroke=color)
            if index in (0, len(points) - 1):
                fig.text(x, y - 11, f"{rows[index][key]:.3f}", size=9, anchor="middle", bold=True)
    for step in [100, 1000, 2000, 3000, 4000]:
        x = x0 + (step - 100) / 3900 * width
        fig.text(x, y0 + height + 25, f"{step:,}", size=9, color=COLORS["muted"], anchor="middle")
    fig.text(455, 430, "Optimizer updates", size=11, anchor="middle", bold=True)
    fig.text(x0, 64, "Loss ↓", size=10, color=COLORS["muted"], bold=True)
    fig.circle(280, 468, 5, fill=COLORS["neutral"])
    fig.text(294, 472, "Training loss", size=10)
    fig.rect(500, 463, 10, 10, fill=COLORS["LARP-Scaler"], stroke=COLORS["LARP-Scaler"])
    fig.text(517, 472, "Validation loss", size=10)
    fig.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    training_data = json.loads(args.training_data.read_text(encoding="utf-8"))
    architecture(args.output)
    data_pipeline(args.output)
    quality(data, args.output)
    latency(data, args.output)
    ablation(data, args.output)
    dataset_composition(data, args.output)
    training_curve(training_data, args.output)
    print(f"Generated seven SVG/PDF/PNG figure sets in {args.output}")


if __name__ == "__main__":
    main()
