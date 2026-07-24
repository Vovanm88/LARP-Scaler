from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def canonical_pair_id(value: str) -> str:
    value = value.removesuffix("_low").removesuffix("_target")
    prefix, separator, remainder = value.partition("_")
    if separator and prefix.isdigit() and "-" in remainder:
        return remainder
    return value


def test_publication_text_has_no_template_content() -> None:
    paths = [ROOT / "README.md", PAPER / "main.tex", PAPER / "references.bib"]
    forbidden = (
        "lorem ipsum",
        "YOUR_GITHUB_USERNAME",
        "example2024",
        "exampleconference2023",
        "state-of-the-art",
        "state of the art",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    for phrase in forbidden:
        assert phrase.lower() not in combined


def test_notebook_is_valid_and_uses_public_repository() -> None:
    path = ROOT / "notebooks" / "larpscaler_inference.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    assert "https://github.com/Vovanm88/LARP-Scaler.git" in source
    assert "VladimirM388/larpscaler-v2-bf16" in source
    assert "YOUR_GITHUB_USERNAME" not in source


def test_benchmark_source_contains_release_claims() -> None:
    data = json.loads((PAPER / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    photo = {row["method"]: row for row in data["quality_x4_512"]["photo"]}
    anime = {row["method"]: row for row in data["quality_x4_512"]["anime"]}
    assert photo["LARP-Scaler"]["psnr_db"] == 31.8170
    assert photo["Real-ESRGAN"]["psnr_db"] == 29.6409
    assert photo["Lanczos"]["psnr_db"] == 33.4340
    assert photo["Bicubic"]["psnr_db"] == 33.2358
    assert anime["LARP-Scaler"]["psnr_db"] == 28.4394
    assert anime["Real-ESRGAN"]["ssim"] == 0.8695
    assert anime["Lanczos"]["mae"] == 0.02184
    # the invalid 512px PiD run is quarantined, not shown in the headline arrays
    assert "PiD" not in photo and "PiD" not in anime
    assert data["quality_x4_512"]["excluded_pid_512px"]["photo"]["psnr_db"] == 16.7071
    # the native-resolution 2k PiD comparison is a separate protocol
    native = {row["method"]: row for row in data["quality_native_2k"]["photo"]}
    assert native["LARP-Scaler"]["psnr_db"] == 31.4801
    assert native["PiD (native 2k)"]["psnr_db"] == 26.1275
    # perceptual metrics present for both domains
    perc = data["perceptual_512"]["results"]
    assert perc["photo"]["LARP-Scaler"]["dists"] < perc["photo"]["Real-ESRGAN"]["dists"]
    assert data["ablation_prompt_adapter_12_photos"]["reported_deltas"][
        "adapter_gain_tagged_prompt_db"
    ] == 0.513800
    assert data["dataset"]["total_rows"] == 288423
    multiscale = {
        (row["method"], row["scale"]): row
        for row in data["quality_multiscale_48_photo"]["rows"]
    }
    assert multiscale[("LARP-Scaler", 8)]["psnr_db"] == 29.1155
    assert multiscale[("Lanczos", 8)]["psnr_db"] == 29.2018


def test_revision_metrics_are_complete_and_self_consistent() -> None:
    revision = PAPER / "data" / "revision"
    summary = json.loads((revision / "revision_benchmarks.json").read_text(encoding="utf-8"))
    benchmarks = json.loads(
        (PAPER / "data" / "benchmarks.json").read_text(encoding="utf-8")
    )
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert summary["metadata"]["bootstrap_samples"] == 10_000
    assert summary["metadata"]["bootstrap_seed"] == 1234
    validation = summary["metadata"]["validation"]
    assert validation["controlled_x4"]["low_size"] == [128, 128]
    assert validation["controlled_x4"]["target_size"] == [512, 512]
    assert validation["multiscale_48_photo"]["target_size"] == [2048, 2048]

    with (revision / "controlled_x4_12_per_image.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        controlled = list(csv.DictReader(handle))
    assert len(controlled) == 2 * 5 * 12
    for domain in ("photo", "anime"):
        id_sets = {
                method: {
                    canonical_pair_id(row["id"])
                for row in controlled
                if row["domain"] == domain and row["method"] == method
            }
            for method in ("LARP-Scaler", "Real-ESRGAN", "LUA", "Bicubic", "Lanczos")
        }
        assert len({frozenset(ids) for ids in id_sets.values()}) == 1

    with (revision / "multiscale_48_photo_per_image.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        multiscale = list(csv.DictReader(handle))
    assert len(multiscale) == 3 * 3 * 48
    for scale in ("2", "4", "8"):
        id_sets = {
            method: {
                row["id"]
                for row in multiscale
                if row["scale"] == scale and row["method"] == method
            }
            for method in ("LARP-Scaler", "Bicubic", "Lanczos")
        }
        assert len({frozenset(ids) for ids in id_sets.values()}) == 1

    for group in summary["controlled_x4_12"]:
        rows = [
            row
            for row in controlled
            if row["domain"] == group["domain"] and row["method"] == group["method"]
        ]
        assert len(rows) == group["n"] == 12
        for metric in ("psnr", "ssim", "mae"):
            mean = sum(float(row[metric]) for row in rows) / len(rows)
            assert abs(mean - group[metric]) < 1e-12
            assert group[f"{metric}_ci95_low"] <= mean <= group[f"{metric}_ci95_high"]
        aggregate = next(
            row
            for row in benchmarks["quality_x4_512"][group["domain"]]
            if row["method"] == group["method"]
        )
        assert aggregate["psnr_db"] == round(group["psnr"], 4)
        assert aggregate["ssim"] == round(group["ssim"], 4)
        assert aggregate["mae"] == round(group["mae"], 5)
        assert f"{group['psnr']:.4f}" in tex
        assert f"{group['ssim']:.4f}" in tex
        assert f"{group['mae']:.5f}" in tex
        assert f"{group['psnr_ci95_low']:.3f}" in tex
        assert f"{group['psnr_ci95_high']:.3f}" in tex
        assert f"{group['ssim_ci95_low']:.3f}" in tex
        assert f"{group['ssim_ci95_high']:.3f}" in tex
        assert f"{group['mae_ci95_low']:.4f}" in tex
        assert f"{group['mae_ci95_high']:.4f}" in tex

    for group in summary["multiscale_48_photo"]:
        rows = [
            row
            for row in multiscale
            if row["scale"] == group["scale"] and row["method"] == group["method"]
        ]
        assert len(rows) == group["n"] == 48
        for metric in ("psnr", "ssim", "mae"):
            mean = sum(float(row[metric]) for row in rows) / len(rows)
            assert abs(mean - group[metric]) < 1e-12
            assert group[f"{metric}_ci95_low"] <= mean <= group[f"{metric}_ci95_high"]
        aggregate = next(
            row
            for row in benchmarks["quality_multiscale_48_photo"]["rows"]
            if row["scale"] == int(group["scale"])
            and row["method"] == group["method"]
        )
        assert aggregate["psnr_db"] == round(group["psnr"], 4)
        assert aggregate["ssim"] == round(group["ssim"], 4)
        assert aggregate["mae"] == round(group["mae"], 5)
        assert f"{group['psnr']:.4f}" in tex
        assert f"{group['ssim']:.4f}" in tex
        assert f"{group['mae']:.5f}" in tex
        assert f"{group['psnr_ci95_low']:.3f}" in tex
        assert f"{group['psnr_ci95_high']:.3f}" in tex
        assert f"{group['ssim_ci95_low']:.3f}" in tex
        assert f"{group['ssim_ci95_high']:.3f}" in tex
        assert f"{group['mae_ci95_low']:.4f}" in tex
        assert f"{group['mae_ci95_high']:.4f}" in tex


def test_training_source_matches_released_run() -> None:
    data = json.loads((PAPER / "data" / "training_run.json").read_text(encoding="utf-8"))
    assert data["provenance"]["trackio_run_id"] == "36b004bca18e41c89ec1112340d5f930"
    assert data["provenance"]["released_transformer_sha256"] == (
        "c0496dd040a34adced02ab8bd1c6f2d12617f9d3ed8b80793d2e4796747ba830"
    )
    assert data["configuration"]["effective_batch_size"] == 96
    assert data["configuration"]["updates"] == 4000
    assert [stage["name"] for stage in data["curriculum"]["stages"]] == [
        "domain_and_schedule_adaptation",
        "adapter_warmup_v1_cpt",
        "adapter_v1_sft",
        "direct_refinement",
    ]
    assert data["curriculum"]["adapter"]["guidance_attention_blocks"] == 10
    assert data["curriculum"]["stage_boundary_state"]["optimizer_state_restored"] is False
    final = data["milestones"][-1]
    assert final["validation_loss"] == 0.14520353078842163
    reconstructed_loss = (
        final["flow_loss"]
        + 0.3 * final["reconstruction_loss"]
        + 0.5 * final["high_frequency_loss"]
    )
    assert abs(reconstructed_loss - final["train_loss"]) < 1e-8


def test_all_citations_have_bibliography_entries() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    entries = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert cited <= entries


def test_tex_has_no_unresolved_release_markers_or_dropped_wide_floats() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "??" not in tex
    assert "\\todo" not in tex
    assert "\\begin{table*}[H]" not in tex
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    references = {
        key.strip()
        for group in re.findall(r"\\(?:c|C)?ref\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    assert references <= labels
    assert "\\label{tab:controlled_ci}" in tex
    assert "\\label{tab:multiscale_ci}" in tex
    assert "\\label{app:implementation}" in tex
    assert "Correct image" not in tex
    assert "Wrong image" not in tex


def test_required_figure_sources_and_raster_assets_exist() -> None:
    names = {
        "architecture",
        "data_pipeline",
        "quality_comparison",
        "latency_1024",
        "prompt_adapter_ablation",
        "dataset_composition",
        "training_curve",
    }
    for name in names:
        svg = PAPER / "figures" / f"{name}.svg"
        raster = PAPER / "figures" / f"{name}.png"
        assert svg.is_file() and svg.stat().st_size > 500
        assert raster.is_file() and raster.stat().st_size > 500
        assert "<title" in svg.read_text(encoding="utf-8")


def test_tex_figure_paths_exist_or_are_explicitly_optional() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert ".pdf}" not in tex
    paths = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", tex)
    for relative in paths:
        assert (PAPER / relative).is_file(), relative


def test_readme_has_release_links_and_explicit_pending_state() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/VladimirM388/larpscaler-v2-bf16" in readme
    assert (
        "https://colab.research.google.com/github/Vovanm88/LARP-Scaler/"
        "blob/main/notebooks/larpscaler_inference.ipynb"
    ) in readme
    assert "https://huggingface.co/spaces/Anonumous/LARP-Scaler" in readme
    assert "Paper | arXiv and Hugging Face Paper Page pending" in readme


if __name__ == "__main__":
    checks = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for check in checks:
        check()
    print(f"{len(checks)} publication checks passed")
