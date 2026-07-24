from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


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
    assert photo["PiD"]["psnr_db"] == 16.7071
    assert anime["LARP-Scaler"]["psnr_db"] == 28.4394
    assert anime["Real-ESRGAN"]["ssim"] == 0.8695
    assert data["ablation_prompt_adapter_12_photos"]["reported_deltas"][
        "adapter_gain_tagged_prompt_db"
    ] == 0.513800
    assert data["dataset"]["total_rows"] == 288423


def test_training_source_matches_released_run() -> None:
    data = json.loads((PAPER / "data" / "training_run.json").read_text(encoding="utf-8"))
    assert data["provenance"]["trackio_run_id"] == "36b004bca18e41c89ec1112340d5f930"
    assert data["provenance"]["released_transformer_sha256"] == (
        "c0496dd040a34adced02ab8bd1c6f2d12617f9d3ed8b80793d2e4796747ba830"
    )
    assert data["configuration"]["effective_batch_size"] == 96
    assert data["configuration"]["updates"] == 4000
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


def test_required_figure_pairs_exist() -> None:
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
        pdf = PAPER / "figures" / f"{name}.pdf"
        assert svg.is_file() and svg.stat().st_size > 500
        assert pdf.is_file() and pdf.stat().st_size > 500
        assert "<title" in svg.read_text(encoding="utf-8")


def test_tex_figure_paths_exist_or_are_explicitly_optional() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    paths = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", tex)
    for relative in paths:
        if relative == "figures/qualitative_comparison.pdf":
            assert "\\IfFileExists{figures/qualitative_comparison.pdf}" in tex
        else:
            assert (PAPER / relative).is_file(), relative


def test_readme_has_release_links_and_explicit_todos() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/VladimirM388/larpscaler-v2-bf16" in readme
    assert (
        "https://colab.research.google.com/github/Vovanm88/LARP-Scaler/"
        "blob/main/notebooks/larpscaler_inference.ipynb"
    ) in readme
    assert "Gradio demo | **TODO:**" in readme
    assert "Paper | **TODO:**" in readme


if __name__ == "__main__":
    checks = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for check in checks:
        check()
    print(f"{len(checks)} publication checks passed")
