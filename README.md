# LARP-Scaler

**LAtent super-Resolution high-Performance image upscaler**

[![Model](https://img.shields.io/badge/🤗%20Model-LARP--Scaler-FFD21E)](https://huggingface.co/VladimirM388/larpscaler-v2-bf16)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vovanm88/LARP-Scaler/blob/main/notebooks/larpscaler_inference.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-Vovanm88%2FLARP--Scaler-181717?logo=github)](https://github.com/Vovanm88/LARP-Scaler)
[![License](https://img.shields.io/badge/code%20license-Apache--2.0-blue)](LICENSE)

> The lightweight `main` branch contains the runtime, examples, and notebooks.
> The complete preprint, benchmark records, publication scripts, and full-resolution
> qualitative assets are preserved in the
> [`paper-full` branch](https://github.com/Vovanm88/LARP-Scaler/tree/paper-full).

## Checkpoints

| Variant | Intended use | Status |
|---|---|---|
| **LARP-Scaler v2 BF16** | Current multi-scale ×2/×4/×8 release | [Available on Hugging Face](https://huggingface.co/VladimirM388/larpscaler-v2-bf16) |
| **LARP-Scaler FP8** | Lower-memory quantized inference | **To be released** |
| **LARP-Scaler NF4** | 4-bit lower-memory inference | **To be released** |
| **LARP-Scaler native 2048×2048** | Resolution-specialized 2K checkpoint | **To be released** |

Performance and memory figures for the planned variants will be published with
their weights; the benchmark results below describe the released BF16 model.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/larp_before_after_hero.png" width="100%" alt="LARP-Scaler before and after: a 512 by 512 real photo input and a LARP-Scaler 4x 2048 by 2048 output.">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/larp_pid_lua_image_upscale_examples.png" width="100%" alt="Image-upscaling examples: native PiD at 512 to 2048 pixels and LUA-FLUX at 256 to 1024 pixels, compared with Real-ESRGAN, LARP-Scaler, and ground truth.">
</p>

The multi-method comparison above uses two separately valid image-upscaling protocols: native
PiD operates at **512→2048**, while LUA-FLUX with the Flux VAE operates at
**256→1024**. They are visual examples rather than a shared metric ranking
across the two resolutions.

LARP-Scaler is a SANA-based image upscaler that directly refines a resized
image in a 32-channel latent space. It supports **×2, ×4, and ×8** enlargement,
uses an independent image-guidance adapter, and includes tiled VAE and DiT
inference for images that do not fit in a single forward pass.

> **Release status:** the public Gradio demo is available; the arXiv/Paper Page
> link remains pending.
> The current draft PDF and its sources live in the
> [`paper-full` branch](https://github.com/Vovanm88/LARP-Scaler/blob/paper-full/larpscaler.pdf).

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/architecture.svg" width="100%" alt="LARP-Scaler architecture: a resized image is refined by a SANA DiT while an independent image is encoded by a guidance adapter.">
</p>

## Highlights

- **Direct latent refinement.** A SANA transformer refines the latent encoding
  of a conventionally resized image instead of running a separate pixel-space
  restoration network.
- **Image-guided detail.** A three-stage guidance branch injects the original
  low-resolution image through zero-initialized gated cross-attention.
- **Text-encoder-free runtime.** Positive and empty prompt tensors are stored in
  the checkpoint; inference does not load a text encoder.
- **Practical large-image inference.** AutoencoderDC handles VAE tiling and the
  DiT uses overlapped latent tiles, Hann blending, shifted grids between steps,
  automatic tile batching, and CUDA OOM backoff.
- **Controlled reconstruction results.** On the 12-photo ×4 benchmark described
  below, LARP-Scaler reaches **31.817 dB PSNR**, compared with 29.641 dB for
  Real-ESRGAN and 19.554 dB for LUA. Protocol-matched Lanczos reaches 33.434
  dB, illustrating the expected distortion–generation trade-off.

## Links

| Resource | Link |
|---|---|
| Model weights | [VladimirM388/larpscaler-v2-bf16](https://huggingface.co/VladimirM388/larpscaler-v2-bf16) |
| Colab notebook | [Open `larpscaler_inference.ipynb`](https://colab.research.google.com/github/Vovanm88/LARP-Scaler/blob/main/notebooks/larpscaler_inference.ipynb) |
| Source code | [Vovanm88/LARP-Scaler](https://github.com/Vovanm88/LARP-Scaler) |
| Gradio demo | [Open the LARP-Scaler Space](https://huggingface.co/spaces/Anonumous/LARP-Scaler) |
| Paper | arXiv and Hugging Face Paper Page pending |

## Installation

Python 3.10 or later is required. A recent CUDA GPU is strongly recommended;
the released checkpoint is intended for BF16 inference.

```bash
pip install "git+https://github.com/Vovanm88/LARP-Scaler.git"
```

For local development:

```bash
git clone https://github.com/Vovanm88/LARP-Scaler.git
cd LARP-Scaler
pip install -e ".[dev,notebook]"
```

## Python API

### Legacy multi-step preset

The runtime supports the original expressive four-step configuration shown in
the example notebook. It is not the recommended paired-reconstruction setting:
additional steps were worse in internal checks for the released checkpoint.

```python
from larpscaler import LarpScaler

upscaler = LarpScaler.from_pretrained(
    "VladimirM388/larpscaler-v2-bf16",
)

image = upscaler.upscale(
    "input.png",
    scale=4,
    steps=4,
    noise_level=1.0,
    guidance_scale=4.5,
    seed=1234,
)
image.save("output.png")
```

### Evaluated reconstruction preset

For the fidelity-oriented setting measured in the paper, use one refinement
step:

```python
image = upscaler.upscale(
    "input.png",
    scale=4,
    steps=1,
    noise_level=0.35,
    guidance_scale=1.0,
    seed=1234,
)
image.save("output-fast.png")
```

### Custom guidance and large-image options

By default, the input image is also used as the guidance image. A different
image can be supplied for controlled ablations or specialized workflows:

```python
image = upscaler.upscale(
    "input.png",
    adapter_image="guidance.png",
    scale=8,
    steps=1,
    noise_level=0.35,
    guidance_scale=1.0,
    conditioning_scale=1.0,
    tile_size=1024,
    tile_overlap=256,
    tile_batch_size="auto",
)
```

Set `use_image_adapter=False` to disable the image-guidance branch. This keeps
the trained LARP-Scaler backbone; it does **not** restore the original upstream
SANA weights.

## Command line

Installing the package exposes the `larpscale` command:

```bash
larpscale input.png output.png \
  --model VladimirM388/larpscaler-v2-bf16 \
  --scale 4 \
  --steps 1 \
  --noise-level 0.35 \
  --guidance-scale 1.0
```

The CLI exposes the common inference controls. Use the Python API for custom
adapter images and explicit tile geometry.

## Controlled reconstruction quality

The following results use reference reconstruction rather than aesthetic
preference: a known 512×512 target is downsampled to 128×128 with Lanczos and
then reconstructed at ×4. Each domain contains 12 images. Higher PSNR and SSIM
are better; lower MAE is better.

### Real photographs

| Method | PSNR, dB ↑ | SSIM ↑ | MAE ↓ |
|---|---:|---:|---:|
| **Lanczos** | **33.4340** | **0.8308** | **0.01910** |
| Bicubic | 33.2358 | 0.8248 | 0.01943 |
| LARP-Scaler | 31.8170 | 0.7877 | 0.02314 |
| Real-ESRGAN ×4 | 29.6409 | 0.7587 | 0.02846 |
| LUA-Flux ×4 | 19.5536 | 0.4071 | 0.09247 |

### Anime images

| Method | PSNR, dB ↑ | SSIM ↑ | MAE ↓ |
|---|---:|---:|---:|
| **Lanczos** | **28.7939** | 0.8650 | **0.02184** |
| Bicubic | 28.3830 | 0.8614 | 0.02190 |
| LARP-Scaler | 28.4394 | 0.8560 | 0.02382 |
| Real-ESRGAN ×4 | 26.7850 | **0.8695** | 0.02559 |
| LUA-Flux ×4 | 17.2268 | 0.3653 | 0.10970 |

LARP-Scaler has the highest PSNR and lowest MAE among the tested learned
methods. Protocol-matched interpolation is stronger on paired distortion
metrics because the inputs are Lanczos downscales and interpolation does not
synthesize new texture. Real-ESRGAN has the highest anime SSIM. These values
should not be read as an aesthetic ranking.

**PiD is reported separately.** The public `PiD_res2k_sr4x` checkpoint targets
~2048 px output; run at 512 px it produces a regular lattice artifact, so it is
excluded from the tables above and measured under its native protocol instead:

### Native-resolution PiD comparison (512→2048, ×4, 12 photos)

| Method | PSNR, dB ↑ | SSIM ↑ | MAE ↓ |
|---|---:|---:|---:|
| **LARP-Scaler** | **31.4801** | **0.8114** | **0.02061** |
| Real-ESRGAN ×4 | 29.2505 | 0.7786 | 0.02571 |
| PiD-Flux (native 2k) | 26.1275 | 0.6683 | 0.03635 |
| LUA-Flux | — | — | Flux-VAE decode OOM at 2048 px on 32 GB |

At its native output scale PiD is artifact-free and valid; it still trails
LARP-Scaler and Real-ESRGAN on this strict paired metric.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/quality_comparison.svg" width="92%" alt="PSNR and SSIM comparisons for the photo and anime reconstruction sets.">
</p>

### Perceptual metrics (LPIPS / DISTS, lower is better)

| Domain | Method | LPIPS ↓ | DISTS ↓ |
|---|---|---:|---:|
| Photo | LARP-Scaler | 0.4221 | 0.1766 |
| Photo | Real-ESRGAN | **0.3778** | 0.1950 |
| Photo | Bicubic | 0.4551 | **0.1726** |
| Photo | Lanczos | 0.4718 | 0.1753 |
| Anime | LARP-Scaler | 0.2624 | **0.1215** |
| Anime | Real-ESRGAN | **0.1537** | **0.1215** |
| Anime | Bicubic | 0.2956 | 0.1253 |
| Anime | Lanczos | 0.2927 | 0.1268 |

Lanczos leads the distortion metrics but not the learned metrics: LARP-Scaler
has lower LPIPS on both domains, while Bicubic narrowly leads photo DISTS.
Real-ESRGAN remains best on LPIPS. No method dominates both exact
reconstruction and learned perceptual similarity.

### Qualitative comparison

Native-resolution outputs from the 12-photo protocol above. Every method sees
the same 512×512 LR input and produces a 2048×2048 output. PiD uses its official
`PiD_res2k_sr4x` four-step regime; it is not forced to generate an unsupported
512 px output. The four illustrative scenes cover different structures and
textures, while aggregate claims use all 12 images.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/qualitative_comparison.svg" width="100%" alt="Native 512-to-2048 qualitative comparison: LR input, Real-ESRGAN, PiD, LARP-Scaler and ground truth on four photographs.">
</p>

## End-to-end latency

Latency was measured on an NVIDIA GeForce RTX 5090 with models already in VRAM.
The interval includes file open, preprocessing, VAE encode when applicable,
inference, and final image conversion. It excludes model loading and writing the
result to disk.

The expanded test contains 24 real photographs, three warm-up calls and ten
timed calls per image, method, and scale: **240 timed calls per table cell**.
Inputs may be large native JPEG files, so the values represent end-to-end file
processing rather than GPU-only kernel time.

| Method | ×2 median, s ↓ | ×4 median, s ↓ | ×8 median, s ↓ |
|---|---:|---:|---:|
| LARP-Scaler | **0.8119** | 0.8227 | 0.8047 |
| LUA-Flux | 2.6323 | **0.6291** | 0.6668 |
| Real-ESRGAN ×4 | 1.0253 | 0.6681 | **0.6134** |

At ×4, LARP-Scaler is slower than LUA and Real-ESRGAN. The comparison therefore supports competitive latency, not a claim
of being universally fastest.

PiD is measured separately at its intended **512→2048 ×4** resolution:
**1.537 s median** across 30 timed calls on three photographs (3 warm-ups and
10 calls per image). This steady-state measurement includes input conversion,
Flux-VAE encoding, four PiD steps, and CPU/PIL conversion, but not model loading
or file writing. It is not mixed into the 1024-pixel ranking.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/latency_1024.svg" width="88%" alt="Median end-to-end latency at x2, x4 and x8 for LARP-Scaler, LUA, PiD and Real-ESRGAN.">
</p>

## Prompt and image-adapter ablation

The ablation uses the same 12 real photographs, 128→512 ×4 reconstruction, one
step, `noise_level=0.35`, `guidance_scale=1`, BF16, and seed 1234.

| Change | ΔPSNR, dB | Approx. MSE reduction |
|---|---:|---:|
| Detailed prompt, adapter off | +0.0166 | 0.38% |
| Detailed prompt, correct adapter | +0.0057 | 0.13% |
| Correct adapter, tagged prompt | **+0.5138** | **11.16%** |
| Correct vs. wrong guidance image | +0.0788 | 1.80% |

The trained backbone contributes most of the gain over upstream SANA: on these
same 12 photographs and the same one-step protocol, the released checkpoint
scores 31.8178 dB against 21.2628 dB for the untrained SANA backbone, a
**10.555 dB** difference. Within the trained model, the image adapter
contributes roughly half a decibel, while the cached prompt provides a smaller
positive change. These effects should not be added across different benchmark
protocols.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/prompt_adapter_ablation.svg" width="78%" alt="Prompt and image-adapter PSNR interaction.">
</p>

## Training corpus

The training metadata contains **288,423** captioned records from photographic,
art, and anime sources. Images were profiled at native resolution, embedded
with DINOv2-large, scored by two VLM-distilled student quality models, filtered with
content-aware deterministic gates, and captioned from source metadata or a
local VLM. A 2,500-image defect-labelled subset was retained as
quality-conditioned negative data.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/data_pipeline.svg" width="100%" alt="Six-stage LARP-Scaler data-curation pipeline.">
</p>

More detail and the exact source counts are included in the paper draft.

## Training recipe

LARP-Scaler was trained as a curriculum rather than converted from SANA in one
step:

1. **Domain and schedule adaptation.** Full-parameter CPT/SFT first moved SANA
   toward the curated training-image distribution and the target Z-Image flow
   schedule. No image adapter was present yet.
2. **Adapter warm-up.** A zero-initialized guidance branch was attached and
   trained jointly with the complete backbone. Half of the samples replaced
   the clean-side latent with a target-sized degraded latent while retaining
   the conventional noise-flow target.
3. **Adapter SFT.** Training continued on a stricter quality-filtered subset
   without the bad pool. `supups-upscaler-v1-sft/checkpoint-6000` was selected
   for the final transition.
4. **Direct refinement.** The path itself changed to 100% degraded-to-clean
   latent refinement and training continued for **4,000** updates with RGB and
   high-frequency reconstruction terms.

The adapter was never trained as an isolated plug-in: its 256-channel encoder,
scale-dependent ×2/×4/×8 upsamplers, ten image cross-attention insertions, and
the full SANA backbone were updated jointly. Zero-initialized gates made the
new branch a no-op when first attached and allowed its contribution to grow
gradually.

The logged objective was:

```text
loss = flow_loss + 0.3 × reconstruction_loss + 0.5 × high_frequency_loss
```

In the final stage, `flow_loss` is MSE on the velocity from the clean latent
toward the interpolated degraded latent. The predicted clean latent is decoded
at 512px through a frozen AutoencoderDC decoder; `reconstruction_loss` is RGB
L1 and `high_frequency_loss` is L1 between normalized 3×3 Laplacian responses.
Gradients pass through the decoder to the transformer and adapter, but the
decoder weights remain frozen.

This final stage used 1024px crops, BF16, DeepSpeed ZeRO-2, AdamW
(`lr=2e-5`, weight decay `0.01`), static loss scale 64, and an effective batch
size of **96**. VAE and text encoders remained frozen, and only the
transformer--adapter module received optimizer updates. Worker allocation,
cache production, seeds, and stage-boundary state are retained in the
machine-readable training record rather than duplicated here.

The final stage took **13 h 36 min 46 s**. Trackio validation loss decreased
from 0.1954 at update 100 to 0.1452 at update 4,000. This duration is not a
claim about total SANA pretraining or the complete predecessor-checkpoint
history.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vovanm88/LARP-Scaler/paper-full/paper/figures/training_curve.svg" width="80%" alt="Training and validation loss milestones across the final 4,000-update direct-refinement stage.">
</p>

The exact recovered config, run IDs, checkpoint hash, loss components, and
milestones are stored in
[`paper/data/training_run.json`](https://github.com/Vovanm88/LARP-Scaler/blob/paper-full/paper/data/training_run.json).

## Large-image inference

LARP-Scaler uses two distinct tiling systems:

1. **VAE tiling** is owned by `AutoencoderDC.enable_tiling`. The pipeline does
   not place a second manual VAE tiler around it.
2. **DiT tiling** runs in latent space. Overlapping predictions are blended
   with a two-dimensional Hann window and normalized by accumulated weights.

The DiT grid shifts by half a stride on alternating steps, reducing persistent
tile boundaries. `tile_batch_size="auto"` estimates a conservative batch size
from free accelerator memory, starts at no more than 16 tiles, and halves the
batch following CUDA out-of-memory errors. Latent accumulation and overlap
weights use FP32; model passes use the checkpoint dtype.

## Reproducing the figures

Publication assets are intentionally kept out of the lightweight branch.
Check out `paper-full` before running the commands below. Benchmark numbers are
stored in
[`paper/data/benchmarks.json`](https://github.com/Vovanm88/LARP-Scaler/blob/paper-full/paper/data/benchmarks.json);
final-stage training provenance and milestones are in
[`paper/data/training_run.json`](https://github.com/Vovanm88/LARP-Scaler/blob/paper-full/paper/data/training_run.json).

```bash
git switch paper-full
python paper/scripts/make_figures.py
```

This writes matching SVG files for GitHub and PDF files for LaTeX. The
qualitative montage is built separately from real benchmark outputs stored in
[`paper/qualitative/`](https://github.com/Vovanm88/LARP-Scaler/tree/paper-full/paper/qualitative):

```bash
python paper/scripts/make_qualitative.py paper/qualitative/manifest.json \
  --output paper/figures/qualitative_comparison.jpg
```

## Limitations

- The direct quality comparisons contain 12 photo and 12 anime images. They are
  useful controlled tests, but not a broad public benchmark. A separate
  48-photo matrix covers ×2, ×4, and ×8.
- PSNR, SSIM, and MAE reward exact reconstruction. They do not fully measure
  perceptual preference and can penalize plausible generated details.
- LPIPS and DISTS are reported; no-reference IQA, FID, and human preference are
  not yet available.
- Reported speed measurements use one GPU family: NVIDIA GeForce RTX 5090.
- LUA multi-scale matrices at ×2 and ×8, and its synthetic and mixed-aspect
  sets, are kept out of the shared ×4 tables; only the ×4 run on the identical
  12 images is merged.
- The recovered 13.6-hour training trace covers the final direct-refinement
  stage, not upstream SANA pretraining or the complete predecessor-checkpoint
  lineage.
- Access to computational resources was limited. Development, final-stage
  training, ablations, and evaluation used one four-GPU RTX 5090 machine,
  limiting broad hyperparameter sweeps, multi-seed training, and larger-scale
  evaluation.
- The arXiv and Hugging Face Paper Page URLs remain pending.

## Citation

Until the arXiv record is available, cite the software release:

```bibtex
@misc{melnikov2026larpscaler,
  title        = {LARP-Scaler: Latent Super-Resolution High-Performance Image Upscaler},
  author       = {Vladimir Melnikov and Maxim Manushin and Ilia Kuleshov},
  year         = {2026},
  howpublished = {\url{https://github.com/Vovanm88/LARP-Scaler}}
}
```

**Publication note:** replace this entry with the arXiv citation and add the
arXiv URL to this README once available so Hugging Face can associate the model
and demo with the corresponding Paper Page.

## License

The source code is released under the [Apache License 2.0](LICENSE). Model
weights and upstream components may carry their own terms; review the model
card and upstream licenses before redistribution.
