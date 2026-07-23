from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file, save_file

from .model import UpscalerSanaTransformer2DModel


# The pipeline intentionally produces very large images (for example 4K x4).
Image.MAX_IMAGE_PIXELS = None


REFINE_SCHEDULES = ("linear", "geometric", "cosine", "late", "early", "karras")
ResampleMode = Literal["lanczos", "bicubic"]
CHECKPOINT_PATTERNS = (
    "conditioning.safetensors",
    "export_manifest.json",
    "scheduler/*",
    "transformer/*",
    "vae/*",
    "vae_decoder/*",
)


def _resolve_checkpoint(
    checkpoint: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    """Return a local snapshot directory for either a path or Hub model ID."""
    candidate = Path(checkpoint).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise ValueError(f"Checkpoint must be a directory: {candidate}")
    downloaded = snapshot_download(
        repo_id=str(checkpoint),
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        token=token,
        local_files_only=local_files_only,
        force_download=force_download,
        allow_patterns=list(CHECKPOINT_PATTERNS),
    )
    return Path(downloaded)


def _validate_checkpoint(checkpoint: Path) -> None:
    required = (
        checkpoint / "conditioning.safetensors",
        checkpoint / "scheduler" / "scheduler_config.json",
        checkpoint / "transformer" / "config.json",
    )
    missing = [str(path.relative_to(checkpoint)) for path in required if not path.is_file()]
    transformer_weights = tuple((checkpoint / "transformer").glob("*.safetensors"))
    if not transformer_weights:
        missing.append("transformer/*.safetensors")
    has_vae = (checkpoint / "vae" / "config.json").is_file()
    has_decoder = (checkpoint / "vae_decoder" / "diffusion_pytorch_model.safetensors").is_file()
    if not has_vae and not has_decoder:
        missing.append("vae/ or vae_decoder/diffusion_pytorch_model.safetensors")
    if missing:
        raise FileNotFoundError(
            f"Incomplete LarpScaler checkpoint at {checkpoint}. Missing: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class Conditioning:
    """Precomputed prompt tensors; this replaces the text encoder at runtime."""

    prompt_embeds: torch.Tensor
    prompt_mask: torch.Tensor
    empty_prompt_embeds: torch.Tensor | None = None
    empty_prompt_mask: torch.Tensor | None = None

    @classmethod
    def from_safetensors(cls, path: str | Path) -> "Conditioning":
        tensors = load_file(str(path), device="cpu")
        required = ("prompt_embeds", "prompt_mask")
        missing = [name for name in required if name not in tensors]
        if missing:
            raise ValueError(f"Conditioning file is missing: {', '.join(missing)}")
        positive = tensors["prompt_embeds"]
        positive_mask = tensors["prompt_mask"]
        if positive.ndim != 3 or positive_mask.ndim != 2:
            raise ValueError("prompt_embeds must be [batch, sequence, channels] and prompt_mask [batch, sequence]")
        empty_embed = tensors.get("empty_prompt_embeds")
        empty_mask = tensors.get("empty_prompt_mask")
        if (empty_embed is None) != (empty_mask is None):
            raise ValueError("Empty conditioning requires both empty_prompt_embeds and empty_prompt_mask")
        if empty_embed is not None and (empty_embed.ndim != 3 or empty_mask.ndim != 2):
            raise ValueError("Empty conditioning tensors have invalid dimensions")
        return cls(positive, positive_mask, empty_embed, empty_mask)

    def save_safetensors(self, path: str | Path) -> None:
        """Write portable conditioning tensors prepared by any offline process."""
        tensors = {
            "prompt_embeds": self.prompt_embeds.contiguous().cpu(),
            "prompt_mask": self.prompt_mask.contiguous().cpu(),
        }
        if self.empty_prompt_embeds is not None and self.empty_prompt_mask is not None:
            tensors["empty_prompt_embeds"] = self.empty_prompt_embeds.contiguous().cpu()
            tensors["empty_prompt_mask"] = self.empty_prompt_mask.contiguous().cpu()
        save_file(tensors, str(path))


def refine_sigma_schedule(noise_level: float, num_inference_steps: int, schedule: str) -> torch.Tensor:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if not 0 < noise_level <= 1:
        raise ValueError("noise_level must be in (0, 1]")
    if schedule not in REFINE_SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}; expected one of {REFINE_SCHEDULES}")
    if num_inference_steps == 1:
        return torch.tensor([noise_level], dtype=torch.float32)
    end = noise_level / num_inference_steps
    progress = torch.linspace(0, 1, num_inference_steps, dtype=torch.float32)
    if schedule == "linear":
        return torch.linspace(noise_level, end, num_inference_steps)
    if schedule == "geometric":
        return noise_level * (end / noise_level) ** progress
    if schedule == "cosine":
        return end + (noise_level - end) * torch.cos(progress * torch.pi / 2)
    if schedule == "late":
        return end + (noise_level - end) * (1 - progress.square())
    if schedule == "early":
        return end + (noise_level - end) * (1 - progress.sqrt())
    rho = 7.0
    return (noise_level ** (1 / rho) + progress * (end ** (1 / rho) - noise_level ** (1 / rho))) ** rho


def _unshift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    return sigmas / (shift - (shift - 1) * sigmas)


class LarpScaler:
    """Text-encoder-free direct-refinement pipeline for LarpScaler checkpoints."""

    def __init__(
        self,
        transformer: UpscalerSanaTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderDC,
        conditioning: Conditioning,
        *,
        device: str | torch.device = "cuda",
        vae_tiling: bool = True,
    ):
        self.device = torch.device(device)
        self.transformer = transformer.requires_grad_(False).to(self.device).eval()
        self.scheduler = scheduler
        self.vae = vae.requires_grad_(False).to(self.device, dtype=self._dtype).eval()
        # AutoencoderDC owns its own tiled encode/decode geometry.  Do not
        # manually tile around it: the two blend/crop schemes use different
        # coordinate conventions and can shift large reconstructed images.
        self.vae_tiling = vae_tiling
        self.conditioning = conditioning

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        vae_model: str | None = None,
        conditioning_path: str | Path | None = None,
        device: str | torch.device | None = None,
        vae_tiling: bool = True,
        dtype: torch.dtype | None = None,
        revision: str | None = None,
        variant: str | None = None,
        vae_revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> "LarpScaler":
        checkpoint = _resolve_checkpoint(
            checkpoint,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        _validate_checkpoint(checkpoint)
        manifest_path = checkpoint / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        if dtype is None:
            checkpoint_dtype = str(
                manifest.get("dtype") or manifest.get("variant") or ""
            ).lower()
            if device.type != "cuda" or checkpoint_dtype in ("fp32", "float32"):
                dtype = torch.float32
            else:
                dtype = torch.bfloat16
        vae_model = vae_model or manifest.get("base_model")
        if not vae_model:
            raise ValueError("vae_model is required when export_manifest.json has no base_model")
        if conditioning_path is None:
            conditioning_path = checkpoint / "conditioning.safetensors"
        else:
            conditioning_path = Path(conditioning_path).expanduser()
            if not conditioning_path.is_absolute() and not conditioning_path.is_file():
                conditioning_path = checkpoint / conditioning_path
        transformer = UpscalerSanaTransformer2DModel.from_pretrained(
            checkpoint / "transformer", torch_dtype=dtype
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint / "scheduler")
        local_vae = checkpoint / "vae"
        if local_vae.is_dir():
            vae = AutoencoderDC.from_pretrained(local_vae, torch_dtype=dtype)
        else:
            vae = AutoencoderDC.from_pretrained(
                vae_model, subfolder="vae", revision=vae_revision,
                variant=variant or manifest.get("variant"), torch_dtype=dtype,
            )
            decoder_path = checkpoint / "vae_decoder" / "diffusion_pytorch_model.safetensors"
            if decoder_path.is_file():
                vae.decoder.load_state_dict(load_file(str(decoder_path), device="cpu"))
        return cls(
            transformer, scheduler, vae, Conditioning.from_safetensors(conditioning_path),
            device=device, vae_tiling=vae_tiling,
        )

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @property
    def _vae_stride(self) -> int:
        configured = getattr(self.vae.config, "spatial_compression_ratio", None)
        if configured:
            return int(configured)
        blocks = getattr(self.vae.config, "block_out_channels", None)
        return 2 ** (len(blocks) - 1) if blocks else 32

    def _conditioning(self, guidance_scale: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
        positive = self.conditioning.prompt_embeds.to(self.device, dtype=self._dtype)
        positive_mask = self.conditioning.prompt_mask.to(self.device)
        if guidance_scale <= 1:
            return positive, positive_mask, False
        negative = self.conditioning.empty_prompt_embeds
        negative_mask = self.conditioning.empty_prompt_mask
        if negative is None or negative_mask is None:
            raise ValueError("guidance_scale > 1 requires empty prompt tensors in conditioning.safetensors")
        sequence_length = positive.shape[1]
        negative = negative[:, :sequence_length]
        negative_mask = negative_mask[:, :sequence_length]
        if negative.shape[1] < sequence_length:
            negative = F.pad(negative, (0, 0, 0, sequence_length - negative.shape[1]))
            negative_mask = F.pad(negative_mask, (0, sequence_length - negative_mask.shape[1]))
        return (
            torch.cat([negative.to(self.device, dtype=self._dtype), positive], dim=0),
            torch.cat([negative_mask.to(self.device), positive_mask], dim=0),
            True,
        )

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1)
        return pixels.unsqueeze(0).float().div(127.5).sub(1)

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        pixels = tensor[0].float().add(1).div(2).clamp(0, 1).mul(255).round().byte()
        return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy(), "RGB")

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        pixels = self._image_to_tensor(image).to(self.device, dtype=self._dtype)
        latents = self.vae.encode(pixels).latent * self.vae.config.scaling_factor
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(
            latents.to(dtype=self._dtype) / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]

    def _configure_vae_tiling(self, tile_size: int, overlap: int) -> None:
        if not self.vae_tiling or not hasattr(self.vae, "enable_tiling"):
            return
        stride = tile_size - overlap
        self.vae.enable_tiling(
            tile_sample_min_height=tile_size,
            tile_sample_min_width=tile_size,
            tile_sample_stride_height=stride,
            tile_sample_stride_width=stride,
        )

    @staticmethod
    def _tile_positions(length: int, tile: int, overlap: int, step_offset: int = 0) -> list[int]:
        if tile < 1 or not 0 <= overlap < tile:
            raise ValueError("tile size must be positive and overlap must be smaller than it")
        if length <= tile:
            return [0]
        stride = tile - overlap
        positions = [0]
        positions.extend(range(step_offset, length - tile + 1, stride))
        positions.append(length - tile)
        return sorted(set(max(0, min(position, length - tile)) for position in positions))

    @staticmethod
    def _blend_window(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = torch.hann_window(height, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
        column = torch.hann_window(width, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
        return row[:, None] * column[None, :]

    def _encode_tiled_image(self, image: Image.Image, tile_size: int, overlap: int) -> torch.Tensor:
        self._configure_vae_tiling(tile_size, overlap)
        return self._encode_image(image)

    def _decode_tiled_latents(self, latents: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
        self._configure_vae_tiling(tile_size, overlap)
        return self._decode_latents(latents)

    def _auto_tile_batch_size(self, tile_count: int, do_cfg: bool) -> int:
        """Choose a conservative DiT tile batch from currently free accelerator memory."""
        if self.device.type != "cuda":
            return 1
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        # Includes DiT activations and the extra conditional batch used by CFG.
        bytes_per_tile = 768 * 1024**2 * (2 if do_cfg else 1)
        budget = int(free_bytes * 0.5)
        return max(1, min(tile_count, 16, budget // bytes_per_tile))

    @torch.inference_mode()
    def upscale(
        self,
        image: Image.Image | str | Path,
        *,
        scale: int = 4,
        steps: int = 4,
        noise_level: float = 1.0,
        guidance_scale: float = 4.5,
        conditioning_scale: float = 1.0,
        schedule: str = "linear",
        seed: int = 1234,
        resample: ResampleMode = "lanczos",
        tile_size: int = 1024,
        tile_overlap: int = 256,
        tile_batch_size: int | str = "auto",
        adapter_image: Image.Image | str | Path | None = None,
        use_image_adapter: bool = True,
    ) -> Image.Image:
        if scale not in (2, 4, 8):
            raise ValueError("scale must be one of 2, 4, or 8")
        if conditioning_scale < 0:
            raise ValueError("conditioning_scale must be non-negative")
        if tile_size % self._vae_stride or tile_overlap % self._vae_stride:
            raise ValueError("tile_size and tile_overlap must be divisible by the VAE stride")
        if tile_batch_size != "auto" and (not isinstance(tile_batch_size, int) or tile_batch_size < 1):
            raise ValueError("tile_batch_size must be a positive integer or 'auto'")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image or an image path")
        if not use_image_adapter and adapter_image is not None:
            raise ValueError("adapter_image requires use_image_adapter=True")
        if isinstance(adapter_image, (str, Path)):
            with Image.open(adapter_image) as opened:
                adapter_image = opened.convert("RGB")
        if adapter_image is not None and not isinstance(adapter_image, Image.Image):
            raise TypeError("adapter_image must be a PIL.Image.Image or an image path")
        target_size = (image.width * scale, image.height * scale)
        multiple = self._vae_stride * scale
        padded_size = (
            math.ceil(target_size[0] / multiple) * multiple,
            math.ceil(target_size[1] / multiple) * multiple,
        )
        interpolation = Image.Resampling.LANCZOS if resample == "lanczos" else Image.Resampling.BICUBIC
        if resample not in ("lanczos", "bicubic"):
            raise ValueError("resample must be 'lanczos' or 'bicubic'")
        degraded = image.resize(padded_size, interpolation)
        guidance_size = (padded_size[0] // scale, padded_size[1] // scale)
        guidance = (adapter_image or image).resize(guidance_size, interpolation)
        degraded_latents = self._encode_tiled_image(degraded, tile_size, tile_overlap)
        guidance_latents = None
        if use_image_adapter:
            guidance_latents = self._encode_tiled_image(guidance, tile_size, tile_overlap) * conditioning_scale
        prompt_embeds, prompt_mask, do_cfg = self._conditioning(guidance_scale)
        effective_sigmas = refine_sigma_schedule(noise_level, steps, schedule)
        shift = float(getattr(self.scheduler.config, "shift", 1.0))
        self.scheduler.set_timesteps(sigmas=_unshift_sigmas(effective_sigmas, shift).tolist(), device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = degraded_latents.float()
        factors = torch.tensor([scale], device=self.device)
        timestep_scale = getattr(self.transformer.config, "timestep_scale", 1.0)
        latent_tile = tile_size // self._vae_stride
        latent_overlap = tile_overlap // self._vae_stride
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            offset = 0 if step_index % 2 == 0 else (latent_tile - latent_overlap) // 2
            next_latents = torch.zeros_like(latents)
            weights = torch.zeros_like(latents[:, :1])
            positions = [
                (top, left, min(top + latent_tile, latents.shape[-2]), min(left + latent_tile, latents.shape[-1]))
                for top in self._tile_positions(latents.shape[-2], latent_tile, latent_overlap, offset)
                for left in self._tile_positions(latents.shape[-1], latent_tile, latent_overlap, offset)
            ]
            batch_size = self._auto_tile_batch_size(len(positions), do_cfg) if tile_batch_size == "auto" else tile_batch_size
            index = 0
            while index < len(positions):
                current_size = min(batch_size, len(positions) - index)
                batch = positions[index : index + current_size]
                try:
                    latent_batch = torch.cat([latents[:, :, top:bottom, left:right] for top, left, bottom, right in batch])
                    guidance_batch = None
                    if guidance_latents is not None:
                        guidance_batch = torch.cat([
                            guidance_latents[:, :, top // scale:bottom // scale, left // scale:right // scale]
                            for top, left, bottom, right in batch
                        ])
                    if do_cfg:
                        model_input = torch.cat([latent_batch, latent_batch])
                        model_guidance = None if guidance_batch is None else torch.cat([guidance_batch, guidance_batch])
                        model_embeds = torch.cat([prompt_embeds[:1].expand(current_size, -1, -1), prompt_embeds[1:].expand(current_size, -1, -1)])
                        model_masks = torch.cat([prompt_mask[:1].expand(current_size, -1), prompt_mask[1:].expand(current_size, -1)])
                        model_factors = factors.expand(current_size).repeat(2)
                    else:
                        model_input, model_guidance = latent_batch, guidance_batch
                        model_embeds = prompt_embeds.expand(current_size, -1, -1)
                        model_masks = prompt_mask.expand(current_size, -1)
                        model_factors = factors.expand(current_size)
                    prediction = self.transformer(
                        model_input.to(dtype=self._dtype), encoder_hidden_states=model_embeds,
                        encoder_attention_mask=model_masks,
                        timestep=timestep.expand(model_input.shape[0]) * timestep_scale,
                        guidance_latents=None if model_guidance is None else model_guidance.to(dtype=self._dtype),
                        downsample_factors=model_factors, return_dict=False,
                    )[0].float()
                    if do_cfg:
                        unconditional, conditional = prediction.chunk(2)
                        prediction = unconditional + guidance_scale * (conditional - unconditional)
                    if prediction.shape[1] == latent_batch.shape[1] * 2:
                        prediction = prediction.chunk(2, dim=1)[0]
                    if hasattr(self.scheduler, "_step_index"):
                        self.scheduler._step_index = step_index
                    updated = self.scheduler.step(
                        prediction, timestep, latent_batch, generator=generator, return_dict=False
                    )[0]
                except torch.OutOfMemoryError:
                    if current_size == 1:
                        raise
                    torch.cuda.empty_cache()
                    batch_size = max(1, current_size // 2)
                    continue
                for tile_index, (top, left, bottom, right) in enumerate(batch):
                    tile = updated[tile_index : tile_index + 1]
                    window = self._blend_window(*tile.shape[-2:], device=self.device, dtype=tile.dtype)
                    next_latents[:, :, top:bottom, left:right] += tile * window
                    weights[:, :, top:bottom, left:right] += window
                index += current_size
            latents = next_latents / weights.clamp_min(1e-6)
        decoded = self._decode_tiled_latents(latents, tile_size, tile_overlap)
        return self._tensor_to_image(decoded).crop((0, 0, *target_size))
