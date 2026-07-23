from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import SanaTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch import nn


class GuidanceUpsampleStage(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.post = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv2(F.silu(self.conv1(hidden_states))) + residual
        hidden_states = F.interpolate(
            hidden_states, scale_factor=2, mode="bilinear", align_corners=False
        )
        return self.post(F.silu(hidden_states))


class AdaZeroGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.proj(F.silu(conditioning)).unsqueeze(1)


class AdaZeroConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 1)
        self.ada_scale = nn.Linear(dim, dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale = 1 + self.ada_scale(F.silu(conditioning)).unsqueeze(-1).unsqueeze(-1)
        return self.conv(hidden_states) * scale


class UpscalerSanaTransformer2DModel(SanaTransformer2DModel):
    """Sana DiT with the guidance adapters stored in LarpScaler checkpoints."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int | None = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: int | None = 20,
        cross_attention_head_dim: int | None = 112,
        cross_attention_dim: int | None = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: int | None = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: str | None = None,
        timestep_scale: float = 1.0,
        guidance_channels: int = 256,
        guidance_attention_every: int = 2,
    ):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
            num_layers=num_layers, num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim, caption_channels=caption_channels,
            mlp_ratio=mlp_ratio, dropout=dropout, attention_bias=attention_bias,
            sample_size=sample_size, patch_size=patch_size,
            norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps,
            interpolation_scale=interpolation_scale, guidance_embeds=guidance_embeds,
            guidance_embeds_scale=guidance_embeds_scale, qk_norm=qk_norm,
            timestep_scale=timestep_scale,
        )
        if guidance_attention_every < 1:
            raise ValueError("guidance_attention_every must be positive")
        inner_dim = num_attention_heads * attention_head_dim
        cross_heads = num_cross_attention_heads or num_attention_heads
        cross_head_dim = cross_attention_head_dim or attention_head_dim
        self.guidance_in = nn.Conv2d(in_channels, guidance_channels, 3, padding=1)
        self.guidance_upsamplers = nn.ModuleList(
            [GuidanceUpsampleStage(guidance_channels) for _ in range(3)]
        )
        self.guidance_projection = nn.Conv2d(guidance_channels, inner_dim, 1)
        selected = range(0, num_layers, guidance_attention_every)
        self.guidance_attentions = nn.ModuleDict({
            str(index): Attention(query_dim=inner_dim, cross_attention_dim=inner_dim,
                                  heads=cross_heads, dim_head=cross_head_dim,
                                  dropout=dropout, bias=True)
            for index in selected
        })
        self.guidance_gates = nn.ModuleDict({
            str(index): AdaZeroGate(inner_dim) for index in selected
        })
        self.guidance_ff_injection = AdaZeroConv(inner_dim)

    def _guidance_features(
        self, guidance_latents: torch.Tensor, downsample_factors: torch.Tensor,
        target_height: int, target_width: int,
    ) -> torch.Tensor:
        batch = guidance_latents.shape[0]
        result = None
        for factor_tensor in downsample_factors.unique(sorted=True):
            factor = int(factor_tensor.item())
            if factor not in (2, 4, 8):
                raise ValueError(f"Unsupported guidance downsample factor: {factor}")
            positions = torch.where(downsample_factors == factor_tensor)[0]
            source = guidance_latents[
                positions, :, : target_height // factor, : target_width // factor
            ]
            features = self.guidance_in(source)
            for stage in self.guidance_upsamplers[: int(math.log2(factor))]:
                features = stage(features)
            if features.shape[-2:] != (target_height, target_width):
                features = F.interpolate(
                    features, size=(target_height, target_width), mode="bilinear",
                    align_corners=False,
                )
            features = self.guidance_projection(features)
            if result is None:
                result = features.new_empty(batch, features.shape[1], target_height, target_width)
            result[positions] = features
        if result is None:
            raise ValueError("Guidance batch cannot be empty")
        return result

    def _forward_block(
        self, block: nn.Module, index: int, hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None, encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None, timestep: torch.Tensor,
        embedded_timestep: torch.Tensor, guidance_map: torch.Tensor | None, height: int, width: int,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = block.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + gate_msa * block.attn1(norm_hidden_states.to(hidden_states.dtype))
        if block.attn2 is not None:
            hidden_states = hidden_states + block.attn2(
                hidden_states, encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
        key = str(index)
        if guidance_map is not None and key in self.guidance_attentions:
            guidance_tokens = guidance_map.flatten(2).transpose(1, 2)
            guidance_output = self.guidance_attentions[key](
                hidden_states, encoder_hidden_states=guidance_tokens
            )
            hidden_states = hidden_states + self.guidance_gates[key](guidance_output, embedded_timestep)
        norm_hidden_states = block.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        if guidance_map is not None and index == 0:
            norm_hidden_states = norm_hidden_states + self.guidance_ff_injection(
                guidance_map, embedded_timestep
            )
        ff_output = block.ff(norm_hidden_states).flatten(2, 3).permute(0, 2, 1)
        return hidden_states + gate_mlp * ff_output

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor, guidance_latents: torch.Tensor | None,
        downsample_factors: torch.Tensor, guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: tuple[torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor, ...] | Transformer2DModelOutput:
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)).unsqueeze(1) * -10000.0
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)).unsqueeze(1) * -10000.0
        batch_size, _, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        post_height, post_width = height // patch_size, width // patch_size
        hidden_states = self.patch_embed(hidden_states)
        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)
        guidance_map = None
        if guidance_latents is not None:
            guidance_map = self._guidance_features(
                guidance_latents, downsample_factors, post_height, post_width
            )
        for index, block in enumerate(self.transformer_blocks):
            args = (block, index, hidden_states, attention_mask, encoder_hidden_states,
                    encoder_attention_mask, timestep, embedded_timestep, guidance_map,
                    post_height, post_width)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(self._forward_block, *args)
            else:
                hidden_states = self._forward_block(*args)
            if controlnet_block_samples is not None and 0 < index <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index - 1]
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_height, post_width, patch_size, patch_size, -1)
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_height * patch_size, post_width * patch_size)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
