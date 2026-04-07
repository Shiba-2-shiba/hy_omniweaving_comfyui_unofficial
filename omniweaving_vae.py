from __future__ import annotations

from contextlib import contextmanager
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.ops
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

ops = comfy.ops.disable_weight_init


def swish(x, inplace: bool = False):
    return F.silu(x, inplace=inplace)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, images: bool):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones((dim, *broadcastable_dims)))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride=1, dilation=1, pad_mode="replicate", disable_causal=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kt = kernel_size[0]
        kh = kernel_size[1]
        kw = kernel_size[2]
        if disable_causal:
            padding = (kw // 2, kw // 2, kh // 2, kh // 2, kt // 2, kt // 2)
        else:
            padding = (kw // 2, kw // 2, kh // 2, kh // 2, kt - 1, 0)
        self.time_causal_padding = padding
        self.pad_mode = pad_mode
        self.conv = ops.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


def _prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = RMSNorm(in_channels, images=False)
        self.q = ops.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = ops.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = ops.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = ops.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, f, h_size, w_size = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = _prepare_causal_attention_mask(f, h_size * w_size, x.dtype, x.device, batch_size=b)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))
        h = rearrange(h, "b 1 (f h w) c -> b c f h w", b=b, c=c, f=f, h=h_size, w=w_size)
        return x + self.proj_out(h)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = RMSNorm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = RMSNorm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = ops.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(swish(self.norm1(x), inplace=True))
        h = self.conv2(swish(self.norm2(h), inplace=True))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 8 if add_temporal_downsample else 4
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            x_first = x[:, :, :1]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            b, _, t, h_size, w_size = x_first.shape
            x_first = x_first.view(b, h.shape[1], self.group_size // 2, t, h_size, w_size).mean(dim=2)

            x_next = x[:, :, 1:]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            b, _, t, h_size, w_size = x_next.shape
            x_next = x_next.view(b, h.shape[1], self.group_size, t, h_size, w_size).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            b, _, t, h_size, w_size = shortcut.shape
            shortcut = shortcut.view(b, h.shape[1], self.group_size, t, h_size, w_size).mean(dim=2)
        return h + shortcut


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 8 if add_temporal_upsample else 4
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:]
            h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            x_first = x[:, :, :1]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)
            x_next = x[:, :, 1:]
            x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    def __init__(self, in_channels: int, z_channels: int, block_out_channels, num_res_blocks: int, ffactor_spatial: int, ffactor_temporal: int, downsample_match_channel: bool = True):
        super().__init__()
        self.z_channels = z_channels
        self.block_out_channels = tuple(block_out_channels)
        self.num_res_blocks = num_res_blocks
        self.conv_in = CausalConv3d(in_channels, self.block_out_channels[0], kernel_size=3)

        self.down = nn.ModuleList()
        block_in = self.block_out_channels[0]
        for i_level, ch in enumerate(self.block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            add_spatial_downsample = i_level < int(math.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and i_level >= int(math.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                block_out = self.block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = Downsample(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = RMSNorm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

    def forward(self, x):
        h = self.conv_in(x)
        for stage in self.down:
            for block in stage.block:
                h = block(h)
            if hasattr(stage, "downsample"):
                h = stage.downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        b, c, t, h_size, w_size = h.shape
        group_size = c // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) t h w -> b c r t h w", r=group_size).mean(dim=2)
        h = self.conv_out(swish(self.norm_out(h), inplace=True))
        return h + shortcut


class Decoder(nn.Module):
    def __init__(self, z_channels: int, out_channels: int, block_out_channels, num_res_blocks: int, ffactor_spatial: int, ffactor_temporal: int, upsample_match_channel: bool = True):
        super().__init__()
        self.z_channels = z_channels
        self.block_out_channels = tuple(block_out_channels)
        self.num_res_blocks = num_res_blocks

        block_in = self.block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level, ch in enumerate(self.block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            add_spatial_upsample = i_level < int(math.log2(ffactor_spatial))
            add_temporal_upsample = i_level < int(math.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                block_out = self.block_out_channels[i_level + 1] if i_level + 1 < len(self.block_out_channels) and upsample_match_channel else block_in
                up.upsample = Upsample(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        self.norm_out = RMSNorm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

    def forward(self, z):
        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for stage in self.up:
            for block in stage.block:
                h = block(h)
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)

        return self.conv_out(swish(self.norm_out(h), inplace=True))


class OmniWeavingAutoencoderKLConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels,
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float | None = None,
        shift_factor: float | None = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            sample_size=sample_size,
            sample_tsize=sample_tsize,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            downsample_match_channel=downsample_match_channel,
            upsample_match_channel=upsample_match_channel,
        )
        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
        )
        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.2):
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // self.config.ffactor_spatial
        self.tile_overlap_factor = tile_overlap_factor

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                row.append(self.encoder(tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def spatial_tiled_decode(self, z: torch.Tensor):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                row.append(self.decoder(tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def encode(self, x: torch.Tensor, device=None):
        assert x.ndim == 5

        def _encode(inp):
            if self.use_spatial_tiling and (inp.shape[-1] > self.tile_sample_min_size or inp.shape[-2] > self.tile_sample_min_size):
                return self.spatial_tiled_encode(inp)
            return self.encoder(inp)

        if self.use_slicing and x.shape[0] > 1:
            h = torch.cat([_encode(x_slice) for x_slice in x.split(1)], dim=0)
        else:
            h = _encode(x)
        return DiagonalGaussianDistribution(h).mode()

    def decode(self, z: torch.Tensor, output_buffer=None):
        def _decode(inp):
            if self.use_spatial_tiling and (inp.shape[-1] > self.tile_latent_min_size or inp.shape[-2] > self.tile_latent_min_size):
                return self.spatial_tiled_decode(inp)
            return self.decoder(inp)

        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([_decode(z_slice) for z_slice in z.split(1)], dim=0)
        else:
            decoded = _decode(z)

        if output_buffer is not None:
            output_buffer.copy_(decoded)
            return output_buffer
        return decoded

    @contextmanager
    def memory_efficient_context(self):
        original_use_slicing = self.use_slicing
        original_use_spatial_tiling = self.use_spatial_tiling
        self.enable_slicing()
        self.enable_tiling()
        try:
            yield
        finally:
            self.use_slicing = original_use_slicing
            self.use_spatial_tiling = original_use_spatial_tiling
