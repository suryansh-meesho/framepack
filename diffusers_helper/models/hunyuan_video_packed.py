from typing import Optional, Tuple

import torch
import einops
import torch.nn as nn
import numpy as np

from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers_helper.dit_common import LayerNorm


enabled_backends = []

if torch.backends.cuda.flash_sdp_enabled():
    enabled_backends.append("flash")
if torch.backends.cuda.math_sdp_enabled():
    enabled_backends.append("math")
if torch.backends.cuda.mem_efficient_sdp_enabled():
    enabled_backends.append("mem_efficient")
if torch.backends.cuda.cudnn_sdp_enabled():
    enabled_backends.append("cudnn")

print("Currently enabled native sdp backends:", enabled_backends)

try:
    # raise NotImplementedError
    from xformers.ops import memory_efficient_attention as xformers_attn_func
    print('Xformers is installed!')
except:
    print('Xformers is not installed!')
    xformers_attn_func = None

try:
    # raise NotImplementedError
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    print('Flash Attn is installed!')
except:
    print('Flash Attn is not installed!')
    flash_attn_varlen_func = None
    flash_attn_func = None

try:
    # raise NotImplementedError
    from sageattention import sageattn_varlen, sageattn
    print('Sage Attn is installed!')
except:
    print('Sage Attn is not installed!')
    sageattn_varlen = None
    sageattn = None



# Pads a 5D tensor (batch, channels, time, height, width) so its spatial/temporal
# dimensions are evenly divisible by the kernel_size. Uses 'replicate' padding
# (repeats edge values) rather than zero padding to avoid border artifacts.
# Needed before applying strided Conv3D or avg_pool3d that requires exact divisibility.
def pad_for_3d_conv(x, kernel_size):
    _, _, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')


# Downsamples a 5D tensor by averaging non-overlapping blocks of kernel_size.
# Used to downsample RoPE position embeddings to match the resolution of
# compressed clean latents (2x and 4x scales in Frame Packing).
def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


# Computes cumulative sequence lengths for variable-length attention (flash/sage attn).
# When batch_size > 1 and different samples have different text lengths, this tells
# the attention kernel where each sequence starts/ends in the flattened batch.
# Only needed for batch_size > 1; for batch_size=1 we just crop the text instead.
def get_cu_seqlens(text_mask, img_len):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


# Applies Rotary Position Embedding (RoPE) to queries/keys.
# RoPE encodes position by rotating pairs of dimensions. Mathematically, it treats
# adjacent pairs of dimensions as 2D vectors and rotates them by an angle that depends
# on the position. This way, the dot product between two tokens naturally encodes
# their relative distance — nearby tokens have similar rotations, far tokens don't.
# The rotation formula: [x_real, x_imag] -> [x_real*cos - x_imag*sin, x_imag*cos + x_real*sin]
def apply_rotary_emb_transposed(x, freqs_cis):
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = x.float() * cos + x_rotated.float() * sin
    out = out.to(x)
    return out


# Chunked attention fallback for MPS / memory-constrained devices.
# Instead of materializing the full [B, heads, seq_len, seq_len] attention matrix
# (which can be 14+ GB), processes queries in chunks of `chunk_size` tokens at a time.
# Each chunk only needs [B, heads, chunk_size, seq_len] — much smaller.
def _chunked_sdp_attention(q, k, v, chunk_size=1024):
    # q/k/v: [B, heads, seq_len, dim]
    B, H, N, D = q.shape
    output = torch.empty_like(q)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        output[:, :, start:end, :] = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, start:end, :], k, v
        )
    return output


# Attention dispatch function. Chooses the fastest available backend:
#   1. SageAttention (fastest, ~quantized attention)
#   2. Flash Attention (fast, exact)
#   3. xformers memory-efficient attention
#   4. PyTorch native scaled_dot_product_attention (fallback)
#   5. Chunked attention for MPS (processes queries in chunks to avoid OOM)
# When cu_seqlens are provided (batch>1), uses variable-length versions that handle
# different sequence lengths per batch element without wasteful padding.
def attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    if cu_seqlens_q is None and cu_seqlens_kv is None and max_seqlen_q is None and max_seqlen_kv is None:
        if sageattn is not None:
            x = sageattn(q, k, v, tensor_layout='NHD')
            return x

        if flash_attn_func is not None:
            x = flash_attn_func(q, k, v)
            return x

        if xformers_attn_func is not None:
            x = xformers_attn_func(q, k, v)
            return x

        # q is in NHD layout [batch, seq_len, heads, dim] — transpose to [batch, heads, seq_len, dim] for SDP
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # On MPS or when sequence is very long, use chunked attention to avoid OOM
        if q.device.type == 'mps' or q.shape[1] > 8192:
            x = _chunked_sdp_attention(qt, kt, vt, chunk_size=1024).transpose(1, 2)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt).transpose(1, 2)
        return x

    B, L, _, _ = q.shape

    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    if sageattn_varlen is not None:
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    elif flash_attn_varlen_func is not None:
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    else:
        raise NotImplementedError('No Attn Installed!')

    x = x.unflatten(0, (B, L))

    return x


class HunyuanAttnProcessorFlashAttnDouble:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb):
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = apply_rotary_emb_transposed(query, image_rotary_emb)
        key = apply_rotary_emb_transposed(key, image_rotary_emb)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        hidden_states = attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        hidden_states = hidden_states.flatten(-2)

        txt_length = encoder_hidden_states.shape[1]
        hidden_states, encoder_hidden_states = hidden_states[:, :-txt_length], hidden_states[:, -txt_length:]

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanAttnProcessorFlashAttnSingle:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb):
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask

        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        txt_length = encoder_hidden_states.shape[1]

        query = torch.cat([apply_rotary_emb_transposed(query[:, :-txt_length], image_rotary_emb), query[:, -txt_length:]], dim=1)
        key = torch.cat([apply_rotary_emb_transposed(key[:, :-txt_length], image_rotary_emb), key[:, -txt_length:]], dim=1)

        hidden_states = attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        hidden_states = hidden_states.flatten(-2)

        hidden_states, encoder_hidden_states = hidden_states[:, :-txt_length], hidden_states[:, -txt_length:]

        return hidden_states, encoder_hidden_states


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=-1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate)

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta

    @torch.no_grad()
    def get_frequency(self, dim, pos):
        T, H, W = pos.shape
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: (dim // 2)] / dim))
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, (T, H, W)).repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward_inner(self, frame_indices, height, width, device):
        GT, GY, GX = torch.meshgrid(
            frame_indices.to(device=device, dtype=torch.float32),
            torch.arange(0, height, device=device, dtype=torch.float32),
            torch.arange(0, width, device=device, dtype=torch.float32),
            indexing="ij"
        )

        FCT, FST = self.get_frequency(self.DT, GT)
        FCY, FSY = self.get_frequency(self.DY, GY)
        FCX, FSX = self.get_frequency(self.DX, GX)

        result = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)

        return result.to(device)

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        frame_indices = frame_indices.unbind(0)
        results = [self.forward_inner(f, height, width, device) for f in frame_indices]
        results = torch.stack(results, dim=0)
        return results


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanAttnProcessorFlashAttnSingle(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanAttnProcessorFlashAttnDouble(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

        return hidden_states, encoder_hidden_states


# Projects CLIP Vision features (1152-dim) to the Transformer's inner dimension (3072-dim).
# Uses an "expand-then-compress" pattern: up-project 3x wider, apply SiLU activation,
# then down-project back. The wider intermediate layer lets the network learn more
# complex transformations than a single linear layer could.
class ClipVisionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x):
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x


# Converts the video latent from pixel-like format to Transformer tokens via Conv3D.
# kernel_size = stride = patch_size, so patches are non-overlapping.
# Input:  [batch, 16 latent_channels, frames, height, width]
# Output: [batch, embed_dim (3072), frames/patch_t, height/patch_h, width/patch_w]
# Each output position is one "token" representing a spatio-temporal patch.
class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


# THE FRAME PACKING CORE: embeds "clean" (already-generated, non-noisy) latent frames
# at three different compression scales. This is what makes the context length constant
# regardless of video length.
#   proj    (1x): kernel (1,2,2)  -- full temporal resolution, 2x spatial downscale
#   proj_2x (2x): kernel (2,4,4)  -- 2x temporal, 4x spatial downscale (8x fewer tokens)
#   proj_4x (4x): kernel (4,8,8)  -- 4x temporal, 8x spatial downscale (64x fewer tokens!)
# Old frames use proj_4x (heavily compressed), recent frames use proj (barely compressed).
class HunyuanVideoPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            'proj.weight': weight.clone(),
            'proj.bias': bias.clone(),
            'proj_2x.weight': einops.repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=2, hk=2, wk=2) / 8.0,
            'proj_2x.bias': bias.clone(),
            'proj_4x.weight': einops.repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=4, hk=4, wk=4) / 64.0,
            'proj_4x.bias': bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd)
        return


class HunyuanVideoTransformer3DModelPacked(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        has_image_proj=False,
        image_proj_dim=1152,
        has_clean_x_embedder=False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(inner_dim, pooled_projection_dim)

        self.clean_x_embedder = None
        self.image_projection = None

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.inner_dim = inner_dim
        self.enable_teacache = False

        if has_image_proj:
            self.install_image_projection(image_proj_dim)

        if has_clean_x_embedder:
            self.install_clean_x_embedder()

        self.high_quality_fp32_output_for_inference = False

    def install_image_projection(self, in_channels):
        self.image_projection = ClipVisionProjection(in_channels=in_channels, out_channels=self.inner_dim)
        self.config['has_image_proj'] = True
        self.config['image_proj_dim'] = in_channels

    def install_clean_x_embedder(self):
        self.clean_x_embedder = HunyuanVideoPatchEmbedForCleanLatents(self.inner_dim)
        self.config['has_clean_x_embedder'] = True

    # TeaCache: a caching mechanism that skips full transformer computation when the
    # input hasn't changed significantly between denoising steps. Measures the relative
    # L1 distance of the modulated input (after AdaLN) and accumulates it. When the
    # accumulated distance exceeds rel_l1_thresh, the full computation runs and the
    # residual (output - input) is cached. Otherwise, the cached residual is reused.
    # The polynomial rescale_func maps raw L1 distance to calibrated importance scores.
    def initialize_teacache(self, enable_teacache=True, num_steps=25, rel_l1_thresh=0.15):
        self.enable_teacache = enable_teacache
        self.cnt = 0
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh  # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.teacache_rescale_func = np.poly1d([7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02])

    # THE FRAME PACKING LOGIC: embeds the noisy latent and all clean context latents
    # at multiple scales (1x, 2x, 4x), computes position embeddings for each, and
    # concatenates them into a single token sequence. The clean latents are PREPENDED
    # to the noisy latent tokens, so the Transformer sees:
    #   [4x_compressed_old_frames | 2x_compressed_recent | 1x_recent | noisy_new_frames]
    # This keeps total token count CONSTANT regardless of video length.
    def process_input_hidden_states(
            self,
            latents, latent_indices=None,
            clean_latents=None, clean_latent_indices=None,
            clean_latents_2x=None, clean_latent_2x_indices=None,
            clean_latents_4x=None, clean_latent_4x_indices=None
    ):
        hidden_states = self.x_embedder.proj(latents)
        B, _, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)

        if clean_latents is not None and clean_latent_indices is not None:
            clean_latents = clean_latents.to(hidden_states)
            clean_latents = self.clean_x_embedder.proj(clean_latents)
            clean_latents = clean_latents.flatten(2).transpose(1, 2)

            clean_latent_rope_freqs = self.rope(frame_indices=clean_latent_indices, height=H, width=W, device=clean_latents.device)
            clean_latent_rope_freqs = clean_latent_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)

        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            clean_latents_2x = clean_latents_2x.to(hidden_states)
            clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
            clean_latents_2x = self.clean_x_embedder.proj_2x(clean_latents_2x)
            clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)

            clean_latent_2x_rope_freqs = self.rope(frame_indices=clean_latent_2x_indices, height=H, width=W, device=clean_latents_2x.device)
            clean_latent_2x_rope_freqs = pad_for_3d_conv(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = center_down_sample_3d(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = clean_latent_2x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            clean_latents_4x = clean_latents_4x.to(hidden_states)
            clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
            clean_latents_4x = self.clean_x_embedder.proj_4x(clean_latents_4x)
            clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

            clean_latent_4x_rope_freqs = self.rope(frame_indices=clean_latent_4x_indices, height=H, width=W, device=clean_latents_4x.device)
            clean_latent_4x_rope_freqs = pad_for_3d_conv(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = clean_latent_4x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

        return hidden_states, rope_freqs

    def forward(
            self,
            hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance,
            latent_indices=None,
            clean_latents=None, clean_latent_indices=None,
            clean_latents_2x=None, clean_latent_2x_indices=None,
            clean_latents_4x=None, clean_latent_4x_indices=None,
            image_embeddings=None,
            return_dict=True
    ):
        batch_size, _, num_frames, height, width = hidden_states.shape
        p, p_t = self.config['patch_size'], self.config['patch_size_t']
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)

        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        if self.image_projection is not None:
            assert image_embeddings is not None, 'You must use image embeddings!'
            extra_encoder_hidden_states = self.image_projection(image_embeddings)
            extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)

            # must cat before (not after) encoder_hidden_states, due to attn masking
            encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

        if batch_size == 1:
            # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
            # If they are not same, then their impls are wrong. Ours are always the correct one.
            text_len = encoder_attention_mask.sum().item()
            encoder_hidden_states = encoder_hidden_states[:, :text_len]
            attention_mask = None, None, None, None
        else:
            img_seq_len = hidden_states.shape[1]
            txt_seq_len = encoder_hidden_states.shape[1]

            cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q

            attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

        if self.enable_teacache:
            modulated_inp = self.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]

            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                curr_rel_l1 = ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
                self.accumulated_rel_l1_distance += self.teacache_rescale_func(curr_rel_l1)
                should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh

                if should_calc:
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = modulated_inp
            self.cnt += 1

            if self.cnt == self.num_steps:
                self.cnt = 0

            if not should_calc:
                hidden_states = hidden_states + self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()

                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        rope_freqs
                    )

                for block in self.single_transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        rope_freqs
                    )

                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    rope_freqs
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    rope_freqs
                )

        hidden_states = self.norm_out(hidden_states, temb)

        hidden_states = hidden_states[:, -original_context_length:, :]

        if self.high_quality_fp32_output_for_inference:
            hidden_states = hidden_states.to(dtype=torch.float32)
            if self.proj_out.weight.dtype != torch.float32:
                self.proj_out.to(dtype=torch.float32)

        hidden_states = self.proj_out(hidden_states)

        hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                         t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
                                         pt=p_t, ph=p, pw=p)

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)

        return hidden_states,
