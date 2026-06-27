# Monkey-patches normalization layer forward methods for dtype-safe mixed-precision inference.
# Problem: when running in bfloat16/float16, the default normalization implementations
# may produce outputs in the wrong dtype, causing silent precision loss or errors.
# These patches ensure the output dtype always matches the input dtype.
#
# Also disables accelerate's automatic fp32 output conversion, which would undo
# the careful dtype management done throughout the model.

import torch
import accelerate.accelerator

from diffusers.models.normalization import RMSNorm, LayerNorm, FP32LayerNorm, AdaLayerNormContinuous


# Disable automatic fp32 conversion -- we handle dtypes manually
accelerate.accelerator.convert_outputs_to_fp32 = lambda x: x


# Patched LayerNorm: ensures output dtype matches input dtype via .to(x)
def LayerNorm_forward(self, x):
    return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).to(x)


LayerNorm.forward = LayerNorm_forward
torch.nn.LayerNorm.forward = LayerNorm_forward


# FP32 LayerNorm: always computes in float32 for numerical stability, then casts back.
# LayerNorm involves division and square roots that can lose precision in float16/bfloat16.
def FP32LayerNorm_forward(self, x):
    origin_dtype = x.dtype
    return torch.nn.functional.layer_norm(
        x.float(),
        self.normalized_shape,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.eps,
    ).to(origin_dtype)


FP32LayerNorm.forward = FP32LayerNorm_forward


# RMSNorm (Root Mean Square Normalization): a simpler alternative to LayerNorm.
# Instead of centering (subtracting mean) AND scaling, it only scales by the
# root-mean-square of the values. Fewer operations = faster, and works just as well
# for transformer models. Used for normalizing Q and K in attention (qk_norm).
# Formula: x_norm = x / sqrt(mean(x^2) + eps) * weight
def RMSNorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

    if self.weight is None:
        return hidden_states.to(input_dtype)

    return hidden_states.to(input_dtype) * self.weight.to(input_dtype)


RMSNorm.forward = RMSNorm_forward


# Adaptive LayerNorm: adjusts normalization based on a conditioning signal (timestep).
# First normalizes x, then applies learned scale and shift derived from the conditioning.
# This lets the model behave differently at different noise levels during denoising.
def AdaLayerNormContinuous_forward(self, x, conditioning_embedding):
    emb = self.linear(self.silu(conditioning_embedding))
    scale, shift = emb.chunk(2, dim=1)
    x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    return x


AdaLayerNormContinuous.forward = AdaLayerNormContinuous_forward
