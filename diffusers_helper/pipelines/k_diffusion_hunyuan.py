import torch
import math

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size


# Applies a logistic time-shift to the noise schedule.
# Instead of linearly spacing noise levels from 1->0, this concentrates more steps
# in the high-noise regime (early denoising) where the model makes structural decisions.
# mu controls how aggressive the shift is (higher = more early steps).
# Mathematically: t' = exp(mu) / (exp(mu) + (1/t - 1)^sigma)
# When t=0.5 (midpoint), the shifted value is biased toward higher noise levels.
def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


# Adaptively computes the time-shift parameter mu based on sequence length.
# Longer sequences (more tokens = more spatial complexity) need a stronger shift
# because there are more structural decisions to make during early denoising.
# Uses simple linear interpolation: at 256 tokens, mu=0.5; at 4096 tokens, mu=1.15.
# Capped at log(7) to prevent extreme shifts.
def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


# Creates the noise schedule: n+1 sigma values from ~1.0 (pure noise) to ~0.0 (clean).
# First creates linearly-spaced values, then applies the time shift to concentrate
# more steps in the high-noise regime.
def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


# Main sampling entry point. Orchestrates the full denoising process:
#   1. Creates initial random noise in latent space
#   2. Computes an adaptive noise schedule based on sequence length
#   3. Wraps the transformer with CFG logic via fm_wrapper
#   4. Packages all conditioning (text embeddings, image embeddings, clean latents)
#   5. Runs the UniPC sampler for num_inference_steps to denoise
# The function supports two guidance scales:
#   - real_guidance_scale: standard CFG (run model twice, amplify difference)
#   - distilled_guidance_scale: embedded in the model itself (the model was trained
#     to accept a guidance signal as input, so it self-guides without needing 2 passes)
@torch.inference_mode()
def sample_hunyuan(
        transformer,
        width=512,
        height=512,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.0,
        num_inference_steps=25,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        callback=None,
        **kwargs,
):
    device = device or transformer.device
    batch_size = int(prompt_embeds.shape[0])

    # Generate pure random noise as the starting point for denoising.
    # Shape: [batch, 16 latent channels, (frames+3)//4 time steps, height//8, width//8]
    # The //4 and //8 account for the VAE's compression factors (4x temporal, 8x spatial).
    print(f'        [sample_hunyuan] Generating random noise: batch={batch_size}, channels=16, '
          f'frames={(frames+3)//4}, h={height//8}, w={width//8}')
    latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)
    print(f'        [sample_hunyuan] Initial noise shape: {latents.shape}')

    _, _, T, H, W = latents.shape
    seq_length = T * H * W // 4
    print(f'        [sample_hunyuan] Sequence length for schedule: {seq_length} tokens (T={T} x H={H} x W={W} / 4)')

    mu = calculate_flux_mu(seq_length, exp_max=7.0)
    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)
    print(f'        [sample_hunyuan] Noise schedule: mu={mu:.3f}, {len(sigmas)} sigma values')
    print(f'        [sample_hunyuan] Sigma range: {sigmas[0]:.4f} (noisy) -> {sigmas[-1]:.4f} (clean)')

    print(f'        [sample_hunyuan] Wrapping transformer with CFG logic (fm_wrapper)...')
    k_model = fm_wrapper(transformer)

    print(f'        [sample_hunyuan] Distilled guidance scale: {distilled_guidance_scale} (embedded in model, x1000 = {distilled_guidance_scale * 1000})')
    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)

    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        )
    )

    results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    return results
