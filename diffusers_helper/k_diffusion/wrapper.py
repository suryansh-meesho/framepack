import torch


# Adds trailing singleton dimensions to a tensor so it can broadcast with a target shape.
# Example: if x has shape [2] and target_dims=5, result has shape [2, 1, 1, 1, 1].
# This is needed when multiplying a per-batch scalar (sigma) with a 5D video tensor.
def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


# Fixes the over-saturation problem caused by high Classifier-Free Guidance (CFG) scales.
# High CFG inflates the standard deviation of the prediction, causing washed-out or
# over-saturated colors. This function rescales the CFG prediction to match the standard
# deviation of the text-conditional prediction, then blends between rescaled and original.
# guidance_rescale=0 means no correction; guidance_rescale=1 means full correction.
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    if guidance_rescale == 0:
        return noise_cfg

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg


# Wraps the diffusion transformer for use with k-diffusion samplers.
# The wrapper handles three key responsibilities:
#   1. Classifier-Free Guidance (CFG): runs the model TWICE -- once with text (positive)
#      and once without text (negative) -- then amplifies the difference by cfg_scale.
#      Formula: result = negative + scale * (positive - negative)
#      This makes the model follow the text prompt more strongly.
#   2. Guidance rescale: optionally corrects CFG's tendency to over-saturate.
#   3. Velocity-to-x0 conversion: the model predicts "velocity" (direction from noise to
#      clean), but the sampler expects x0 (the clean image estimate).
#      Formula: x0 = noisy_input - velocity * sigma (noise level)
def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        dtype = extra_args['dtype']
        cfg_scale = extra_args['cfg_scale']
        cfg_rescale = extra_args['cfg_rescale']

        original_dtype = x.dtype
        sigma = sigma.float()

        x = x.to(dtype)
        # Convert sigma (0..1 range) to timestep (0..1000 range) as expected by the model
        timestep = (sigma * t_scale).to(dtype)

        # Forward pass WITH text conditioning (the "positive" prediction)
        pred_positive = transformer(hidden_states=x, timestep=timestep, return_dict=False, **extra_args['positive'])[0].float()

        # Forward pass WITHOUT text conditioning (the "negative" / unconditional prediction)
        # When cfg_scale=1, skip the negative pass entirely (no guidance needed)
        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            pred_negative = transformer(hidden_states=x, timestep=timestep, return_dict=False, **extra_args['negative'])[0].float()

        # Apply Classifier-Free Guidance:
        # Amplify the difference between conditioned and unconditioned predictions
        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale)

        # Convert velocity prediction to x0 (clean image estimate):
        # In flow matching, the model predicts the flow velocity v such that:
        #   x_noisy = x_clean + sigma * v
        # Therefore: x_clean = x_noisy - sigma * v
        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)

        return x0.to(dtype=original_dtype)

    return k_model
