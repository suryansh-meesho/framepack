import torch

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask


# Encodes a text prompt into two separate embeddings using dual text encoders:
#   1. LLaMA (large language model) -> per-token semantic vectors [1, up_to_256, 4096]
#      We extract hidden_states[-3] (3rd-to-last layer) because final layers are too
#      specialized for next-token prediction. Middle layers carry richer, more general
#      semantic features ideal for conditioning a generative model.
#   2. CLIP-L (vision-language model) -> single pooled summary vector [1, 768]
#      Captures the global "visual vibe" of the text in one fixed-size vector.
# The prompt is wrapped in an instruction template (system/user format) because LLaMA
# was trained on instruction-tuned data. crop_start skips the template's system tokens
# so only the user's prompt content is kept.
@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)

    prompt = [prompt]

    # LLAMA

    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]
    print(f'        [encode_prompt] Wrapped prompt in template, crop_start={crop_start}')

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())
    print(f'        [encode_prompt] LLaMA tokenized: {llama_input_ids.shape} token IDs, '
          f'{llama_attention_length} real tokens (rest is padding)')

    print(f'        [encode_prompt] Running LLaMA forward pass (this streams layers through GPU via DynamicSwap)...')
    text_encoder.config.output_hidden_states=True
    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

    # Extract 3rd-to-last hidden state, cropped to skip system-prompt tokens.
    # hidden_states[-3] is chosen because it has rich semantic content without being
    # over-specialized for LLaMA's original text-generation objective.
    num_layers = len(llama_outputs.hidden_states)
    print(f'        [encode_prompt] LLaMA returned {num_layers} hidden state layers, using layer [{num_layers - 3}] (3rd from last)')
    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]
    print(f'        [encode_prompt] Cropped from position {crop_start} to {llama_attention_length} -> shape {llama_vec.shape}')

    # CLIP -- produces a single 768-dim "pooler_output" that summarizes the entire
    # sentence's visual meaning. Used as a global conditioning signal.

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    print(f'        [encode_prompt] CLIP tokenized: {clip_l_input_ids.shape} (max 77 tokens)')
    print(f'        [encode_prompt] Running CLIP-L forward pass...')
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output
    print(f'        [encode_prompt] CLIP pooler output: {clip_l_pooler.shape}')

    return llama_vec, clip_l_pooler


# Approximate VAE decode: instead of running the full decoder network, projects the
# 16-channel latent directly to 3-channel RGB using a pre-computed 16x3 linear matrix.
# This is a 1x1x1 convolution -- essentially a matrix multiply at each spatial position.
# Result is blurry but nearly instant. Used for the live preview during generation.
# The factors were empirically determined (from ComfyUI) to approximate what the full
# VAE decoder would produce.
@torch.no_grad()
def vae_decode_fake(latents):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703]
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)

    return images


# Full VAE decode: decompresses latent-space tensors back to pixel-space video.
# The scaling_factor (~0.13) was applied during encoding to keep latent values in a
# nice range for diffusion; we undo it here before decoding.
@torch.no_grad()
def vae_decode(latents, vae):
    print(f'        [vae_decode] Input latent: {latents.shape}, scaling_factor={vae.config.scaling_factor:.4f}')
    latents = latents / vae.config.scaling_factor
    print(f'        [vae_decode] Running VAE decoder network (latent -> pixels)...')
    image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    print(f'        [vae_decode] Output pixels: {image.shape}, value range [{image.min():.2f}, {image.max():.2f}]')
    return image


# VAE encode: compresses pixel-space images/video into latent space.
# The encoder outputs a probability distribution (mean + variance), not a single point.
# .sample() draws one random point from that distribution -- this randomness is what
# makes VAEs "variational" and enables generative capabilities.
# The scaling_factor normalizes latent values to a range suitable for the diffusion model.
@torch.no_grad()
def vae_encode(image, vae):
    print(f'        [vae_encode] Input image: {image.shape}, device={image.device}')
    print(f'        [vae_encode] Running VAE encoder -> produces mean + variance (distribution)...')
    latent_dist = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist
    print(f'        [vae_encode] Sampling one point from the latent distribution...')
    latents = latent_dist.sample()
    print(f'        [vae_encode] Raw latent: {latents.shape}, range [{latents.min():.2f}, {latents.max():.2f}]')
    latents = latents * vae.config.scaling_factor
    print(f'        [vae_encode] After scaling (x{vae.config.scaling_factor:.4f}): range [{latents.min():.2f}, {latents.max():.2f}]')
    return latents
