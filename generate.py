"""
FramePack CLI — Generate video from an image + text prompt, no UI needed.

Usage:
    python generate.py --image photo.jpg --prompt "The girl dances gracefully"

Optional flags:
    --seed 31337                   Random seed (default: 31337)
    --length 5                     Video length in seconds, 1-120 (default: 5)
    --steps 25                     Denoising steps, 1-100 (default: 25)
    --gs 10.0                      Distilled guidance scale, 1-32 (default: 10.0)
    --teacache / --no-teacache     TeaCache on/off (default: on, faster but slightly worse hands)
    --gpu-reserve 6                GPU memory to keep free in GB, 6-128 (default: 6)
    --crf 16                       MP4 quality, 0=lossless, 51=worst (default: 16)
    --output ./outputs             Output directory (default: ./outputs)
"""

from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import torch
import numpy as np
import argparse
import sys

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import gpu, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


# ── CLI arguments ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='FramePack CLI — image-to-video generation')
parser.add_argument('--image', type=str, required=True, help='Path to input image (jpg/png)')
parser.add_argument('--prompt', type=str, required=True, help='Text prompt describing the motion')
parser.add_argument('--seed', type=int, default=31337, help='Random seed (default: 31337)')
parser.add_argument('--length', type=float, default=5, help='Video length in seconds, range 1-120 (default: 5)')
parser.add_argument('--steps', type=int, default=25, help='Denoising steps, range 1-100 (default: 25)')
parser.add_argument('--gs', type=float, default=10.0, help='Distilled guidance scale, range 1-32 (default: 10.0)')
parser.add_argument('--teacache', action=argparse.BooleanOptionalAction, default=True, help='Use TeaCache for ~2x speed (slightly worse hands)')
parser.add_argument('--gpu-reserve', type=float, default=6, help='GPU memory to keep free in GB, range 6-128 (default: 6)')
parser.add_argument('--crf', type=int, default=16, help='MP4 quality: 0=lossless, 23=default, 51=worst (default: 16)')
parser.add_argument('--output', type=str, default='./outputs', help='Output directory (default: ./outputs)')
args = parser.parse_args()

# Validate inputs
if not os.path.isfile(args.image):
    print(f'Error: image not found: {args.image}')
    sys.exit(1)

args.length = max(1, min(120, args.length))
args.steps = max(1, min(100, args.steps))
args.gs = max(1.0, min(32.0, args.gs))
args.gpu_reserve = max(6, min(128, args.gpu_reserve))
args.crf = max(0, min(51, args.crf))

print(f'Settings: seed={args.seed}, length={args.length}s, steps={args.steps}, '
      f'guidance={args.gs}, teacache={args.teacache}, gpu_reserve={args.gpu_reserve}GB, crf={args.crf}')


# ── Load models ────────────────────────────────────────────────────────────────

print('Loading models...')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

vae.enable_slicing()
vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True

DynamicSwapInstaller.install_model(transformer, device=gpu)
DynamicSwapInstaller.install_model(text_encoder, device=gpu)

os.makedirs(args.output, exist_ok=True)

print('Models loaded.')


# ── Generate ───────────────────────────────────────────────────────────────────

LATENT_WINDOW_SIZE = 9

@torch.no_grad()
def generate():
    total_latent_sections = (args.length * 30) / (LATENT_WINDOW_SIZE * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    # Clean GPU
    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    # ── Step 1: Text encoding ──────────────────────────────────────────────
    print('[1/5] Encoding text...')

    fake_diffusers_current_device(text_encoder, gpu)
    load_model_as_complete(text_encoder_2, target_device=gpu)

    # Force config-level setting for transformers compatibility
    text_encoder.config.output_hidden_states = True

    llama_vec, clip_l_pooler = encode_prompt_conds(args.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    # CFG is fixed at 1.0 (no negative prompt needed for this distilled model)
    llama_vec_n = torch.zeros_like(llama_vec)
    clip_l_pooler_n = torch.zeros_like(clip_l_pooler)

    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    # ── Step 2: Image processing ───────────────────────────────────────────
    print('[2/5] Processing image...')

    input_image = np.array(Image.open(args.image).convert('RGB'))
    H, W, C = input_image.shape
    height, width = find_nearest_bucket(H, W, resolution=640)
    input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

    Image.fromarray(input_image_np).save(os.path.join(args.output, f'{job_id}.png'))

    input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
    input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

    # ── Step 3: VAE + CLIP Vision encoding ─────────────────────────────────
    print('[3/5] VAE encoding...')

    load_model_as_complete(vae, target_device=gpu)
    start_latent = vae_encode(input_image_pt, vae)

    print('[3/5] CLIP Vision encoding...')

    load_model_as_complete(image_encoder, target_device=gpu)
    image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
    image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

    # Cast all embeddings to transformer dtype
    llama_vec = llama_vec.to(transformer.dtype)
    llama_vec_n = llama_vec_n.to(transformer.dtype)
    clip_l_pooler = clip_l_pooler.to(transformer.dtype)
    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

    # ── Step 4: Diffusion sampling ─────────────────────────────────────────
    print(f'[4/5] Sampling {total_latent_sections} section(s)...')

    rnd = torch.Generator("cpu").manual_seed(args.seed)
    num_frames = LATENT_WINDOW_SIZE * 4 - 3

    history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
    history_pixels = None
    total_generated_latent_frames = 0

    latent_paddings = reversed(range(total_latent_sections))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    output_filename = None

    for section_idx, latent_padding in enumerate(latent_paddings):
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * LATENT_WINDOW_SIZE

        print(f'  Section {section_idx + 1}/{total_latent_sections} '
              f'(padding={latent_padding_size}, last={is_last_section})')

        indices = torch.arange(0, sum([1, latent_padding_size, LATENT_WINDOW_SIZE, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, LATENT_WINDOW_SIZE, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        unload_complete_models()
        move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=args.gpu_reserve)

        transformer.initialize_teacache(enable_teacache=args.teacache, num_steps=args.steps)

        def callback(d):
            current_step = d['i'] + 1
            total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
            video_len = max(0, total_frames / 30)
            print(f'    Step {current_step}/{args.steps} | '
                  f'{total_frames} frames | {video_len:.1f}s', end='\r')

        generated_latents = sample_hunyuan(
            transformer=transformer,
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=1.0,
            distilled_guidance_scale=args.gs,
            guidance_rescale=0.0,
            num_inference_steps=args.steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=callback,
        )

        print()  # newline after \r progress

        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

        # ── Step 5: VAE decode this section ────────────────────────────────
        offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
        load_model_as_complete(vae, target_device=gpu)

        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

        if history_pixels is None:
            history_pixels = vae_decode(real_history_latents, vae).cpu()
        else:
            section_latent_frames = (LATENT_WINDOW_SIZE * 2 + 1) if is_last_section else (LATENT_WINDOW_SIZE * 2)
            overlapped_frames = LATENT_WINDOW_SIZE * 4 - 3
            current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

        unload_complete_models()

        output_filename = os.path.join(args.output, f'{job_id}_{total_generated_latent_frames}.mp4')
        save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=args.crf)

        total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
        print(f'  Saved: {output_filename} ({total_frames} frames, {total_frames/30:.1f}s)')

        if is_last_section:
            break

    print(f'\nDone! Final video: {output_filename}')
    return output_filename


if __name__ == '__main__':
    generate()
