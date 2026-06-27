"""
FramePack CLI — Generate video from an image + text prompt, no UI needed.
Includes checkpointing: if it crashes mid-generation, re-run the same command to resume.

Usage:
    python generate.py --image photo.jpg --prompt "The girl dances gracefully"
    # If it crashes, just run the EXACT same command again — it resumes from last checkpoint.

Optional flags:
    --seed 31337                   Random seed (default: 31337)
    --length 5                     Video length in seconds, 1-120 (default: 5)
    --steps 25                     Denoising steps, 1-100 (default: 25)
    --gs 10.0                      Distilled guidance scale, 1-32 (default: 10.0)
    --teacache / --no-teacache     TeaCache on/off (default: on, faster but slightly worse hands)
    --gpu-reserve 6                GPU memory to keep free in GB, 6-128 (default: 6)
    --crf 16                       MP4 quality, 0=lossless, 51=worst (default: 16)
    --output ./outputs             Output directory (default: ./outputs)
    --fresh                        Ignore any existing checkpoint, start from scratch
"""

from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import torch
import numpy as np
import argparse
import sys
import time
import hashlib
import json

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
parser.add_argument('--length', type=float, default=2, help='Video length in seconds, range 1-120 (default: 5)')
parser.add_argument('--steps', type=int, default=15, help='Denoising steps, range 1-100 (default: 25)')
parser.add_argument('--gs', type=float, default=10.0, help='Distilled guidance scale, range 1-32 (default: 10.0)')
parser.add_argument('--teacache', action=argparse.BooleanOptionalAction, default=True, help='Use TeaCache for ~2x speed (slightly worse hands)')
parser.add_argument('--gpu-reserve', type=float, default=8, help='GPU memory to keep free in GB, range 6-128 (default: 6)')
parser.add_argument('--crf', type=int, default=16, help='MP4 quality: 0=lossless, 23=default, 51=worst (default: 16)')
parser.add_argument('--output', type=str, default='./outputs', help='Output directory (default: ./outputs)')
parser.add_argument('--fresh', action='store_true', help='Ignore existing checkpoint, start from scratch')
args = parser.parse_args()

if not os.path.isfile(args.image):
    print(f'Error: image not found: {args.image}')
    sys.exit(1)

args.length = max(1, min(120, args.length))
args.steps = max(1, min(100, args.steps))
args.gs = max(1.0, min(32.0, args.gs))
args.gpu_reserve = max(6, min(128, args.gpu_reserve))
args.crf = max(0, min(51, args.crf))


# ── Checkpoint system ──────────────────────────────────────────────────────────
# Saves intermediate tensors after each expensive step so crashes don't lose work.
# Keyed by image name + prompt + all generation parameters.
# Re-running the same command auto-resumes from the last saved checkpoint.

CHECKPOINT_DIR = os.path.join(args.output, '.checkpoints')

def _checkpoint_key():
    """Unique key from image name + prompt + all params that affect generation."""
    image_name = os.path.basename(args.image)
    raw = f'{image_name}|{args.prompt}|{args.seed}|{args.length}|{args.steps}|{args.gs}|{args.teacache}'
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def _checkpoint_path(stage):
    """Path for a checkpoint file: .checkpoints/<key>_<stage>.pt"""
    return os.path.join(CHECKPOINT_DIR, f'{_checkpoint_key()}_{stage}.pt')

def _checkpoint_meta_path():
    return os.path.join(CHECKPOINT_DIR, f'{_checkpoint_key()}_meta.json')

def save_checkpoint(stage, data_dict):
    """Save tensors to disk after completing an expensive step."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = _checkpoint_path(stage)
    torch.save(data_dict, path)
    # Also save/update metadata
    meta_path = _checkpoint_meta_path()
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta['image'] = os.path.basename(args.image)
    meta['prompt'] = args.prompt
    meta['last_stage'] = stage
    meta['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'        [checkpoint] Saved stage "{stage}" -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)')

def load_checkpoint(stage):
    """Load a previously saved checkpoint. Returns None if not found."""
    path = _checkpoint_path(stage)
    if os.path.exists(path) and not args.fresh:
        data = torch.load(path, map_location='cpu', weights_only=True)
        print(f'        [checkpoint] RESUMED stage "{stage}" from {path}')
        return data
    return None

def has_checkpoint(stage):
    return os.path.exists(_checkpoint_path(stage)) and not args.fresh

def clean_checkpoints():
    """Remove all checkpoints for this job after successful completion."""
    key = _checkpoint_key()
    if os.path.isdir(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            if f.startswith(key):
                os.remove(os.path.join(CHECKPOINT_DIR, f))
        print(f'        [checkpoint] Cleaned up checkpoint files for this job')


# ── VAE decode with frame-by-frame fallback for MPS OOM ───────────────────────

def vae_decode_safe(latents, vae):
    """Decode latents to pixels. Falls back to frame-by-frame if OOM."""
    try:
        return vae_decode(latents, vae)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'MPS' in str(e):
            num_frames = latents.shape[2]
            print(f'        [vae_decode_safe] OOM decoding {num_frames} frames at once!')
            print(f'        [vae_decode_safe] Falling back to frame-by-frame decode...')
            frames = []
            for i in range(num_frames):
                print(f'        [vae_decode_safe] Decoding frame {i + 1}/{num_frames}...')
                frame_latent = latents[:, :, i:i+1, :, :]
                frame_pixels = vae_decode(frame_latent, vae)
                frames.append(frame_pixels.cpu())
            result = torch.cat(frames, dim=2)
            print(f'        [vae_decode_safe] All frames decoded. Shape: {result.shape}')
            return result
        raise


# ── Print banner ───────────────────────────────────────────────────────────────

print('=' * 70)
print('FRAMEPACK CLI — Image-to-Video Generation')
print('=' * 70)
print(f'  Image:         {args.image}')
print(f'  Prompt:        "{args.prompt}"')
print(f'  Seed:          {args.seed}')
print(f'  Video length:  {args.length}s')
print(f'  Steps:         {args.steps}')
print(f'  Guidance:      {args.gs}')
print(f'  TeaCache:      {args.teacache}')
print(f'  GPU reserve:   {args.gpu_reserve} GB')
print(f'  CRF:           {args.crf}')
print(f'  Output dir:    {args.output}')
print(f'  Device:        {gpu}')
print(f'  Checkpoint key:{_checkpoint_key()}')
if has_checkpoint('encodings'):
    print(f'  RESUMABLE:     Yes! Found checkpoint for this image+prompt+params')
print('=' * 70)


# ── Load models ────────────────────────────────────────────────────────────────

print('\n>>> PHASE 0: Loading models to CPU (this downloads ~30GB on first run)')
print('-' * 70)

t0 = time.time()

print('  [0.1] Loading LLaMA text encoder (language model, ~13GB)...')
print('        This converts your text prompt into semantic vectors.')
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
print(f'        Done. Parameters: {sum(p.numel() for p in text_encoder.parameters()) / 1e9:.2f}B')

print('  [0.2] Loading CLIP-L text encoder (~340MB)...')
print('        This creates a single "summary vector" of your prompt\'s visual meaning.')
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
print(f'        Done. Parameters: {sum(p.numel() for p in text_encoder_2.parameters()) / 1e6:.1f}M')

print('  [0.3] Loading tokenizers (LLaMA + CLIP vocabularies)...')
print('        These split text into token IDs the models understand.')
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
print(f'        LLaMA vocab size: {tokenizer.vocab_size}, CLIP vocab size: {tokenizer_2.vocab_size}')

print('  [0.4] Loading VAE (Variational Autoencoder, ~400MB)...')
print('        This compresses images to latent space and decompresses back.')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
print(f'        Done. Scaling factor: {vae.config.scaling_factor}')

print('  [0.5] Loading SigLIP image encoder (~400MB)...')
print('        This understands WHAT is in your input image (objects, scene, style).')
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
print(f'        Done. Parameters: {sum(p.numel() for p in image_encoder.parameters()) / 1e6:.1f}M')

print('  [0.6] Loading FramePack Transformer (the 13B diffusion model)...')
print('        This is the brain — it predicts what the next video frames should look like.')
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
print(f'        Done. Parameters: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B')
print(f'        Dual-stream blocks: {len(transformer.transformer_blocks)}')
print(f'        Single-stream blocks: {len(transformer.single_transformer_blocks)}')
print(f'        Inner dim: {transformer.inner_dim}')

print('  [0.7] Setting all models to eval mode (disabling dropout/batchnorm training behavior)...')
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

print('  [0.8] Enabling VAE slicing + tiling (processes video in small chunks to save memory)...')
vae.enable_slicing()
vae.enable_tiling()

print('  [0.9] Enabling fp32 output for final projection (better quality)...')
transformer.high_quality_fp32_output_for_inference = True

print('  [0.10] Installing DynamicSwapInstaller on transformer + LLaMA...')
print('         This patches __getattr__ so weights move to GPU only when accessed.')
print('         The full 13B model stays on CPU; only the active layer is on GPU at a time.')
DynamicSwapInstaller.install_model(transformer, device=gpu)
DynamicSwapInstaller.install_model(text_encoder, device=gpu)

os.makedirs(args.output, exist_ok=True)

print(f'\n  All models loaded in {time.time() - t0:.1f}s')
print(f'  Total model memory on CPU: ~{sum(sum(p.numel() * p.element_size() for p in m.parameters()) for m in [text_encoder, text_encoder_2, vae, image_encoder, transformer]) / 1e9:.1f} GB')


# ── Generate ───────────────────────────────────────────────────────────────────

LATENT_WINDOW_SIZE = 9

@torch.no_grad()
def generate():
    total_latent_sections = (args.length * 30) / (LATENT_WINDOW_SIZE * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    print('\n' + '=' * 70)
    print('>>> GENERATION STARTED')
    print('=' * 70)
    print(f'  Job ID: {job_id}')
    print(f'  Target: {args.length}s at 30fps = {int(args.length * 30)} frames')
    print(f'  Latent window size: {LATENT_WINDOW_SIZE} (each section generates {LATENT_WINDOW_SIZE * 4 - 3} pixel frames)')
    print(f'  Total sections to generate: {total_latent_sections}')
    print(f'  Generation order: INVERTED (ending frames first, then backward to start)')
    gen_start = time.time()

    # Clean GPU — move everything to CPU first
    print('\n  Clearing GPU — moving all models to CPU...')
    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    # ── Steps 1-3: Encoding (checkpointed as one unit) ─────────────────────
    cached = load_checkpoint('encodings')

    if cached is not None:
        print('\n>>> STEPS 1-3: SKIPPED (loaded from checkpoint)')
        print('-' * 70)
        llama_vec = cached['llama_vec']
        llama_vec_n = cached['llama_vec_n']
        clip_l_pooler = cached['clip_l_pooler']
        clip_l_pooler_n = cached['clip_l_pooler_n']
        llama_attention_mask = cached['llama_attention_mask']
        llama_attention_mask_n = cached['llama_attention_mask_n']
        start_latent = cached['start_latent']
        image_encoder_last_hidden_state = cached['image_encoder_last_hidden_state']
        height = int(cached['height'])
        width = int(cached['width'])
        print(f'  Restored: llama_vec={llama_vec.shape}, start_latent={start_latent.shape}, '
              f'clip_vision={image_encoder_last_hidden_state.shape}')
        print(f'  Resolution: {width}x{height}')
    else:
        # ── Step 1: Text encoding ──────────────────────────────────────────
        print('\n>>> STEP 1/5: TEXT ENCODING')
        print('-' * 70)
        print(f'  Prompt: "{args.prompt}"')
        t1 = time.time()

        print('  [1.1] Moving one LLaMA parameter to GPU (tricks diffusers into thinking model is on GPU)...')
        fake_diffusers_current_device(text_encoder, gpu)

        print('  [1.2] Loading CLIP-L text encoder to GPU...')
        load_model_as_complete(text_encoder_2, target_device=gpu)

        print('  [1.3] Setting output_hidden_states=True on LLaMA config...')
        text_encoder.config.output_hidden_states = True

        print('  [1.4] Running dual text encoding...')
        print('        LLaMA: tokenize -> run through 32 transformer layers -> extract layer[-3] features')
        print('        CLIP:  tokenize -> encode -> extract pooler output (single summary vector)')
        llama_vec, clip_l_pooler = encode_prompt_conds(args.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        print(f'        LLaMA output shape: {llama_vec.shape}  (1 batch, {llama_vec.shape[1]} tokens, {llama_vec.shape[2]} dims)')
        print(f'        CLIP pooler shape:  {clip_l_pooler.shape}  (1 batch, {clip_l_pooler.shape[1]} dims)')

        print('  [1.5] Creating empty negative embeddings (CFG=1, no negative prompt needed)...')
        llama_vec_n = torch.zeros_like(llama_vec)
        clip_l_pooler_n = torch.zeros_like(clip_l_pooler)

        print('  [1.6] Padding/cropping text embeddings to fixed length of 512 tokens...')
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        real_tokens = int(llama_attention_mask.sum())
        print(f'        After padding: {llama_vec.shape} — {real_tokens} real tokens, {512 - real_tokens} padding tokens')
        print(f'        Attention mask: {llama_attention_mask.shape} (True=real, False=padding)')

        print(f'  Text encoding done in {time.time() - t1:.1f}s')

        # ── Step 2: Image processing ───────────────────────────────────────
        print('\n>>> STEP 2/5: IMAGE PROCESSING')
        print('-' * 70)
        t2 = time.time()

        print(f'  [2.1] Loading image: {args.image}')
        input_image = np.array(Image.open(args.image).convert('RGB'))
        H, W, C = input_image.shape
        print(f'        Original size: {W}x{H} (width x height), {C} channels, dtype={input_image.dtype}')

        print('  [2.2] Finding nearest resolution bucket (model was trained on specific sizes)...')
        height, width = find_nearest_bucket(H, W, resolution=640)
        print(f'        Selected bucket: {width}x{height} (best aspect ratio match from ~640p buckets)')

        print('  [2.3] Resize + center crop to exact bucket dimensions...')
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        print(f'        Result: {input_image_np.shape[1]}x{input_image_np.shape[0]}')

        ref_path = os.path.join(args.output, f'{job_id}.png')
        Image.fromarray(input_image_np).save(ref_path)
        print(f'        Saved reference image: {ref_path}')

        print('  [2.4] Converting image to pytorch tensor...')
        print(f'        Pixel range [0, 255] -> normalize to [-1, 1]')
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        print(f'        Shape before permute: {input_image_pt.shape} (H, W, C)')
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        print(f'        Shape after permute:  {input_image_pt.shape} (Batch=1, C=3, T=1, H={height}, W={width})')

        print(f'  Image processing done in {time.time() - t2:.1f}s')

        # ── Step 3: VAE + CLIP Vision encoding ─────────────────────────────
        print('\n>>> STEP 3/5: ENCODING IMAGE TO LATENT SPACE')
        print('-' * 70)
        t3 = time.time()

        print('  [3.1] Loading VAE to GPU...')
        load_model_as_complete(vae, target_device=gpu)

        print('  [3.2] VAE encoding: compressing image pixels -> latent representation...')
        print(f'        Input:  {input_image_pt.shape}  (pixels: 3 channels, {height}x{width})')
        start_latent = vae_encode(input_image_pt, vae)
        print(f'        Output: {start_latent.shape}  (latent: {start_latent.shape[1]} channels, '
              f'{start_latent.shape[3]}x{start_latent.shape[4]})')
        print(f'        Compression: {height}x{width} -> {start_latent.shape[3]}x{start_latent.shape[4]} '
              f'(8x spatial downscale)')
        print(f'        Channels: 3 RGB -> {start_latent.shape[1]} latent channels')

        print('  [3.3] Loading SigLIP image encoder to GPU (unloads VAE first)...')
        load_model_as_complete(image_encoder, target_device=gpu)

        print('  [3.4] CLIP Vision encoding: understanding image content...')
        print(f'        Input: {input_image_np.shape} numpy array (uint8)')
        print(f'        The image is split into 14x14 pixel patches -> each patch becomes a vector')
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print(f'        Output: {image_encoder_last_hidden_state.shape}  '
              f'({image_encoder_last_hidden_state.shape[1]} patches, '
              f'{image_encoder_last_hidden_state.shape[2]} dims each)')

        print('  [3.5] Casting all embeddings to transformer dtype (bfloat16)...')
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        print(f'        All embeddings now in {transformer.dtype}')

        print(f'  Encoding done in {time.time() - t3:.1f}s')

        # Save checkpoint — this is the most expensive part to redo
        save_checkpoint('encodings', {
            'llama_vec': llama_vec.cpu(),
            'llama_vec_n': llama_vec_n.cpu(),
            'clip_l_pooler': clip_l_pooler.cpu(),
            'clip_l_pooler_n': clip_l_pooler_n.cpu(),
            'llama_attention_mask': llama_attention_mask.cpu(),
            'llama_attention_mask_n': llama_attention_mask_n.cpu(),
            'start_latent': start_latent.cpu(),
            'image_encoder_last_hidden_state': image_encoder_last_hidden_state.cpu(),
            'height': torch.tensor(height),
            'width': torch.tensor(width),
        })

    # ── Step 4: Diffusion sampling ─────────────────────────────────────────
    print('\n>>> STEP 4/5: DIFFUSION SAMPLING (this is the slow part)')
    print('-' * 70)
    t4 = time.time()

    print(f'  [4.1] Creating random noise generator with seed={args.seed}...')
    rnd = torch.Generator("cpu").manual_seed(args.seed)
    num_frames = LATENT_WINDOW_SIZE * 4 - 3
    print(f'        Each section: {LATENT_WINDOW_SIZE} latent frames -> {num_frames} pixel frames')

    print(f'  [4.2] Initializing history buffers...')
    history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
    print(f'        history_latents: {history_latents.shape}')
    print(f'        The 19 frames (1+2+16) are: 1x recent + 2x medium + 16x compressed context')
    history_pixels = None
    total_generated_latent_frames = 0

    print(f'  [4.3] Computing section schedule (inverted order)...')
    latent_paddings = list(reversed(range(total_latent_sections)))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
    print(f'        Padding sequence: {latent_paddings}')
    print(f'        (Higher padding = further from start of video, generated first)')

    # Check which sections are already done
    start_section = 0
    for si in range(len(latent_paddings)):
        section_ckpt = load_checkpoint(f'section_{si}')
        if section_ckpt is not None:
            history_latents = section_ckpt['history_latents']
            total_generated_latent_frames = int(section_ckpt['total_generated_latent_frames'])
            if 'history_pixels' in section_ckpt and section_ckpt['history_pixels'] is not None:
                history_pixels = section_ckpt['history_pixels']
            # Advance the RNG to the same state by generating the same random numbers
            for _ in range(si + 1):
                torch.randn((1, 16, (num_frames + 3) // 4, height // 8, width // 8), generator=rnd)
            start_section = si + 1
            print(f'  Restored section {si + 1}/{len(latent_paddings)} from checkpoint')
            print(f'  history_latents: {history_latents.shape}, total_frames: {total_generated_latent_frames}')
        else:
            break

    if start_section > 0:
        print(f'  Resuming from section {start_section + 1}/{len(latent_paddings)}')
    else:
        print(f'  Starting from scratch, {len(latent_paddings)} sections')

    output_filename = None

    for section_idx, latent_padding in enumerate(latent_paddings):
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * LATENT_WINDOW_SIZE
        section_start = time.time()

        print(f'\n  ---- Section {section_idx + 1}/{total_latent_sections} ----')
        print(f'  Padding: {latent_padding} x {LATENT_WINDOW_SIZE} = {latent_padding_size} blank frames')
        print(f'  Is last section (start of video): {is_last_section}')

        print(f'  [4.4.{section_idx}] Building position indices for Frame Packing...')
        total_index_length = 1 + latent_padding_size + LATENT_WINDOW_SIZE + 1 + 2 + 16
        print(f'        Total index length: {total_index_length} = '
              f'1(start) + {latent_padding_size}(padding) + {LATENT_WINDOW_SIZE}(noisy) + '
              f'1(recent_1x) + 2(recent_2x) + 16(old_4x)')
        indices = torch.arange(0, total_index_length).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, LATENT_WINDOW_SIZE, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        print(f'        Noisy latent indices: {latent_indices[0].tolist()[:5]}... (the frames being generated)')
        print(f'        Clean 1x indices:     {clean_latent_indices[0].tolist()} (full-res context)')
        print(f'        Clean 2x indices:     {clean_latent_2x_indices[0].tolist()} (2x compressed)')
        print(f'        Clean 4x indices:     {clean_latent_4x_indices[0].tolist()[:5]}... (4x compressed, 16 frames)')

        print(f'  [4.5.{section_idx}] Preparing clean latent context (Frame Packing)...')
        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        print(f'        clean_latents (1x): {clean_latents.shape}  — start frame + most recent frame')
        print(f'        clean_latents_2x:   {clean_latents_2x.shape}  — 2x compressed recent history')
        print(f'        clean_latents_4x:   {clean_latents_4x.shape}  — 4x compressed full history')

        print(f'  [4.6.{section_idx}] Moving transformer to GPU (partial load, respecting {args.gpu_reserve}GB reserve)...')
        unload_complete_models()
        move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=args.gpu_reserve)

        print(f'  [4.7.{section_idx}] Initializing TeaCache: enabled={args.teacache}')
        if args.teacache:
            print(f'        TeaCache will skip transformer blocks when input change < threshold')
            print(f'        This gives ~2x speedup but may produce slightly worse hands/fingers')
        transformer.initialize_teacache(enable_teacache=args.teacache, num_steps=args.steps)

        print(f'  [4.8.{section_idx}] Starting {args.steps}-step denoising with UniPC sampler...')
        print(f'        UniPC: Unified Predictor-Corrector ODE solver (order 3)')
        print(f'        Each step: transformer forward pass -> predict clean video -> update latent')
        step_times = []

        def callback(d):
            current_step = d['i'] + 1
            total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
            video_len = max(0, total_frames / 30)

            elapsed = time.time() - section_start
            avg_per_step = elapsed / current_step if current_step > 0 else 0
            remaining = avg_per_step * (args.steps - current_step)

            print(f'        Step {current_step:3d}/{args.steps} | '
                  f'~{avg_per_step:.1f}s/step | '
                  f'ETA {remaining:.0f}s | '
                  f'{total_frames} total frames | '
                  f'{video_len:.1f}s video')

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

        print(f'        Sampling done. Generated latent shape: {generated_latents.shape}')
        print(f'        Section took {time.time() - section_start:.1f}s')

        if is_last_section:
            print(f'  [4.9.{section_idx}] Last section — prepending start frame to generated latents')
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            print(f'        After prepend: {generated_latents.shape}')

        total_generated_latent_frames += int(generated_latents.shape[2])
        print(f'  Total generated latent frames so far: {total_generated_latent_frames}')

        print(f'  [4.10.{section_idx}] Prepending to history (inverted order: new frames go to front)...')
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        print(f'        history_latents shape: {history_latents.shape}')

        # ── Step 5: VAE decode this section ────────────────────────────────
        print(f'\n>>> STEP 5/5: VAE DECODING (section {section_idx + 1})')
        print('-' * 70)
        t5 = time.time()

        print(f'  [5.1] Offloading transformer, loading VAE to GPU...')
        offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
        load_model_as_complete(vae, target_device=gpu)

        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        print(f'  [5.2] Decoding latents -> pixels...')
        print(f'        Latent shape to decode: {real_history_latents.shape}')

        if history_pixels is None:
            print(f'        First section — decoding all latent frames at once')
            history_pixels = vae_decode_safe(real_history_latents, vae).cpu()
            print(f'        Decoded pixel shape: {history_pixels.shape}')
            print(f'        Pixel frames: {history_pixels.shape[2]}, Resolution: {history_pixels.shape[3]}x{history_pixels.shape[4]}')
        else:
            section_latent_frames = (LATENT_WINDOW_SIZE * 2 + 1) if is_last_section else (LATENT_WINDOW_SIZE * 2)
            overlapped_frames = LATENT_WINDOW_SIZE * 4 - 3
            print(f'        Decoding section: {section_latent_frames} latent frames')
            print(f'        Overlap with previous: {overlapped_frames} pixel frames (for smooth blending)')
            current_pixels = vae_decode_safe(real_history_latents[:, :, :section_latent_frames], vae).cpu()
            print(f'        Decoded section pixels: {current_pixels.shape}')
            print(f'        Blending with history using soft_append (linear interpolation in overlap zone)...')
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            print(f'        After blending: {history_pixels.shape}')

        unload_complete_models()

        output_filename = os.path.join(args.output, f'{job_id}_{total_generated_latent_frames}.mp4')

        print(f'  [5.3] Saving MP4 (fps=30, crf={args.crf})...')
        save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=args.crf)

        total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
        print(f'        Saved: {output_filename}')
        print(f'        {total_frames} frames, {total_frames/30:.1f}s video')
        print(f'  Decode + save took {time.time() - t5:.1f}s')

        # Checkpoint this section (save latents + pixels for resume)
        save_checkpoint(f'section_{section_idx}', {
            'history_latents': history_latents.cpu(),
            'total_generated_latent_frames': torch.tensor(total_generated_latent_frames),
            'history_pixels': history_pixels.cpu() if history_pixels is not None else None,
        })

        if is_last_section:
            print(f'\n  Last section complete — video generation finished!')
            break

    # Clean up checkpoints on success
    total_time = time.time() - gen_start
    print('\n' + '=' * 70)
    print(f'DONE! Total generation time: {total_time:.1f}s ({total_time/60:.1f} min)')
    print(f'Final video: {output_filename}')
    total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
    print(f'Video: {total_frames} frames, {total_frames/30:.1f}s at 30fps')
    print(f'Speed: {total_time/max(total_frames,1):.2f}s per frame')
    print('=' * 70)

    clean_checkpoints()
    return output_filename


if __name__ == '__main__':
    generate()
