# FramePack - A Deep Learning Guide for Aspiring Data Scientists

> **Who is this for?** You know Python. You know basic PyTorch (tensors, models, `.to(device)`).
> You want to understand how this codebase **actually generates a video from an image and text**.
> This guide walks through every concept — encoders, decoders, transformers, diffusion, sampling —
> with real code from this repository and plain-English explanations.

---

## Table of Contents

1. [The 30-Second Summary](#1-the-30-second-summary)
2. [What is a Diffusion Model?](#2-what-is-a-diffusion-model)
3. [The Complete Pipeline (Start Here)](#3-the-complete-pipeline-start-here)
4. [Step 1 — Text Encoding: Turning Words into Numbers](#4-step-1--text-encoding-turning-words-into-numbers)
5. [Step 2 — Image Encoding with CLIP Vision](#5-step-2--image-encoding-with-clip-vision)
6. [Step 3 — VAE: The Compression Engine](#6-step-3--vae-the-compression-engine)
7. [Step 4 — The Transformer: The Brain](#7-step-4--the-transformer-the-brain)
8. [Step 5 — Sampling: Removing Noise Step by Step](#8-step-5--sampling-removing-noise-step-by-step)
9. [Step 6 — Frame Packing: The Key Innovation](#9-step-6--frame-packing-the-key-innovation)
10. [Step 7 — VAE Decoding: Latents Back to Pixels](#10-step-7--vae-decoding-latents-back-to-pixels)
11. [Step 8 — Stitching Sections into a Video](#11-step-8--stitching-sections-into-a-video)
12. [TeaCache: Skipping Unnecessary Work](#12-teacache-skipping-unnecessary-work)
13. [Memory Management: Running 13B Models on 6GB GPUs](#13-memory-management-running-13b-models-on-6gb-gpus)
14. [Glossary](#14-glossary)

---

## 1. The 30-Second Summary

FramePack takes **one image** and **one text prompt** and generates a **video**.

It does this by:
1. Converting the text into number-arrays (vectors) that the AI can understand.
2. Converting the image into a compressed representation (latent).
3. Using a giant neural network (Transformer) to predict "what the next frames should look like" in that compressed space.
4. Decompressing those predicted frames back into actual pixels.
5. Repeating steps 3-4 to extend the video, section by section.

The magic trick: previously generated frames are compressed at multiple scales (1x, 2x, 4x smaller) so the Transformer always sees the same amount of data, no matter how long the video is. **That is Frame Packing.**

---

## 2. What is a Diffusion Model?

Before we dive into code, you need one mental model.

### The Core Idea

Imagine you have a photo. You add random noise to it — like TV static — until it becomes pure noise. A diffusion model **learns to reverse this process**: given pure noise, it removes noise step by step until a clean image appears.

```
Clean Image  -->  add noise -->  add more noise -->  ... -->  Pure Random Noise
Pure Random Noise  -->  remove noise -->  remove noise -->  ... -->  Clean Image (generated!)
```

**Training**: The model sees millions of examples of "noisy image at step t" and learns to predict "what was the clean image?"

**Generation**: Start with pure random noise, then ask the model to denoise it step by step. Each step, the image gets a little cleaner. After 25 steps, you have a brand new image.

### Video Diffusion

Same idea, but instead of a 2D image, the model works on a 3D block: `(frames, height, width)`. It removes noise from an entire video clip at once.

### What is "Latent" Diffusion?

Working directly on pixels is expensive. A 640x480 image has 921,600 numbers (per color channel). So we first **compress** the image to a much smaller "latent" representation using a VAE (we'll learn this in Step 3), run diffusion in that compressed space, then **decompress** back to pixels.

```
Pixel Space (big):   [3 channels, 640 height, 480 width]     = 921,600 numbers
Latent Space (small): [16 channels, 80 height, 60 width]     = 76,800 numbers
                                                                ~12x smaller!
```

All the heavy computation happens in latent space. This is why it's called **Latent Diffusion**.

---

## 3. The Complete Pipeline (Start Here)

Open `demo_gradio.py`. Ignore the Gradio UI code. The entire generation logic lives in the `worker()` function (line 103). Here is the pipeline in order:

```
Input: [Image (numpy array)] + [Text Prompt (string)]
                    |
                    v
    +-------------------------------+
    | STEP 1: Text Encoding         |
    |   LLaMA  --> text features    |  (diffusers_helper/hunyuan.py)
    |   CLIP   --> text pool        |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | STEP 2: Image Encoding (CLIP) |
    |   SigLIP --> image features   |  (diffusers_helper/clip_vision.py)
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | STEP 3: VAE Encode            |
    |   Image pixels --> latent     |  (diffusers_helper/hunyuan.py)
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | STEP 4: Diffusion Loop        |  (loops multiple "sections")
    |   For each section:           |
    |     - Pack frame context      |  <-- The key innovation
    |     - Denoise with Transformer|  (diffusers_helper/models/hunyuan_video_packed.py)
    |     - Using UniPC sampler     |  (diffusers_helper/k_diffusion/uni_pc_fm.py)
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | STEP 5: VAE Decode            |
    |   Latent --> pixel frames     |  (diffusers_helper/hunyuan.py)
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | STEP 6: Stitch & Save         |
    |   Blend overlaps, write MP4   |  (diffusers_helper/utils.py)
    +-------------------------------+
                    |
                    v
Output: [MP4 video file]
```

Let's walk through each step.

---

## 4. Step 1 — Text Encoding: Turning Words into Numbers

**File:** `diffusers_helper/hunyuan.py` - function `encode_prompt_conds()`

### The Problem

Neural networks don't understand words. They understand numbers — specifically, arrays of floating-point numbers called **vectors**. We need to convert `"The girl dances gracefully"` into a numeric representation that captures its *meaning*.

### Why TWO Text Encoders?

FramePack uses two different text encoders, and they each capture different things:

| Encoder | Model | What it captures | Output shape |
|---------|-------|-----------------|--------------|
| **LLaMA** (text_encoder) | A large language model (like a small ChatGPT) | Deep semantic meaning, relationships between words, context | `[1, up_to_256, 4096]` — a vector per token |
| **CLIP-L** (text_encoder_2) | A vision-language model | How text relates to visual concepts | `[1, 768]` — one single summary vector |

Think of it like this:
- **LLaMA** gives you a detailed essay describing what each word means in context.
- **CLIP** gives you a single "vibe check" — one vector that summarizes the whole sentence's visual energy.

### How the Code Works

```python
# From diffusers_helper/hunyuan.py, line 8

def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2):
```

**Step 1 — Tokenization:**

A tokenizer converts text into "token IDs" — numbers from a vocabulary.

```python
# "The girl dances" might become [464, 2576, 29621]
# Each number is an index in a vocabulary of ~32,000 words/subwords
llama_inputs = tokenizer(prompt_llama, padding="max_length", max_length=256 + crop_start, ...)
```

Think of a vocabulary like a Python dictionary:
```python
vocabulary = {"the": 464, "girl": 2576, "dances": 29621, "gracefully": 18232, ...}
# tokenizer("The girl dances") --> [464, 2576, 29621]
```

**Step 2 — Prompt Template:**

The prompt is wrapped in a template before being fed to LLaMA:
```python
prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
# Turns "The girl dances" into something like:
# "<|system|>Describe the video.<|user|>The girl dances<|end|>"
```

This is because LLaMA was trained on instruction-formatted text.

**Step 3 — LLaMA Encoding:**

```python
llama_outputs = text_encoder(input_ids=llama_input_ids, attention_mask=llama_attention_mask,
                              output_hidden_states=True)
llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
```

This is a critical detail: we don't take the model's final output. We take `hidden_states[-3]` — the output of the **3rd-to-last layer**. Why?

> **Concept: Hidden States**
>
> A neural network has many layers stacked on top of each other. Each layer transforms the data
> and passes it to the next one. The output of each layer is called a "hidden state."
>
> ```
> Layer 1:  [raw token embeddings]         <-- too simple, just word meanings
> Layer 2:  [slightly enriched]
> ...
> Layer N-3: [rich semantic features]       <-- we use THIS (good balance)
> Layer N-2: [even more abstract]
> Layer N-1: [task-specific features]
> Layer N:   [final prediction logits]      <-- too specialized for text generation
> ```
>
> The last layers are too specialized for LLaMA's original job (predicting next words).
> The 3rd-to-last layer has rich, general-purpose meaning — perfect for guiding image/video generation.

The `crop_start` skips the system-prompt tokens so we only keep the user's actual prompt.

**Step 4 — CLIP-L Encoding:**

```python
clip_l_pooler = text_encoder_2(clip_l_input_ids).pooler_output
# Shape: [1, 768] — one vector for the entire sentence
```

"Pooler output" means the model's summary of the entire input, compressed into a single vector. It acts as a global conditioning signal — "what is this text about, visually?"

### What Do These Outputs Look Like?

```python
llama_vec.shape   = [1, 256, 4096]
# Meaning: 1 batch, up to 256 tokens, each token is a 4096-dimensional vector

clip_l_pooler.shape = [1, 768]
# Meaning: 1 batch, one 768-dimensional summary vector
```

**Analogy**: Imagine describing a painting.
- `llama_vec` is like writing a 256-word detailed description, where each word is itself a nuanced concept (4096 dimensions of meaning).
- `clip_l_pooler` is like picking one paint color that best represents the overall mood.

### Padding and Masking

After encoding, the text is padded or cropped to exactly 512 tokens:

```python
# From demo_gradio.py, line 133
llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
```

This function (in `diffusers_helper/utils.py`, line 477) works like this:

```python
def crop_or_pad_yield_mask(x, length):
    B, F, C = x.shape  # Batch, Features (token count), Channels (vector size)

    if F < length:
        # Text is shorter than 512 tokens -- pad with zeros
        y = torch.zeros((B, length, C))   # Create a bigger all-zeros tensor
        mask = torch.zeros((B, length))    # Mask: 0 = "this position is fake padding"
        y[:, :F, :] = x                   # Copy real data into the beginning
        mask[:, :F] = True                 # Mark real positions as True
        return y, mask

    # Text is longer than 512 tokens -- crop it
    return x[:, :length, :], torch.ones((B, length))  # All positions are real
```

**Why padding?** The Transformer needs a fixed-size input. If your prompt is only 10 tokens, the remaining 502 positions are filled with zeros, and the mask tells the model to ignore them.

**Example:**
```
Real tokens:    ["the", "girl", "dances"]
After padding:  ["the", "girl", "dances", PAD, PAD, PAD, ..., PAD]  (512 total)
Mask:           [True,   True,   True,  False, False, False, ..., False]
```

### Negative Prompt (Classifier-Free Guidance)

```python
# From demo_gradio.py, line 128
if cfg == 1:
    llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
else:
    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, ...)
```

The "negative prompt" is used for **Classifier-Free Guidance (CFG)** — a trick to make the model follow the prompt more strongly. We'll cover this in detail in [Step 5 (Sampling)](#8-step-5--sampling-removing-noise-step-by-step).

When CFG scale = 1 (the default in this UI), the negative embeddings are just zeros (unconditional), meaning: "here's what the model would produce with NO text guidance."

---

## 5. Step 2 — Image Encoding with CLIP Vision

**File:** `diffusers_helper/clip_vision.py`

### What is CLIP?

**CLIP** (Contrastive Language-Image Pre-training) is a model trained on millions of image-text pairs. It learned to understand the relationship between images and text. It can look at an image and produce a vector that represents "what this image is about."

FramePack uses a variant called **SigLIP** (Sigmoid Loss for Language-Image Pre-training) — a newer, improved version of CLIP.

### Why Do We Need This?

The input image isn't just something to copy — it's the **starting point** for the video. The model needs to understand what's IN the image (a person? a landscape? what are they wearing?) so it can generate plausible continuation frames.

The CLIP Vision encoder gives the Transformer a rich understanding of the image content, separate from the pixel-level latent (which the VAE handles).

### How the Code Works

```python
# From diffusers_helper/clip_vision.py

def hf_clip_vision_encode(image, feature_extractor, image_encoder):
    # image is a numpy array: shape [Height, Width, 3], values 0-255

    # Step 1: Preprocess -- resize, normalize, convert to tensor
    preprocessed = feature_extractor.preprocess(images=image, return_tensors="pt")
    # This resizes the image to 384x384 (SigLIP's expected size),
    # normalizes pixel values to [-1, 1] range, and converts to a PyTorch tensor

    # Step 2: Run through the vision transformer
    image_encoder_output = image_encoder(**preprocessed)
    # Output contains:
    #   .last_hidden_state  shape: [1, 729, 1152]
    #       729 = 27x27 patches (the image is split into a grid)
    #       1152 = vector dimension for each patch
    #   .pooler_output      shape: [1, 1152]
    #       A single summary vector for the whole image

    return image_encoder_output
```

### How Vision Transformers Split an Image into "Tokens"

Just like a text encoder turns words into tokens, a vision encoder turns image **patches** into tokens.

```
Original image (384 x 384 pixels):
+--+--+--+--+--+--+--+     ...  +--+
|  |  |  |  |  |  |  |          |  |
+--+--+--+--+--+--+--+     ...  +--+
|  |  |  |  |  |  |  |          |  |    Each small square = one 14x14 pixel patch
+--+--+--+--+--+--+--+     ...  +--+
|  |  |  |  |  |  |  |          |  |    384 / 14 = ~27 patches per side
  ...                                   27 x 27 = 729 patches total
+--+--+--+--+--+--+--+     ...  +--+

Each patch becomes a 1152-dimensional vector.
Result: [729 patches, 1152 dimensions] = [729, 1152]
```

**Analogy**: It's like cutting a photo into a 27x27 grid of small tiles, then having an expert describe each tile with 1152 numbers that capture its content.

### Where This Output Goes

```python
# From demo_gradio.py, line 166
image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
# Shape: [1, 729, 1152]
```

This gets passed to the Transformer later, where it's projected and prepended to the text tokens. So the Transformer sees: `[image_patches, text_tokens, video_patches]` and can attend to all of them.

---

## 6. Step 3 — VAE: The Compression Engine

**File:** `diffusers_helper/hunyuan.py` - functions `vae_encode()` and `vae_decode()`

### What is a VAE?

A **Variational Autoencoder** is a neural network with two halves:

```
            ENCODER                              DECODER
 [Big Image] -----> [Small Latent] -----> [Reconstructed Image]
 (640x480x3)        (80x60x16)            (640x480x3)
  921,600 numbers    76,800 numbers        921,600 numbers
```

The encoder **compresses** images. The decoder **decompresses** them. The small thing in the middle is called the **latent representation** — it's like a ZIP file of the image.

The VAE is trained so that `Decoder(Encoder(image)) ≈ image`. It learns to keep only the important information and throw away the noise.

### Why 16 Channels?

A regular image has 3 channels (Red, Green, Blue). But the latent space has **16 channels**. Why?

3 channels aren't enough to capture all the information in a compressed space. The VAE learns its own "channels" — you can think of them as 16 different "aspects" of the image:

```
Channel 0:  might capture brightness patterns
Channel 1:  might capture edge locations
Channel 2:  might capture color hue
Channel 3:  might capture texture frequency
...
Channel 15: might capture some other abstract feature
```

These aren't manually designed — the network discovers what's useful during training.

### The 3D VAE for Video

This isn't a regular image VAE — it's a **3D VAE** that understands time. It compresses along all three dimensions:

```
Input video:   [3 channels,  Frames,  Height,  Width ]
Latent:        [16 channels, Frames/4, Height/8, Width/8]
```

- Height and Width are compressed **8x** each
- Time (Frames) is compressed **4x**
- Channels expand from 3 to 16

So a 33-frame clip at 640x480 becomes a latent of shape `[16, 9, 80, 60]`.

### Encoding (Image to Latent)

```python
# From diffusers_helper/hunyuan.py, line 107

def vae_encode(image, vae):
    # image shape: [1, 3, 1, Height, Width]  (Batch, Channels, Time=1 frame, H, W)

    latents = vae.encode(image).latent_dist.sample()
    # .encode() gives a probability distribution (not a single point!)
    # .sample() picks one point from that distribution
    # This randomness is what makes VAEs "variational"

    latents = latents * vae.config.scaling_factor
    # Scale the latent values to a nice range for the diffusion model
    # scaling_factor is typically ~0.13 — keeps values near [-1, 1]

    return latents
    # Shape: [1, 16, 1, Height//8, Width//8]
```

**Why `.latent_dist.sample()`?**

Unlike a regular autoencoder that produces one fixed compressed representation, a VAE produces a **distribution** (defined by a mean and variance). We then sample from that distribution. This is what makes generation possible — during generation, we can sample different points from latent space to get different outputs.

```
Regular Autoencoder:   image --> [one fixed point in latent space]
VAE:                   image --> [mean, variance] --> sample a point
```

### The Scaling Factor

```python
latents = latents * vae.config.scaling_factor  # ~0.13
```

Raw latent values might be in range [-20, 20]. The diffusion model expects values near [-1, 1]. The scaling factor normalizes them. Think of it like converting Celsius to a 0-1 scale — same information, just re-ranged.

### Decoding (Latent to Pixels)

```python
# From diffusers_helper/hunyuan.py, line 93

def vae_decode(latents, vae):
    latents = latents / vae.config.scaling_factor    # Undo the scaling
    image = vae.decode(latents).sample               # Decompress back to pixels
    return image
    # Output shape: [1, 3, Frames, Height, Width], values roughly in [-1, 1]
```

### Fake Decoding (Fast Preview)

Full VAE decoding is slow. For the live preview during generation, a **fake decode** is used:

```python
# From diffusers_helper/hunyuan.py, line 62

def vae_decode_fake(latents):
    # Instead of running the full decoder neural network,
    # just multiply the 16-channel latent by a 16x3 matrix
    # to project it down to 3 RGB channels.

    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],    # How much channel 0 contributes to R, G, B
        [0.0696, 0.0795, 0.0518],       # How much channel 1 contributes to R, G, B
        ...                              # (16 rows total, one per latent channel)
    ]

    weight = torch.tensor(latent_rgb_factors).transpose(0, 1)  # Shape: [3, 16, 1, 1, 1]
    images = torch.nn.functional.conv3d(latents, weight, bias=bias)
    # This is a 1x1x1 convolution = just a matrix multiply per spatial position
    # Input:  [1, 16, T, H, W]  (16 channels)
    # Output: [1, 3,  T, H, W]  (3 channels = RGB)

    return images  # Blurry but fast preview!
```

This is like a "rough sketch" — the colors and shapes are approximately right but the details are blurry. It takes almost zero time compared to the full decode.

### VAE Tiling and Slicing (Low VRAM)

```python
# From demo_gradio.py, line 67
if not high_vram:
    vae.enable_slicing()   # Decode one frame at a time instead of all at once
    vae.enable_tiling()    # Decode in spatial tiles instead of the full resolution
```

On low-VRAM GPUs, decoding a full video latent at once would run out of memory. These options break the work into smaller pieces. Slower, but uses less memory.

---

## 7. Step 4 — The Transformer: The Brain

**File:** `diffusers_helper/models/hunyuan_video_packed.py`

This is the largest and most important file. The Transformer is the 13-billion-parameter neural network that actually **predicts what the video should look like**. Everything else is just preparing inputs or processing outputs.

### What is a Transformer?

A Transformer is a neural network architecture built around one key operation: **Attention**.

**Attention** answers the question: "When generating part X of the output, how much should I look at each part of the input?"

```
Generating frame 5 of a dance video:
- Look at frame 4 A LOT (what pose comes next?)
- Look at frame 1 A LITTLE (remember the starting position)
- Look at the text "dancing" A LOT (what motion to generate?)
- Look at the text "the" VERY LITTLE (not very informative)
```

The Transformer learns these attention patterns from training data.

### The Architecture Overview

This model (HunyuanVideo) is a **Diffusion Transformer (DiT)** — a Transformer specifically designed for diffusion models. Here's the architecture:

```
INPUTS:
  [Noisy video latent] + [Clean context latents (1x, 2x, 4x)] + [Text tokens] + [Image tokens]
     |
     v
  +---------------------------------+
  | Patch Embedding                 |  Convert latent "pixels" to Transformer tokens
  +---------------------------------+
     |
     v
  +---------------------------------+
  | Rotary Position Embeddings      |  Tell the model WHERE each token is in 3D space
  +---------------------------------+
     |
     v
  +---------------------------------+
  | 20x Dual-Stream Blocks          |  Image tokens and text tokens TALK to each other
  +---------------------------------+
     |
     v
  +---------------------------------+
  | 40x Single-Stream Blocks        |  Image and text MERGED into one stream
  +---------------------------------+
     |
     v
  +---------------------------------+
  | Output Projection               |  Convert back to latent pixel shape
  +---------------------------------+
     |
     v
OUTPUT:
  [Predicted clean video latent]
```

Let's go through each part.

### 7.1 Patch Embedding — Turning Latents into Tokens

```python
# From hunyuan_video_packed.py, line 694
class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        # patch_size = (1, 2, 2)  -- 1 frame, 2x2 spatial
        # in_chans = 16 (latent channels)
        # embed_dim = 3072 (inner_dim = 24 heads * 128 dim_per_head)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

Just like the CLIP Vision encoder splits an image into patches, this splits the video latent into patches:

```
Latent: [16 channels, 9 frames, 80 height, 60 width]

Patch size: (1, 2, 2) = 1 frame, 2x2 spatial

After patching:
  Temporal:  9 / 1 = 9 positions
  Height:    80 / 2 = 40 positions
  Width:     60 / 2 = 30 positions
  Total tokens: 9 * 40 * 30 = 10,800 tokens

Each token: 3072 dimensions (the "embedding dimension")
Result: [10,800 tokens, 3072 dimensions]
```

The Conv3D does two things at once: it extracts the patches AND projects them to the embedding dimension. Stride = kernel_size means non-overlapping patches.

### 7.2 Rotary Position Embeddings (RoPE) — Where Is Each Token?

A Transformer processes tokens as a **set** — it doesn't inherently know that token #5 is next to token #6. We need to explicitly tell it the position of each token.

**RoPE** (Rotary Position Embedding) does this by rotating the token vectors based on their position. Nearby tokens get similar rotations, far-away tokens get very different rotations.

```python
# From hunyuan_video_packed.py, line 425
class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        self.DT, self.DY, self.DX = rope_dim  # (16, 56, 56)
        # DT=16 dimensions encode the TIME position
        # DY=56 dimensions encode the Y (height) position
        # DX=56 dimensions encode the X (width) position
        # Total: 16+56+56 = 128 = attention_head_dim
        self.theta = theta  # 256.0 — controls the frequency of rotations
```

**Why 3D RoPE?** Video has three spatial dimensions (time, height, width). Each token needs a position in all three. The RoPE frequencies are computed independently for each dimension and concatenated.

```python
def get_frequency(self, dim, pos):
    # This creates sinusoidal frequencies at different scales
    freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2) / dim))
    # theta=256, dim=16 gives frequencies like:
    # [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    # Low frequencies = detect coarse positions (near vs. far)
    # High frequencies = detect fine positions (adjacent patches)

    freqs = torch.outer(freqs, positions)  # Multiply each frequency by each position
    return freqs.cos(), freqs.sin()        # Return cos and sin for rotation
```

**Analogy**: Think of a clock.
- The hour hand (low frequency) tells you roughly what time it is.
- The minute hand (medium frequency) gives more precision.
- The second hand (high frequency) gives exact timing.

RoPE uses many "hands" at different speeds to encode position precisely.

### 7.3 Attention: How Tokens Communicate

The core operation. Let's build intuition.

**Every token asks three questions:**
1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I contain that others might want?"
3. **Value (V):** "What information do I provide when someone looks at me?"

```python
# Simplified attention (from HunyuanAttnProcessorFlashAttnDouble, line 143)

query = attn.to_q(hidden_states)   # [batch, tokens, heads * dim_per_head]
key = attn.to_k(hidden_states)
value = attn.to_v(hidden_states)

# Reshape to separate attention heads
query = query.unflatten(2, (num_heads, dim_per_head))
# From [1, 10800, 3072] to [1, 10800, 24, 128]
# 24 heads, each with 128 dimensions -- like 24 parallel attention operations

# Apply RoPE (position encoding)
query = apply_rotary_emb_transposed(query, image_rotary_emb)
key = apply_rotary_emb_transposed(key, image_rotary_emb)

# Compute attention: which tokens should pay attention to which?
# attention_score[i][j] = dot_product(query[i], key[j]) / sqrt(dim)
# This gives high scores when token i's "question" matches token j's "answer"

# Output: weighted sum of values, where weights = attention scores
output = scaled_dot_product_attention(query, key, value)
```

**Example with 4 tokens:**
```
Token 0 (sky patch):    Q="What color should I be?"
Token 1 (grass patch):  Q="What texture should I have?"
Token 2 (text "blue"):  Q="Who needs color info?"
Token 3 (text "sky"):   Q="Who is a sky region?"

Attention scores (who looks at whom):
         Key0   Key1   Key2   Key3
Query0:  0.1    0.0    0.7    0.8   <-- Token 0 (sky) attends to "blue" and "sky"
Query1:  0.0    0.3    0.1    0.0   <-- Token 1 (grass) mostly self-attends
Query2:  0.4    0.1    0.2    0.6   <-- "blue" attends to sky-related tokens
Query3:  0.8    0.0    0.5    0.3   <-- "sky" attends to the sky patch
```

### 7.4 Multi-Head Attention

This model uses **24 attention heads**, each with 128 dimensions. Why multiple heads?

Each head can learn a different type of relationship:
```
Head 0:  might learn spatial proximity (nearby patches attend to each other)
Head 1:  might learn color similarity
Head 2:  might learn temporal relationships (same region across frames)
Head 3:  might learn text-to-image alignment
...
Head 23: might learn some other pattern
```

The outputs of all heads are concatenated and projected back:
```python
# Each head: [1, 10800, 128]
# Concatenated: [1, 10800, 24 * 128] = [1, 10800, 3072]
# Then a linear layer projects 3072 --> 3072
```

### 7.5 Dual-Stream vs Single-Stream Blocks

The model has two types of transformer blocks:

**Dual-Stream Blocks (20 of them):**

```python
# From hunyuan_video_packed.py, line 608
class HunyuanVideoTransformerBlock:
    def forward(self, hidden_states, encoder_hidden_states, temb, ...):
        # hidden_states = video tokens
        # encoder_hidden_states = text tokens
        # These are processed SEPARATELY but attend to EACH OTHER

        # 1. Normalize both streams (with separate norm parameters)
        norm_hidden_states = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, emb=temb)

        # 2. Joint attention -- video and text tokens see each other
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states, ...
        )

        # 3. Separate feed-forward networks for each stream
        ff_output = self.ff(hidden_states)                       # Video FFN
        context_ff_output = self.ff_context(encoder_hidden_states)  # Text FFN

        return hidden_states, encoder_hidden_states
```

**Why separate streams?** Video and text are fundamentally different modalities. They need to attend to each other (so the video knows what the text says), but their internal transformations (feed-forward networks) should be specialized.

**Single-Stream Blocks (40 of them):**

```python
# From hunyuan_video_packed.py, line 534
class HunyuanVideoSingleTransformerBlock:
    def forward(self, hidden_states, encoder_hidden_states, temb, ...):
        # Here, video and text tokens are CONCATENATED into one sequence
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # Everything processed together -- one attention, one FFN
        # Then split back at the end
        hidden_states, encoder_hidden_states = hidden_states[:, :-text_len], hidden_states[:, -text_len:]
```

**Architecture reasoning:**
- First 20 blocks (dual-stream): Let video and text develop their own representations while communicating through attention. Like two teams collaborating but each doing their own internal work.
- Last 40 blocks (single-stream): Merge everything together for fine-grained integration. Like the two teams merging into one for the final push.

### 7.6 Adaptive Layer Normalization (AdaLN)

Regular Layer Normalization treats every input the same. **Adaptive** Layer Normalization adjusts its behavior based on a **conditioning signal** — in this case, the timestep.

```python
# From hunyuan_video_packed.py, line 463
class AdaLayerNormZero(nn.Module):
    def forward(self, x, emb):  # emb = timestep embedding
        emb = self.linear(self.silu(emb))  # Project timestep to 6x the hidden size
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)

        x = self.norm(x) * (1 + scale_msa) + shift_msa
        # Regular LayerNorm: normalize to mean=0, std=1
        # Then SHIFT and SCALE based on the timestep
        # This tells the model "behave differently at high noise vs low noise"

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

**Why does the timestep matter?**

At the beginning of denoising (high noise, t near 1.0), the model needs to figure out the overall structure — "is this a person or a landscape?" At the end (low noise, t near 0.0), it's refining details — "what exact shade of blue?"

AdaLN lets the model adapt its behavior at each noise level:
- `scale` controls how much to amplify/dampen features
- `shift` adds a bias
- `gate` controls how much the attention/FFN output affects the final result (0 = ignore, 1 = full effect)

### 7.7 Timestep and Guidance Embeddings

```python
# From hunyuan_video_packed.py, line 219
class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def forward(self, timestep, guidance, pooled_projection):
        # timestep: what noise level are we at? (e.g., 0.8 = very noisy)
        # guidance: how strongly to follow the prompt? (e.g., 10.0 = strongly)
        # pooled_projection: CLIP text summary vector

        timesteps_emb = self.timestep_embedder(self.time_proj(timestep))
        guidance_emb = self.guidance_embedder(self.time_proj(guidance))
        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + guidance_emb + pooled_projections
        return conditioning  # Shape: [batch, 3072]
```

This combines three signals into one conditioning vector:
1. **Timestep** — "how noisy is the current input?" (so the model knows how aggressively to denoise)
2. **Guidance scale** — "how creative vs. prompt-faithful should I be?"
3. **Text summary** — "what is this video about?" (the CLIP pooler output)

This conditioning vector is fed to every AdaLN in every block, influencing the model's behavior throughout.

### 7.8 Token Refiner — Enriching Text Representations

```python
# From hunyuan_video_packed.py, line 376
class HunyuanVideoTokenRefiner(nn.Module):
    def forward(self, hidden_states, timestep, attention_mask):
        # hidden_states: raw LLaMA text embeddings [1, 512, 4096]
        # This refiner is a small 2-layer transformer that enriches the text

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)  # Project 4096 --> 3072
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)
        # 2 blocks of self-attention + FFN with timestep conditioning

        return hidden_states  # [1, 512, 3072] -- same text, enriched and resized
```

**Why refine?** The LLaMA embeddings were trained for text generation, not video generation. The refiner adapts them to be more useful for this specific task, and also adjusts them based on the current timestep (since the text should guide the model differently at different noise levels).

### 7.9 The Image Projection

```python
# From hunyuan_video_packed.py, line 683
class ClipVisionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.up = nn.Linear(in_channels, out_channels * 3)   # 1152 -> 9216
        self.down = nn.Linear(out_channels * 3, out_channels) # 9216 -> 3072

    def forward(self, x):
        return self.down(nn.functional.silu(self.up(x)))
```

This projects the CLIP Vision features (1152 dim) to the Transformer's inner dimension (3072 dim) using an **expand-then-compress** pattern with a SiLU activation:

```
[1152] --up--> [9216] --SiLU--> [9216] --down--> [3072]
```

**Why expand then compress?** The wider intermediate layer lets the network learn more complex transformations. It's like explaining something in great detail (expand) and then summarizing (compress) — the summary is richer than if you just directly translated.

The projected image embeddings are **prepended** to the text tokens:
```python
# From the forward() method, line 930
encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
# [image_patches (729 tokens), text_tokens (512 tokens)] = 1241 "context" tokens
```

---

## 8. Step 5 — Sampling: Removing Noise Step by Step

**Files:**
- `diffusers_helper/pipelines/k_diffusion_hunyuan.py` — orchestrates the sampling
- `diffusers_helper/k_diffusion/wrapper.py` — wraps the transformer for the sampler
- `diffusers_helper/k_diffusion/uni_pc_fm.py` — the UniPC sampler algorithm

### The Sampling Pipeline

Here's what happens at a high level:

```
1. Generate pure random noise (the starting point)
2. Create a "noise schedule" — a plan for how to remove noise over 25 steps
3. For each step:
   a. Ask the Transformer: "Given this noisy video at noise level t, what's the clean video?"
   b. Use the answer to slightly denoise the video
   c. Move to the next noise level
4. After 25 steps, the video is (approximately) clean
```

### The Noise Schedule

```python
# From k_diffusion_hunyuan.py, line 9

def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
```

The noise schedule determines how much noise exists at each step. It goes from `sigma=1.0` (pure noise) to `sigma=0.0` (no noise).

But not linearly! The **time shift** concentrates more steps in the high-noise regime where the model needs to make the biggest decisions:

```
Linear schedule:     [1.0, 0.96, 0.92, 0.88, ..., 0.08, 0.04, 0.0]  (evenly spaced)
Time-shifted:        [1.0, 0.85, 0.72, 0.61, ..., 0.03, 0.01, 0.0]  (more steps early on)
```

**Why?** The early steps (high noise) decide the overall composition and motion. The late steps (low noise) just sharpen details. You want more "thinking time" for the big decisions.

```python
def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15):
    # mu controls HOW MUCH to shift the schedule
    # Longer sequences (more tokens) need more shift
    # because there are more spatial decisions to make
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    return mu
```

This adaptively scales the shift based on the sequence length. More tokens = more complex scene = need more steps in the early (structural) phase.

### The Model Wrapper

```python
# From wrapper.py, line 19

def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        # x: the current noisy latent
        # sigma: the current noise level (e.g., 0.7)

        timestep = sigma * t_scale  # Scale to model's expected range [0, 1000]

        # Optionally concatenate conditioning latent (not used in this model)
        if concat_latent is not None:
            hidden_states = torch.cat([x, concat_latent], dim=1)
        else:
            hidden_states = x

        # Run the Transformer TWICE:
        # 1. With the text prompt (positive prediction)
        pred_positive = transformer(hidden_states, timestep, **extra_args['positive'])

        # 2. Without the text prompt (negative prediction)
        pred_negative = transformer(hidden_states, timestep, **extra_args['negative'])

        # Classifier-Free Guidance (CFG)
        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        #
        # What this means:
        # Start with the "no text" prediction (pred_negative)
        # Add the DIFFERENCE that text makes, amplified by cfg_scale
        #
        # cfg_scale=1:  just use the positive prediction as-is
        # cfg_scale=7:  the text influence is amplified 7x
        # cfg_scale=15: very strong text adherence, but can cause artifacts

        # CFG Rescale (optional) — fix the "too saturated" problem from high CFG
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale)

        # Convert velocity prediction to x0 prediction:
        x0 = x - pred * sigma
        # The model predicts "velocity" (the direction from noise to clean)
        # x0 = clean image estimate = current_noisy - velocity * noise_level

        return x0

    return k_model
```

### Classifier-Free Guidance (CFG) — Deep Dive

This is one of the most important tricks in modern image/video generation. Let's build intuition.

**The Problem:** If you just ask the model "generate a video of a girl dancing," it might produce something vaguely related but not very specific.

**The Solution:** Run the model twice:
1. **With text** (conditional): "Here's what a video of 'a girl dancing' looks like"
2. **Without text** (unconditional): "Here's what any random video looks like"

Then amplify the difference:

```
result = unconditional + scale * (conditional - unconditional)
       = unconditional + scale * (what_the_text_adds)
```

```
scale=1.0:  result = conditional                    (just the normal prediction)
scale=7.0:  result = unconditional + 7 * difference (7x the text influence!)
scale=15.0: result = unconditional + 15 * difference (very strong, might over-saturate)
```

**Analogy**: Imagine asking an artist to draw a "happy sunny landscape."
- The unconditional prediction is a generic landscape.
- The conditional prediction is a happy sunny landscape.
- The difference is: more yellow, brighter sky, flowers.
- CFG scale = 1: normal amount of sunshine.
- CFG scale = 10: EXTREMELY sunny, very saturated yellows, almost too bright.

**CFG Rescale** fixes the over-saturation:

```python
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    # noise_cfg has inflated standard deviation from high CFG
    # noise_pred_text has the "correct" standard deviation
    # We rescale noise_cfg to match noise_pred_text's std

    std_text = noise_pred_text.std(...)
    std_cfg = noise_cfg.std(...)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # Blend between rescaled and original based on guidance_rescale
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
```

### The UniPC Sampler

```python
# From uni_pc_fm.py

class FlowMatchUniPC:
    # UniPC = Unified Predictor-Corrector
    # It's a solver for ODEs (Ordinary Differential Equations)
```

**Why is denoising an ODE?**

In flow matching, we think of the diffusion process as a continuous flow:

```
Pure Noise (t=1)  ---flows--->  Clean Image (t=0)
```

The model predicts the flow velocity at any point. The sampler follows this flow from t=1 to t=0 by taking discrete steps. This is exactly what an ODE solver does.

**UniPC's trick:** It uses information from **previous steps** to take more accurate steps. Like how a GPS that remembers where you've been can predict the road ahead better than one that only knows your current position.

```python
def update_fn(self, x, model_prev_list, t_prev_list, t, order):
    # x: current denoised estimate
    # model_prev_list: model predictions from previous steps
    # order: how many previous predictions to use (up to 3)

    # Uses polynomial interpolation of previous predictions
    # to estimate the trajectory and take a more accurate step

    # Predictor step: estimate where we'll be at time t
    x_t = x_t_ - B_h * pred_res   # Move along the predicted trajectory

    # Corrector step: evaluate the model at the predicted point,
    # then correct our estimate
    model_t = self.model_fn(x_t, t)
    x_t = x_t_ - B_h * (corr_res + rhos_c[-1] * D1_t)  # Corrected position

    return x_t, model_t
```

**Why order 3?**
- Order 1: Use only the current prediction. Like driving looking only at the road directly ahead.
- Order 2: Also use the previous prediction. Like remembering where the road curved last.
- Order 3: Use the last 3 predictions. Like a smooth GPS trajectory.

Higher order = fewer steps needed for the same quality. UniPC with order 3 can produce good results in just 25 steps.

---

## 9. Step 6 — Frame Packing: The Key Innovation

**File:** `diffusers_helper/models/hunyuan_video_packed.py` — method `process_input_hidden_states()`

This is what makes FramePack special. Here's the problem it solves:

### The Problem

In normal video generation, the model needs to look at ALL previously generated frames to generate the next one. As the video gets longer, the number of tokens grows linearly:

```
Frame 1:    10,800 tokens
Frame 1-2:  21,600 tokens
Frame 1-3:  32,400 tokens
...
Frame 1-30: 324,000 tokens  <-- IMPOSSIBLE to fit in GPU memory!
```

### The Solution: Multi-Scale Compression

Instead of feeding all previous frames at full resolution, FramePack compresses them at multiple scales:

```
Clean Latent 4x: Previous frames at 4x downsampling (very compressed)
                  Each frame compressed spatially AND temporally by 4x
                  A 16-frame history becomes ~1 token per frame

Clean Latent 2x: Recent frames at 2x downsampling (medium compression)
                  Each frame compressed by 2x spatially and temporally
                  More detail than 4x but still compact

Clean Latent 1x: Very recent frames at full resolution
                  The most recent 1-2 frames, uncompressed
                  Full detail for smooth transitions

Noisy Latent:    The new frames being generated (what we're denoising)
```

```
Total tokens = 4x_tokens + 2x_tokens + 1x_tokens + noisy_tokens
             = (small)   + (medium)  + (small)   + (fixed)
             = CONSTANT regardless of video length!
```

### How the Code Does It

```python
# From hunyuan_video_packed.py, line 839

def process_input_hidden_states(self, latents, latent_indices,
                                 clean_latents, clean_latent_indices,
                                 clean_latents_2x, clean_latent_2x_indices,
                                 clean_latents_4x, clean_latent_4x_indices):

    # 1. Embed the NOISY latent (the frames being generated)
    hidden_states = self.x_embedder.proj(latents)
    # Input: [1, 16, T, H/8, W/8]    (16 latent channels, T frames)
    # Output: [1, 3072, T, H/16, W/16] (3072 embedding dims, patches)

    # 2. Compute position embeddings for the noisy latent
    rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W)

    # 3. Embed CLEAN latent at 1x scale (most recent frames)
    if clean_latents is not None:
        clean_latents = self.clean_x_embedder.proj(clean_latents)
        # Same conv as noisy but separate weights -- different meaning!
        # kernel_size=(1,2,2), stride=(1,2,2)

        # Position embeddings at matching indices
        clean_latent_rope_freqs = self.rope(frame_indices=clean_latent_indices, ...)

        # PREPEND to the token sequence
        hidden_states = torch.cat([clean_latents, hidden_states], dim=1)
        rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)

    # 4. Embed CLEAN latent at 2x scale (slightly older frames, compressed 2x)
    if clean_latents_2x is not None:
        clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
        clean_latents_2x = self.clean_x_embedder.proj_2x(clean_latents_2x)
        # kernel_size=(2,4,4), stride=(2,4,4) -- 2x temporal, 4x spatial compression
        # A 2-frame, 80x60 region becomes a single token!

        # Downsample position embeddings to match
        clean_latent_2x_rope_freqs = center_down_sample_3d(
            self.rope(frame_indices=clean_latent_2x_indices, ...), (2, 2, 2)
        )

        hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
        rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

    # 5. Embed CLEAN latent at 4x scale (oldest frames, heavily compressed)
    if clean_latents_4x is not None:
        clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
        clean_latents_4x = self.clean_x_embedder.proj_4x(clean_latents_4x)
        # kernel_size=(4,8,8), stride=(4,8,8) -- 4x temporal, 8x spatial compression
        # A 4-frame, 160x120 region becomes a single token!

        hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
        rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

    return hidden_states, rope_freqs
```

### Visual Example

Let's say we've generated 60 frames and are generating frames 61-69:

```
WITHOUT Frame Packing (impossible on most GPUs):
[frame1][frame2]...[frame60][frame61_noisy]...[frame69_noisy]
= 60 * 10,800 + 9 * 10,800 = 745,200 tokens  (WAY too many)

WITH Frame Packing:
[frames1-16 at 4x] [frames17-32 at 4x] [frames33-48 at 4x] [frames49-56 at 2x] [frames57-60 at 1x] [frames61-69 noisy]
   ~170 tokens         ~170 tokens          ~170 tokens         ~1350 tokens        ~10800 tokens       ~10800 tokens
= ~23,460 tokens (CONSTANT, fits in any GPU!)
```

The oldest frames are the most compressed (4x = 64x fewer tokens), recent frames are less compressed (2x = 8x fewer tokens), and the most recent frames are at full resolution. The model can still "see" the entire history — just at lower resolution for older parts.

### Position Indices — How the Model Knows the Timeline

```python
# From demo_gradio.py, line 206
indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, \
    clean_latent_2x_indices, clean_latent_4x_indices = indices.split([...], dim=1)
```

Each token gets a position index that tells the model WHERE in the timeline it belongs. Even though the 4x-compressed tokens are physically adjacent to the noisy tokens in the sequence, their position indices say "I'm from frames 1-16" so the model knows they're from the distant past.

```
Position indices: [0] [1..pad..] [pad+1..pad+9] [pad+10] [pad+11, pad+12] [pad+13..pad+28]
                   ^      ^           ^              ^          ^                ^
                start  padding     noisy frames   recent 1x   recent 2x       old 4x
                frame  (blank)    (being generated)
```

---

## 10. Step 7 — VAE Decoding: Latents Back to Pixels

After the Transformer predicts the clean latent, we need to convert it back to actual video pixels.

```python
# From demo_gradio.py, line 285
history_pixels = vae_decode(real_history_latents, vae).cpu()
```

This calls:
```python
# From diffusers_helper/hunyuan.py
def vae_decode(latents, vae):
    latents = latents / vae.config.scaling_factor   # Undo the scaling from encoding
    image = vae.decode(latents).sample              # Run the decoder network
    return image
    # Input:  [1, 16, T_latent, H/8, W/8]   latent space
    # Output: [1, 3, T_latent*4, H, W]       pixel space (upsampled!)
```

The decoder mirrors the encoder:
- 16 channels --> 3 channels (RGB)
- Height * 8, Width * 8 (spatial upsampling)
- Frames * 4 (temporal upsampling)

---

## 11. Step 8 — Stitching Sections into a Video

The video is generated section by section. Each section overlaps with the previous one. We need to blend the overlapping parts smoothly.

```python
# From diffusers_helper/utils.py, line 252

def soft_append_bcthw(history, current, overlap):
    # history: all previously generated pixel frames  [1, 3, 100, H, W]
    # current: newly generated pixel frames           [1, 3, 33, H, W]
    # overlap: number of overlapping frames            e.g., 33

    # Create a linear blend weight: goes from 1.0 to 0.0 over the overlap region
    weights = torch.linspace(1, 0, overlap)
    # weights = [1.0, 0.97, 0.94, ..., 0.06, 0.03, 0.0]

    # Blend: gradually transition from history to current
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    # At the start of overlap:  100% history, 0% current (smooth continuation)
    # In the middle:            50% history, 50% current (mixing)
    # At the end of overlap:    0% history, 100% current (fully new)

    # Concatenate: [history without overlap] + [blended region] + [current without overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output
```

**Why overlap?** Without blending, you'd see visible "seams" between sections — sudden jumps in motion or color. The linear blend ensures a smooth transition.

```
Section 1: [AAAAAAAAAAAAA]
Section 2:          [BBBBBBBBBBBBB]
Overlap:            [A..blend..B]

Result:    [AAAAAAA][A..blend..B][BBBBBBBB]
```

### Saving as MP4

```python
# From diffusers_helper/utils.py, line 266

def save_bcthw_as_mp4(x, output_filename, fps=10, crf=0):
    # x: video tensor [Batch, Channels, Time, Height, Width], values in [-1, 1]

    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    # Convert from [-1, 1] to [0, 255] range for video encoding
    # clamp ensures no values go outside the valid range

    x = x.detach().cpu().to(torch.uint8)
    # Convert to 8-bit unsigned integers (standard video pixel format)

    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    # Rearrange from [Batch, Channels, Time, Height, Width]
    #             to [Time, Height, Width, Channels]
    # If batch > 1, tiles them in a grid (for training visualization)

    torchvision.io.write_video(output_filename, x, fps=fps,
                               video_codec='libx264', options={'crf': str(int(crf))})
    # Write using H.264 codec
    # crf = Constant Rate Factor: 0 = lossless, 23 = default, 51 = worst quality
    # Lower crf = bigger file, better quality
```

---

## 12. TeaCache: Skipping Unnecessary Work

**File:** `diffusers_helper/models/hunyuan_video_packed.py` — method `initialize_teacache()` and the `forward()` method

### The Insight

During the 25 denoising steps, the model's behavior doesn't change dramatically between adjacent steps. If step 15 and step 16 have very similar inputs, they'll produce very similar outputs. So why run the full 60-block transformer twice?

TeaCache detects when the input hasn't changed much and **reuses the previous output**.

### How It Works

```python
# From the forward() method, line 950

if self.enable_teacache:
    # Compute the modulated input (the input AFTER adaptive normalization)
    modulated_inp = self.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]

    if self.cnt == 0 or self.cnt == self.num_steps - 1:
        # Always compute the first and last steps (they're most important)
        should_calc = True
    else:
        # Measure how much the input has changed since last step
        curr_rel_l1 = (
            (modulated_inp - self.previous_modulated_input).abs().mean()
            / self.previous_modulated_input.abs().mean()
        ).item()
        # This is the "relative L1 distance" -- like asking
        # "what percentage of the input has changed?"

        # Apply a polynomial rescaling (learned calibration curve)
        self.accumulated_rel_l1_distance += self.teacache_rescale_func(curr_rel_l1)

        # Only recompute if enough change has accumulated
        should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh

    if not should_calc:
        # SKIP all 60 transformer blocks!
        # Just add the previously computed residual
        hidden_states = hidden_states + self.previous_residual
        # This is like saying: "the output changed by the same amount as last time"
    else:
        # Run all 60 blocks normally
        ori_hidden_states = hidden_states.clone()
        # ... run all blocks ...
        self.previous_residual = hidden_states - ori_hidden_states
        # Save the delta for potential reuse
```

### The Rescaling Polynomial

```python
self.teacache_rescale_func = np.poly1d([7.33e+02, -4.01e+02, 6.76e+01, -3.15, 9.61e-02])
```

This is a 4th-degree polynomial that converts the raw L1 distance to a calibrated "importance score." It was determined empirically — the relationship between "how much the input changed" and "how much the output changes" is non-linear. This polynomial captures that relationship.

```
Small input change (0.01) --> very small importance score  --> likely skip
Medium input change (0.05) --> moderate importance score    --> might skip
Large input change (0.10) --> high importance score        --> must compute
```

### Speedup vs Quality Tradeoff

```
rel_l1_thresh = 0.10  -->  ~1.6x speedup, barely noticeable quality loss
rel_l1_thresh = 0.15  -->  ~2.1x speedup, slight quality degradation (hands/fingers)
```

TeaCache skips roughly 40-50% of the denoising steps. Each skipped step saves one full forward pass through 13 billion parameters — significant!

---

## 13. Memory Management: Running 13B Models on 6GB GPUs

**File:** `diffusers_helper/memory.py`

### The Challenge

The model has 13 billion parameters. In bfloat16 (2 bytes each), that's **26 GB** just for the model weights. Plus the VAE (~400MB), text encoders (~7GB), intermediate activations, etc. Total: 40+ GB.

A laptop GPU might have 6GB. How?

### Strategy: Only One Model on GPU at a Time

```python
# During text encoding:
load_model_as_complete(text_encoder_2, target_device=gpu)  # Load CLIP to GPU
llama_vec, clip_l_pooler = encode_prompt_conds(...)        # Use it
unload_complete_models()                                    # Move CLIP to CPU

# During image encoding:
load_model_as_complete(image_encoder, target_device=gpu)   # Load SigLIP to GPU
image_encoder_output = hf_clip_vision_encode(...)          # Use it
unload_complete_models()                                    # Move to CPU

# During diffusion:
move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=6)
# Move transformer to GPU, but STOP if free memory drops below 6GB

# During decoding:
offload_model_from_device_for_memory_preservation(transformer, ...)  # Move transformer to CPU
load_model_as_complete(vae, target_device=gpu)                       # Load VAE to GPU
```

### DynamicSwapInstaller — The Magic Trick

The most clever part. For the Transformer (26GB), we can't even fit it entirely on a 6GB GPU. The `DynamicSwapInstaller` solves this by moving individual **layers** on-demand:

```python
class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module, **kwargs):
        # Replace the module's __getattr__ so that when the forward pass
        # tries to access a parameter (e.g., module.weight), it automatically
        # moves that parameter to GPU first.

        def hacked_get_attr(self, name):
            if name in self._parameters:
                p = self._parameters[name]
                return p.to(**kwargs)  # Move to GPU on access!
            if name in self._buffers:
                return self._buffers[name].to(**kwargs)
            return super().__getattr__(name)

        # Monkey-patch the module's class
        module.__class__ = type('DynamicSwap_' + original_class.__name__,
                                (original_class,), {'__getattr__': hacked_get_attr})
```

**How it works in practice:**

```
Forward pass enters Block 0:
  Block 0 accesses self.weight  --> __getattr__ intercepts, moves weight to GPU
  Block 0 computes              --> result on GPU
  Block 0 finishes              --> weight stays on GPU (cached)

Forward pass enters Block 1:
  Block 1 accesses self.weight  --> moves to GPU (PyTorch may evict Block 0's weight)
  ...and so on

After forward pass:
  Most weights are back on CPU, only the most recently used ones are on GPU
```

This is like "streaming" the model through the GPU — only the currently-active layer needs to fit in memory. It's the same idea as HuggingFace's `enable_sequential_offload()` but implemented more efficiently (3x faster according to the comments).

### Preserved Memory

```python
def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            return  # STOP! We've hit the memory limit
        if hasattr(m, 'weight'):
            m.to(device=target_device)  # Move this submodule to GPU
```

This moves layers to GPU one by one, checking free memory after each move. When free memory drops below the preservation threshold (default 6GB), it stops. The remaining layers stay on CPU and are streamed on-demand via `DynamicSwapInstaller`.

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **Latent** | A compressed representation of an image/video, smaller than the original pixels |
| **VAE** | Variational Autoencoder — compresses images to latents and decompresses back |
| **Encoder** | Neural network that converts input (text/image) into a vector representation |
| **Decoder** | Neural network that converts a representation back to the original domain |
| **Token** | A basic unit of input for a Transformer — a word piece (text) or a patch (image/video) |
| **Embedding** | The vector representation of a token (e.g., a 3072-dimensional vector) |
| **Attention** | The mechanism that lets tokens communicate — each token decides which other tokens to "look at" |
| **Multi-Head Attention** | Running multiple independent attention operations in parallel |
| **RoPE** | Rotary Position Embedding — encodes position information by rotating vectors |
| **DiT** | Diffusion Transformer — a Transformer architecture designed for diffusion models |
| **CFG** | Classifier-Free Guidance — amplifies the effect of the text prompt |
| **Sigma / Noise Level** | How much noise is in the current sample (1.0 = pure noise, 0.0 = clean) |
| **Denoising Step** | One iteration of removing noise from the sample |
| **Sampler** | Algorithm that decides how to denoise step by step (UniPC in this case) |
| **UniPC** | Unified Predictor-Corrector — an efficient ODE solver for diffusion |
| **Flow Matching** | A training/inference framework where diffusion is modeled as a continuous flow |
| **AdaLN** | Adaptive Layer Normalization — adjusts normalization based on a condition (timestep) |
| **TeaCache** | Caching mechanism that skips transformer computation when input hasn't changed much |
| **Frame Packing** | Compressing historical frames at multiple scales to keep token count constant |
| **Patch** | A small region of an image/video that becomes one token |
| **Hidden State** | The internal representation at any layer of a neural network |
| **Pooler Output** | A single summary vector for an entire sequence |
| **bfloat16** | A 16-bit floating point format optimized for neural networks |
| **VRAM** | Video RAM — the GPU's memory |
| **ODE** | Ordinary Differential Equation — the math framework behind diffusion sampling |

---

## Appendix A: The Two Demo Files Compared

| Feature | `demo_gradio.py` (Inverted) | `demo_gradio_f1.py` (Forward/F1) |
|---------|---------------------------|----------------------------------|
| **Model** | `FramePackI2V_HY` | `FramePack_F1_I2V_HY_20250503` |
| **Generation order** | End of video first, then works backward | Start of video first, extends forward |
| **Section loop** | `reversed(range(total_sections))` | `range(total_sections)` |
| **History accumulation** | Prepend new to history | Append new to history |
| **Context packing** | `[start, padding, noisy, end_1x, 2x, 4x]` | `[start, 4x, 2x, 1x, noisy]` |

**Inverted sampling** (demo_gradio.py) generates the ending first and works backward. This prevents "drift" — the tendency for long videos to gradually lose coherence. By fixing the ending first, the model can plan backward from a known endpoint.

**Forward sampling** (demo_gradio_f1.py / F1 variant) generates naturally from start to end. Simpler and more intuitive, but may accumulate errors over very long videos.

---

## Appendix B: Data Flow Shapes

Here's a concrete example tracing tensor shapes through the pipeline for a 640x480 input image generating a 5-second (150-frame) video:

```
INPUT IMAGE: numpy array [480, 640, 3] (Height, Width, RGB), values 0-255

After bucket resize: [480, 640, 3] --> nearest bucket: (480, 640)
After normalize:     torch tensor [1, 3, 1, 480, 640], values [-1, 1]

TEXT ENCODING:
  LLaMA input tokens:     [1, 275] (token IDs, padded to max_length)
  LLaMA hidden states:    [1, 256, 4096] (extracted from layer -3)
  After pad to 512:       [1, 512, 4096]
  CLIP-L pooler:          [1, 768]

IMAGE ENCODING (CLIP Vision):
  SigLIP input:           [1, 3, 384, 384] (resized and normalized)
  SigLIP output:          [1, 729, 1152] (27x27 patches, 1152-dim each)

VAE ENCODE:
  Input:                  [1, 3, 1, 480, 640]
  Output (start_latent):  [1, 16, 1, 60, 80]

TRANSFORMER (one denoising step):
  Noisy latent input:     [1, 16, 9, 60, 80]  (latent_window_size=9 frames)
  After patch embed:      [1, 3072, 9, 30, 40] --> flatten --> [1, 10800, 3072]
  Clean 1x tokens:        [1, ~2400, 3072]
  Clean 2x tokens:        [1, ~600, 3072]
  Clean 4x tokens:        [1, ~170, 3072]
  Total sequence:         [1, ~14000, 3072]  (CONSTANT regardless of video length!)
  Text tokens:            [1, ~1241, 3072]  (729 image + 512 text)
  Output:                 [1, 10800, 3072] --> reshape --> [1, 16, 9, 60, 80]

VAE DECODE:
  Input:                  [1, 16, T_total, 60, 80]
  Output:                 [1, 3, T_total*4-3, 480, 640]

FINAL VIDEO: MP4 file, 30 FPS, H.264 codec
```

---

*This guide was generated from the FramePack source code. For the original paper, see:
"Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models" — Zhang et al., NeurIPS 2025.*
