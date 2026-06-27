import os
import torch
import einops
import random
import numpy as np
import datetime
import torchvision

from PIL import Image


# Resizes an image to fill the target dimensions (maintaining aspect ratio) then
# center-crops to exact target size. This ensures the model always receives the
# correct resolution from the bucket list without distorting the image.
# The scale_factor uses max() so the image fully covers the target area (no black bars).
def resize_and_center_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


# Stitches two video tensors (BCTHW format) together with smooth blending in the
# overlap region. Uses linear interpolation: at the start of the overlap, 100% history
# and 0% current; at the end, 0% history and 100% current. This prevents visible
# "seam" artifacts where two independently-generated video sections meet.
# The weights tensor [1.0, 0.97, ..., 0.03, 0.0] creates this gradual transition.
def soft_append_bcthw(history, current, overlap=0):
    if overlap <= 0:
        return torch.cat([history, current], dim=2)

    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"

    weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)


# Converts a BCTHW float tensor (values in [-1, 1]) to an MP4 video file.
# Steps: clamp -> rescale to [0, 255] -> cast to uint8 -> rearrange for video writer.
# CRF (Constant Rate Factor) controls H.264 compression: 0=lossless, 23=default, 51=worst.
def save_bcthw_as_mp4(x, output_filename, fps=10, crf=0):
    b, c, t, h, w = x.shape

    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break

    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    torchvision.io.write_video(output_filename, x, fps=fps, video_codec='libx264', options={'crf': str(int(crf))})
    return x


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int):
    if tensor is None:
        return None

    first_dim = tensor.shape[0]

    if first_dim == batch_size:
        return tensor

    if batch_size % first_dim != 0:
        raise ValueError(f"Cannot evenly repeat first dim {first_dim} to match batch_size {batch_size}.")

    repeat_times = batch_size // first_dim

    return tensor.repeat(repeat_times, *[1] * (tensor.dim() - 1))


# Pads or crops a 3D tensor (Batch, SeqLen, Channels) to exactly `length` tokens.
# Returns a boolean mask indicating which positions are real (True) vs padding (False).
# Used to standardize text encoder output to a fixed length (512) -- shorter prompts
# get zero-padded, and the mask tells the Transformer to ignore the padded positions.
def crop_or_pad_yield_mask(x, length):
    B, F, C = x.shape
    device = x.device
    dtype = x.dtype

    if F < length:
        y = torch.zeros((B, length, C), dtype=dtype, device=device)
        mask = torch.zeros((B, length), dtype=torch.bool, device=device)
        y[:, :F, :] = x
        mask[:, :F] = True
        return y, mask

    return x[:, :length, :], torch.ones((B, length), dtype=torch.bool, device=device)




def generate_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    milliseconds = f"{int(now.microsecond / 1000):03d}"
    random_number = random.randint(0, 9999)
    return f"{timestamp}_{milliseconds}_{random_number}"
