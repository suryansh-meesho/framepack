import numpy as np


# Encodes an input image through a CLIP/SigLIP Vision Transformer.
# The feature_extractor resizes the image to the model's expected size (384x384),
# normalizes pixel values, and converts to a tensor. The vision encoder then splits
# the image into a grid of patches (e.g., 27x27 = 729 patches) and produces a
# 1152-dimensional vector for each patch, capturing what's IN that region.
# Output .last_hidden_state shape: [1, 729, 1152] -- used to condition the video
# Transformer so it understands the visual content of the input image.
def hf_clip_vision_encode(image, feature_extractor, image_encoder):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    preprocessed = feature_extractor.preprocess(images=image, return_tensors="pt").to(device=image_encoder.device, dtype=image_encoder.dtype)
    image_encoder_output = image_encoder(**preprocessed)

    return image_encoder_output
