# Pre-defined (height, width) buckets at ~640p total resolution.
# All pairs have approximately equal pixel count (~400K pixels) but different aspect ratios.
# The model was trained on these specific sizes, so inputs must be resized to one of them.
# Using non-standard sizes would produce artifacts because the patch embeddings and
# positional encodings were calibrated for these dimensions.
bucket_options = {
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
}


# Finds the bucket whose aspect ratio best matches the input image.
# Uses cross-multiplication to compare ratios without division:
#   h/w ≈ bucket_h/bucket_w  <==>  h*bucket_w ≈ w*bucket_h
# The metric |h*bucket_w - w*bucket_h| is zero for a perfect aspect ratio match.
# This avoids floating-point division and works purely with integers.
def find_nearest_bucket(h, w, resolution=640):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

