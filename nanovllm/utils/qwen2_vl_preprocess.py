import math
from typing import Tuple
import cv2
import numpy as np
from numba import jit, prange
import torch
from transformers import Qwen2VLImageProcessor
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 4000 * IMAGE_FACTOR * IMAGE_FACTOR


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _normalize(arr: np.ndarray, dim: int):
    arr /= 255.0
    arr -= OPENAI_CLIP_MEAN[dim]
    arr /= OPENAI_CLIP_STD[dim]


NORMALIZED_VALUES_MAP = np.empty((3, 256), dtype=np.float32)
for dim in range(NORMALIZED_VALUES_MAP.shape[0]):
    NORMALIZED_VALUES_MAP[dim, :] = np.arange(256)
    _normalize(NORMALIZED_VALUES_MAP[dim], dim)


@jit("void(uint8[:,:,:,:],float32[:,:,:,:])", nopython=True, nogil=True, parallel=True)
def _fast_rescale_normalize_transpose(vid_in: np.ndarray, vid_out: np.ndarray):
    for i in prange(vid_in.shape[0]):
        for j in range(vid_in.shape[1]):
            for k in range(vid_in.shape[2]):
                for dim in range(3):
                    vid_out[i, dim, j, k] = NORMALIZED_VALUES_MAP[dim, vid_in[i, j, k, dim]]


def _fast_resize(images: np.ndarray, image_processor: Qwen2VLImageProcessor):
    nframes, height, width, channels = images.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_processor.patch_size * image_processor.merge_size,
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    resized_images = np.empty((nframes, resized_height, resized_width, channels), dtype=np.uint8)
    for idx in range(nframes):
        cv2.resize(images[idx], (resized_width, resized_height), dst=resized_images[idx], interpolation=cv2.INTER_CUBIC)
    return resized_images


def _build_patches(processed_images: np.ndarray, image_processor: Qwen2VLImageProcessor):
    patches = torch.from_numpy(processed_images)
    channel = patches.shape[1]
    resized_height, resized_width = patches.shape[2], patches.shape[3]
    grid_t = patches.shape[0] // image_processor.temporal_patch_size
    grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
    patches = patches.view(
        grid_t,
        image_processor.temporal_patch_size,
        channel,
        grid_h // image_processor.merge_size,
        image_processor.merge_size,
        image_processor.patch_size,
        grid_w // image_processor.merge_size,
        image_processor.merge_size,
        image_processor.patch_size,
    )
    patches = patches.permute((0, 3, 6, 4, 7, 2, 1, 5, 8))
    patches = patches.contiguous()
    patches = patches.view(
        grid_t * grid_h * grid_w,
        channel * image_processor.temporal_patch_size * image_processor.patch_size * image_processor.patch_size,
    )
    return patches, [grid_t, grid_h, grid_w]


def fast_qwen2_vl_preprocess(images: np.ndarray, image_processor: Qwen2VLImageProcessor):
    processed_images = _fast_resize(images, image_processor)
    processed_images_float = np.empty(
        (processed_images.shape[0], processed_images.shape[3], processed_images.shape[1], processed_images.shape[2]),
        dtype=np.float32,
    )
    _fast_rescale_normalize_transpose(processed_images, processed_images_float)
    patches, grid_thw = _build_patches(processed_images_float, image_processor)
    return {
        "pixel_values": patches,
        "image_grid_thw": torch.tensor([grid_thw], dtype=torch.int64),
    }

