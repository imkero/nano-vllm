# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union

import numba
import numpy as np
import torch
from transformers import PretrainedConfig


def mrope_get_input_positions_and_delta(
    input_tokens: Union[list[int], np.ndarray],
    hf_config: PretrainedConfig,
    image_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
    video_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
    second_per_grid_ts: Optional[list[float]],
    context_len: int = 0,
    seq_len: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    input_tokens = np.asarray(input_tokens, dtype=np.int64)

    if image_grid_thw is None or len(image_grid_thw) == 0:
        image_grid_thw = np.empty((0, 3), dtype=np.int64)
    elif isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.numpy()
    else:
        image_grid_thw = np.array(image_grid_thw, dtype=np.int64)

    if video_grid_thw is None or len(video_grid_thw) == 0:
        video_grid_thw = np.empty((0, 3), dtype=np.int64)
    elif isinstance(video_grid_thw, torch.Tensor):
        video_grid_thw = video_grid_thw.numpy()
    else:
        video_grid_thw = np.array(video_grid_thw, dtype=np.int64)

    if second_per_grid_ts is None:
        second_per_grid_ts_np = np.empty((0, ), dtype=np.float64)
    else:
        second_per_grid_ts_np = np.array(second_per_grid_ts,
                                            dtype=np.float64)

    (
        input_positions,
        mrope_position_delta,
    ) = _vl_get_input_positions_numba(
        input_tokens=input_tokens,
        image_token_id=int(hf_config.image_token_id),
        video_token_id=int(hf_config.video_token_id),
        spatial_merge_size=int(
            hf_config.vision_config.spatial_merge_size),
        tokens_per_second=float(
            getattr(hf_config.vision_config, "tokens_per_second",
                    1.0)),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts_np,
    )

    input_positions = torch.from_numpy(input_positions)
    if context_len != 0 or seq_len is not None:
        input_positions = input_positions[:, context_len:seq_len]
    
    return input_positions, mrope_position_delta


@numba.jit(nopython=True)
def _vl_get_input_positions_numba(
    input_tokens: np.ndarray,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    tokens_per_second: float,
    image_grid_thw: np.ndarray,
    video_grid_thw: np.ndarray,
    second_per_grid_ts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Get mrope input positions and delta value for Qwen2/2.5-VL

    This is the optimized numba implementation
    """

    mrope_pos = np.empty((3, len(input_tokens)), dtype=np.int64)

    cur_t = -1

    cur_image_idx = -1
    cur_video_idx = -1

    i = 0
    while i < len(input_tokens):
        token_id = input_tokens[i]
        if token_id == image_token_id:
            cur_image_idx += 1
            if cur_image_idx >= len(image_grid_thw):
                with numba.objmode():
                    _raise_missing_mm_item_error(MM_TYPE_IMAGE, cur_image_idx)

            i, cur_t = _emit_image_tokens(
                mrope_pos,
                i=i,
                image_grid_thw=image_grid_thw[cur_image_idx],
                start_t=cur_t + 1,
                spatial_merge_size=spatial_merge_size,
            )

            if i == ERR_EXCEEDED:
                with numba.objmode():
                    _raise_tokens_out_of_bound_error(MM_TYPE_IMAGE,
                                                     cur_image_idx)
        elif token_id == video_token_id:
            cur_video_idx += 1
            if cur_video_idx >= len(video_grid_thw):
                with numba.objmode():
                    _raise_missing_mm_item_error(MM_TYPE_VIDEO, cur_video_idx)

            i, cur_t = _emit_video_tokens(
                mrope_pos,
                i=i,
                video_grid_thw=video_grid_thw[cur_video_idx],
                start_t=cur_t + 1,
                spatial_merge_size=spatial_merge_size,
                tokens_per_second=tokens_per_second,
                second_per_grid_t=1.0 \
                    if cur_video_idx >= len(second_per_grid_ts) \
                    else second_per_grid_ts[cur_video_idx],
            )

            if i == ERR_EXCEEDED:
                with numba.objmode():
                    _raise_tokens_out_of_bound_error(MM_TYPE_VIDEO,
                                                     cur_video_idx)
        else:
            cur_t += 1
            i = _emit_1d_token(
                mrope_pos,
                i=i,
                t=cur_t,
            )

    num_unused_images = len(image_grid_thw) - cur_image_idx - 1
    if num_unused_images > 0:
        with numba.objmode():
            _raise_unused_mm_items_error(MM_TYPE_IMAGE, num_unused_images)

    num_unused_videos = len(video_grid_thw) - cur_video_idx - 1
    if num_unused_videos > 0:
        with numba.objmode():
            _raise_unused_mm_items_error(MM_TYPE_VIDEO, num_unused_videos)

    mrope_position_delta = cur_t + 1 - len(input_tokens)
    return mrope_pos, mrope_position_delta

# following functions are used by:
# - _vl_get_input_positions_numba

MM_TYPE_IMAGE = "image_grid_thw"
MM_TYPE_VIDEO = "video_grid_thw"
ERR_EXCEEDED = -1


@numba.jit(nopython=True, inline="always")
def _emit_2d_tokens(
    mrope_pos: np.ndarray,
    i: int,
    num_h: int,
    num_w: int,
    cur_t: int,
    start_hw: int,
) -> int:
    if i + num_h * num_w > mrope_pos.shape[1]:
        return ERR_EXCEEDED

    for h in range(num_h):
        for w in range(num_w):
            mrope_pos[0, i] = cur_t
            mrope_pos[1, i] = start_hw + h
            mrope_pos[2, i] = start_hw + w
            i += 1

    return i


@numba.jit(nopython=True)
def _emit_image_tokens(
    mrope_pos: np.ndarray,
    i: int,
    image_grid_thw: np.ndarray,
    start_t: int,
    spatial_merge_size: int,
) -> tuple[int, int]:
    num_h = image_grid_thw[1] // spatial_merge_size
    num_w = image_grid_thw[2] // spatial_merge_size
    for t in range(start_t, start_t + image_grid_thw[0]):
        i = _emit_2d_tokens(
            mrope_pos,
            i=i,
            num_h=num_h,
            num_w=num_w,
            cur_t=t,
            start_hw=start_t,
        )
        if i == ERR_EXCEEDED:
            return ERR_EXCEEDED, ERR_EXCEEDED

    cur_t = start_t + max(image_grid_thw[0], num_h, num_w) - 1
    return i, cur_t


@numba.jit(nopython=True)
def _emit_video_tokens(
    mrope_pos: np.ndarray,
    i: int,
    video_grid_thw: np.ndarray,
    start_t: int,
    spatial_merge_size: int,
    tokens_per_second: int,
    second_per_grid_t: float,
) -> tuple[int, int]:
    num_h = video_grid_thw[1] // spatial_merge_size
    num_w = video_grid_thw[2] // spatial_merge_size

    tokens_per_grid_t = tokens_per_second * second_per_grid_t

    for t in range(video_grid_thw[0]):
        i = _emit_2d_tokens(
            mrope_pos,
            i=i,
            num_h=num_h,
            num_w=num_w,
            cur_t=start_t + int(t * tokens_per_grid_t),
            start_hw=start_t,
        )
        if i == ERR_EXCEEDED:
            return ERR_EXCEEDED, ERR_EXCEEDED

    cur_t = start_t + max(
        int((video_grid_thw[0] - 1) * tokens_per_grid_t),
        num_h - 1,
        num_w - 1,
    )
    return i, cur_t


@numba.jit(nopython=True, inline="always")
def _emit_1d_token(
    mrope_pos: np.ndarray,
    i: int,
    t: int,
) -> int:
    mrope_pos[0, i] = t
    mrope_pos[1, i] = t
    mrope_pos[2, i] = t
    return i + 1


@numba.jit()
def _raise_missing_mm_item_error(mm_type: str, mm_index: int):
    raise ValueError(f"Mismatch between input_tokens and {mm_type}"
                     f" ({mm_type}[{mm_index}] is missing)."
                     " Please check your prompt and multi_modal_data.")


@numba.jit()
def _raise_tokens_out_of_bound_error(mm_type: str, mm_index: int):
    raise ValueError(
        f"Mismatch between input_tokens and {mm_type}"
        f" (input_tokens out of bounds while processing {mm_type}[{mm_index}])."
        " Please check your prompt and multi_modal_data.")


@numba.jit()
def _raise_unused_mm_items_error(mm_type: str, unused_num: int):
    raise ValueError(f"Mismatch between input_tokens and {mm_type}"
                     f" ({mm_type} has {unused_num} unused items)."
                     " Please check your prompt and multi_modal_data.")
