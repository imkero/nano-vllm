import os
from PIL import Image
import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor

from nanovllm import LLM, SamplingParams
from nanovllm.utils.qwen2_vl_preprocess import fast_qwen2_vl_preprocess
from nanovllm.utils.mrope_positions import mrope_get_input_positions_and_delta


IMAGE_TOKEN_ID = 151655


def main():
    path = os.path.expanduser("~/huggingface/Qwen2-VL-7B-Instruct/")

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    processor = AutoProcessor.from_pretrained(path)
    image_processor = processor.image_processor
    image_processor.max_pixels = 1024 * 1024

    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # https://cdn.dizzylab.net/media/cover/%E7%BD%91%E6%98%93%E4%BA%91cover.jpg
    image = Image.open("cover.jpg")
    img_np = np.array(image)
    images = np.stack([img_np, img_np])
    processed = fast_qwen2_vl_preprocess(images, image_processor)
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]

    config = llm.model_runner.config
    model = llm.model_runner.model
    with torch.inference_mode():
        image_embeds = model.visual(pixel_values.cuda(), image_grid_thw.cuda())

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "描述这幅图片"},
            ]
        },
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(text)

    prompt_token_ids = []
    img_idx = 0
    for token in token_ids:
        if token == IMAGE_TOKEN_ID:
            pad_len = image_grid_thw[img_idx].prod().item() // 4
            prompt_token_ids.extend([IMAGE_TOKEN_ID] * pad_len)
            img_idx += 1
        else:
            prompt_token_ids.append(token)

    prompt_token_ids = torch.tensor(prompt_token_ids, dtype=torch.int64)
    prompt_embeds = model.get_input_embeddings(prompt_token_ids.cuda())
    prompt_embeds[prompt_token_ids == IMAGE_TOKEN_ID] = image_embeds

    position_ids, _ = mrope_get_input_positions_and_delta(
        prompt_token_ids, config, image_grid_thw=image_grid_thw, video_grid_thw=None, second_per_grid_ts=None
    )

    request = dict(prompt_token_ids=prompt_token_ids, prompt_embeds=prompt_embeds, position_ids=position_ids)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    outputs = llm.generate([request], sampling_params)

    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
