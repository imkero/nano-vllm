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
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    image = Image.open("input.jpg")
    img_np = np.array(image)
    images = np.stack([img_np, img_np])
    processed = fast_qwen2_vl_preprocess(images, processor.image_processor)
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]

    model = llm.get_driver_model()
    with torch.inference_mode():
        image_embeds = model.visual(pixel_values.cuda())

    conversation = [{"role": "user", "content": "<image> 请描述图片内容"}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(text)

    padded_token_ids = []
    img_idx = 0
    for token in token_ids:
        if token == IMAGE_TOKEN_ID:
            pad_len = int(np.prod(image_grid_thw[img_idx].tolist()) // 4)
            padded_token_ids.extend([IMAGE_TOKEN_ID] * pad_len)
            img_idx += 1
        else:
            padded_token_ids.append(token)

    embeds = model.get_input_embeddings(torch.tensor(padded_token_ids).cuda())
    mask = torch.tensor(padded_token_ids) == IMAGE_TOKEN_ID
    embeds[mask] = image_embeds

    position_ids, _ = mrope_get_input_positions_and_delta(
        padded_token_ids, model.config, image_grid_thw=image_grid_thw, video_grid_thw=None, second_per_grid_ts=None
    )

    request = dict(prompt_token_ids=padded_token_ids, prompt_embeds=embeds, position_ids=position_ids)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    outputs = llm.generate([request], sampling_params)

    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
