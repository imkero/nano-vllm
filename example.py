import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    requests = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        embeds = llm.model_runner.call("get_input_embeddings", token_ids)
        pos_ids = list(range(len(token_ids)))
        hashes = list(token_ids)
        requests.append(dict(prompt_token_ids=token_ids, prompt_embeds=embeds, position_ids=pos_ids, token_hashes=hashes))
    outputs = llm.generate(requests, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
