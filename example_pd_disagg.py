"""Demo showing how to transfer KV cache blocks between LLM instances.

This example demonstrates a simple scenario where a prefill worker builds the
KV cache for multiple sequences and then transfers the unique cache blocks to a
decode worker.  Blocks shared by different sequences are only copied once.
"""

import os
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus

def prefill_worker(prompts: list[dict], llm: LLM):
    """Run prefill on the source worker and return shared KV cache."""
    seqs = []
    next_token_id_map: dict[int, int] = {}

    sampling_params = SamplingParams(max_tokens=1)
    for prompt in prompts:
        seq = llm.add_request(prompt, sampling_params)
        seqs.append(seq)

    while len(llm.scheduler.waiting) > 0:
        step_seqs, _ = llm.scheduler.schedule()
        step_next_token_ids = llm.model_runner.call("run", step_seqs, True)  # only run prefill
        for seq, next_token_id in zip(step_seqs, step_next_token_ids):
            next_token_id_map[seq.seq_id] = next_token_id
            llm.scheduler.running.remove(seq)

    seq_infos: list[dict] = []
    s2i_block_id_map: dict[int, int] = {}
    src_block_ids = []
    block_hashes = []

    for seq in seqs:
        intermediate_block_table = []
        next_token_id = next_token_id_map[seq.seq_id]
        seq_info = {
            "token_ids": seq.token_ids,
            "position_ids": seq.position_ids.tolist() if seq.position_ids is not None else None,
            "token_hashes": seq.token_hashes,
            "intermediate_block_table": intermediate_block_table,
            "next_token_id": next_token_id,
            "num_cached_tokens": seq.num_cached_tokens,
        }
        seq_infos.append(seq_info)
        
        for src_block_id in seq.block_table:
            if src_block_id not in s2i_block_id_map:
                src_block = llm.scheduler.block_manager.blocks[src_block_id]
                block_hashes.append(src_block.hash)

                intermediate_block_id = len(src_block_ids)
                src_block_ids.append(src_block_id)
                s2i_block_id_map[src_block_id] = intermediate_block_id

            intermediate_block_table.append(s2i_block_id_map[src_block_id])

    src_block_ids_tensor = torch.tensor(
        src_block_ids,
        dtype=torch.long,
    )

    intermediate_kv_cache = llm.model_runner.kv_cache[:, :, src_block_ids_tensor].clone()

    for seq in seqs:
        llm.scheduler.block_manager.deallocate(seq)

    return seq_infos, intermediate_kv_cache, block_hashes


def decode_worker(
    seq_infos: list[dict],
    intermediate_kv_cache: torch.Tensor,
    block_hashes: list[int],
    llm: LLM,
):
    """Receive KV cache blocks from the prefill worker and continue decoding."""

    seqs: list[Sequence] = []
    dst_block_ids = [None] * len(block_hashes)

    for seq_info in seq_infos:
        sampling_params = SamplingParams(max_tokens=16)

        seq = Sequence(token_ids=seq_info["token_ids"],
                       sampling_params=sampling_params,
                       input_embeds=None,
                       position_ids=seq_info["position_ids"],
                       token_hashes=seq_info["token_hashes"],
                       )

        llm.scheduler.block_manager.allocate(seq)
        
        for idx, intermediate_block_id in enumerate(seq_info["intermediate_block_table"]):
            dst_block_ids[intermediate_block_id] = seq.block_table[idx]

        seq.append_token(seq_info["next_token_id"])
        seq.status = SequenceStatus.RUNNING
        llm.scheduler.running.append(seq)
        seqs.append(seq)

    
    dst_block_ids_tensor = torch.tensor(
        dst_block_ids,
        dtype=torch.long,
    )

    llm.model_runner.kv_cache[:, :, dst_block_ids_tensor] = intermediate_kv_cache

    outputs: dict[int, list[int]] = {}
    while any(not s.is_finished for s in seqs):
        out, _ = llm.step()
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids

    return [llm.tokenizer.decode(outputs[s.seq_id]) for s in seqs]


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    llm_prefill = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    llm_decode = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    tokenizer = llm_prefill.tokenizer
    prompts = [
        dict(prompt_token_ids=tokenizer.encode("Hello, how are you?")),
        dict(prompt_token_ids=tokenizer.encode("Hello Hello Hello Hello Hello Hello Hello Hello. Hello, how is the weather today?")),
    ]

    seq_infos, dense_kv_cache, hash_to_idx = prefill_worker(
        prompts, llm_prefill)
    outputs = decode_worker(seq_infos, dense_kv_cache, hash_to_idx, llm_decode)
    for out in outputs:
        print("Generated:", out)


if __name__ == "__main__":
    main()