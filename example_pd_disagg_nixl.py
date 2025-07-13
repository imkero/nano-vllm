"""Demo showing how to transfer KV cache blocks between LLM instances.

This example demonstrates a simple scenario where a prefill worker builds the
KV cache for multiple sequences and then transfers the unique cache blocks to a
decode worker.  Blocks shared by different sequences are only copied once.
"""

import os
from PIL import Image
import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
import multiprocessing as mp
import uuid
import time
import pickle
from typing import List, Tuple

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.qwen2_vl_preprocess import fast_qwen2_vl_preprocess
from nanovllm.utils.mrope_positions import mrope_get_input_positions_and_delta
from nixl._api import nixl_agent  # noqa: E402


IMAGE_TOKEN_ID = 151655
model_path = os.path.expanduser("~/model/Qwen2-VL-2B-Instruct/")


class NixlConnector:
    """Helper that hides the raw nixl_agent API calls we need for the demo."""

    def __init__(self, engine_id: str, rank: int):
        # Create a unique NIXL agent for this worker.
        self.nixl_wrapper = nixl_agent(str(uuid.uuid4()), None)
        self.engine_id = engine_id
        self.rank = rank
        self.block_len: int | None = None
        self.num_blocks: int | None = None
        # Handles/descriptors populated by the registration phase
        self.local_blocks_descs = None
        self.local_xfer_side_handle = None
        self.remote_xfer_side_handle = None

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_cache: torch.Tensor):
        """Register a *contiguous* KV-cache tensor with NIXL for sharing."""
        # The tensor layout depends on your attention implementation; here we
        # assume (num_blocks, block_size, num_heads, head_dim).
        num_blocks, block_size, num_heads, head_dim = kv_cache.shape
        element_size = kv_cache.element_size()
        print("tensor shape", kv_cache.shape, flush=True)

        self.block_len = block_size * num_heads * head_dim * element_size
        self.num_blocks = num_blocks

        base_addr = kv_cache.data_ptr()
        region_len = num_blocks * self.block_len
        caches_data = [(base_addr, region_len, self.rank, "")]

        # 1️⃣ Register the contiguous region covering the whole tensor
        descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
        self.nixl_wrapper.register_memory(descs)

        # 2️⃣ Build block-level descriptors for later transfers
        blocks_data = [
            (base_addr + block_id * self.block_len, self.block_len, self.rank)
            for block_id in range(num_blocks)
        ]
        self.local_blocks_descs = self.nixl_wrapper.get_xfer_descs(
            blocks_data, "VRAM"
        )

        # 3️⃣ Cache a *local* transfer-side handle (other side is remote)
        self.local_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "", self.local_blocks_descs
        )

    # ------------------------------------------------------------------
    # Remote-agent helpers
    # ------------------------------------------------------------------

    def get_agent_metadata(self) -> Tuple[bytes, bytes]:
        """Return bytes objects you can ship to a remote peer."""
        print("descs len", len(pickle.dumps(self.local_blocks_descs)), flush=True)
        return self.nixl_wrapper.get_agent_metadata(), pickle.dumps(self.local_blocks_descs)

    def add_remote_agent(
        self, agent_metadata: bytes, remote_blocks_descs: bytes
    ) -> str:
        """Connect to a *remote* NIXL agent & prep the remote xfer handle."""
        agent_name = self.nixl_wrapper.add_remote_agent(agent_metadata)
        self.remote_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            agent_name, remote_blocks_descs
        )
        return agent_name

    # ------------------------------------------------------------------
    # Transfer helper
    # ------------------------------------------------------------------

    def write_blocks(
        self,
        local_block_ids: List[int],
        remote_block_ids: List[int],
        notify_msg: str = "kv_transfer",
    ):
        handle = self.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            self.local_xfer_side_handle,
            local_block_ids,
            self.remote_xfer_side_handle,
            remote_block_ids,
            notify_msg,
        )
        status = self.nixl_wrapper.transfer(handle)
        return status


def prefill_worker(prompts: list[dict], llm: LLM):
    """Run prefill on the source worker and return shared KV cache."""
    seqs = []
    next_token_id_map: dict[int, int] = {}

    sampling_params = SamplingParams(temperature=0, max_tokens=1)
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

    for seq in seqs:
        llm.scheduler.block_manager.deallocate(seq)

    return seq_infos, src_block_ids, block_hashes


def decode_worker(
    seq_infos: list[dict],
    block_hashes: list[int],
    llm: LLM,
):
    """Receive KV cache blocks from the prefill worker and continue decoding."""

    seqs: list[Sequence] = []
    dst_block_ids = [None] * len(block_hashes)

    for seq_info in seq_infos:
        sampling_params = SamplingParams(max_tokens=16, temperature=0)

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


    outputs: dict[int, list[int]] = {}
    while any(not s.is_finished for s in seqs):
        out, _ = llm.step()
        for seq_id, token_ids in out:
            outputs[seq_id] = token_ids

    return [llm.tokenizer.decode(outputs[s.seq_id]) for s in seqs]

def prefill_process(pipe):
    """Worker that *sends* its KV-cache blocks to the decode worker."""
    torch.cuda.set_device(0)
    connector = NixlConnector("prefill_engine", rank=1)

    print("loading model")
    llm_prefill = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.4, num_kvcache_blocks=100)
    print("model loaded")
    kv_cache = llm_prefill.model_runner.kv_cache
    connector.register_kv_caches(kv_cache)

    # Get remote agent info from the decode worker
    decode_metadata = pipe.recv_bytes()
    decode_block_descs = pipe.recv_bytes()
    decode_block_descs = pickle.loads(decode_block_descs)
    connector.add_remote_agent(decode_metadata, decode_block_descs)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(model_path)
    image_processor = processor.image_processor
    image_processor.max_pixels = 1024 * 1024

    # https://cdn.dizzylab.net/media/cover/%E7%BD%91%E6%98%93%E4%BA%91cover.jpg
    image = Image.open("cover.jpg")
    img_np = np.array(image)
    images = np.stack([img_np, img_np])
    processed = fast_qwen2_vl_preprocess(images, image_processor)
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]

    config = llm_prefill.model_runner.config.hf_config
    model = llm_prefill.model_runner.model
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

    prompt_token_ids_tensor = torch.tensor(prompt_token_ids, dtype=torch.int64)
    prompt_embeds = model.get_input_embeddings(prompt_token_ids_tensor.cuda())
    prompt_embeds[prompt_token_ids_tensor == IMAGE_TOKEN_ID] = image_embeds
    position_ids, _ = mrope_get_input_positions_and_delta(
        prompt_token_ids_tensor, config, image_grid_thw=image_grid_thw, video_grid_thw=None, second_per_grid_ts=None
    )

    prompts = [dict(prompt_token_ids=prompt_token_ids, prompt_embeds=prompt_embeds, position_ids=position_ids)]

    print("prefilling")
    seq_infos, src_block_ids, block_hashes = prefill_worker(
        prompts, llm_prefill)

    # Transfer blocks 1-to-1 (local→remote)
    remote_ids = list(range(len(src_block_ids)))
    status = connector.write_blocks(src_block_ids, remote_ids)
    print(f"[prefill] Transfer completed with status {status}", flush=True)

    # Notify decode worker & send reference data for verification
    pipe.send("kv_transfer_done")
    pipe.send(seq_infos)
    pipe.send(block_hashes)

    print("seqs", seq_infos)
    print("block_hashes", block_hashes)

    return

def decode_process(pipe):
    """Worker that *receives* KV-cache blocks from the prefill worker."""
    torch.cuda.set_device(1)
    connector = NixlConnector("decode_engine", rank=0)

    print("loading model")
    llm_decode = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.4, num_kvcache_blocks=100)
    print("model loaded")
    kv_cache = llm_decode.model_runner.kv_cache
    connector.register_kv_caches(kv_cache)

    # Share own agent metadata with the prefill peer
    metadata, block_descs = connector.get_agent_metadata()
    pipe.send_bytes(metadata)
    pipe.send_bytes(block_descs)

    print("[decode] Waiting for KV-cache transfer …", flush=True)

    # Blocking wait for the prefill worker to say we're done
    done_flag = pipe.recv()  # should be a simple string
    assert done_flag == "kv_transfer_done"

    seq_infos = pipe.recv()
    block_hashes = pipe.recv()

    print("---")
    print("decoding")
    outputs = decode_worker(seq_infos, block_hashes, llm_decode)
    for out in outputs:
        print("Generated:", out)


def main():
    mp.set_start_method('spawn', force=True)
    # Bidirectional pipe for metadata + sync
    pipe_decode, pipe_prefill = mp.Pipe()

    p_decode = mp.Process(target=decode_process, args=(pipe_decode,), name="decode")
    p_decode.start()

    prefill_process(pipe_prefill)

    p_decode.join()


if __name__ == "__main__":
    main()
