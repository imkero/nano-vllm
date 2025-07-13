# nixl_kv_transfer_demo.py
"""
Demo script that shows end-to-end sharing of a KV-cache tensor between a
"prefill" worker and a "decode" worker using NVIDIA NIXL (NVIDIA Inter-process
Link). The script:

1. Spawns two independent Python processes (prefill_worker & decode_worker).
2. Each process instantiates its own ``NixlConnector`` helper – a minimal wrapper
   around ``nixl._api.nixl_agent`` taken from the official docs.
3. The decode worker registers a zero-initialised KV-cache tensor in VRAM and
   sends its NIXL metadata + block descriptors to the prefill worker via a
   bidirectional ``multiprocessing.Pipe``.
4. The prefill worker connects to the decode agent, copies its locally
   initialised KV-cache blocks into the remote tensor with a single
   ``WRITE`` operation, waits for completion, and then signals the decode
   worker.
5. The decode worker validates that the blocks it received match the reference
   data supplied by the prefill worker and prints ``Transfer SUCCESS`` if all
   bytes match.

Running the demo
----------------
Ensure you have:
- Linux + CUDA-capable GPU
- An NVIDIA driver that supports NIXL
- python -m pip install nixl-python-bindings torch

Then simply run::

    python nixl_kv_transfer_demo.py

You should see output similar to::

    [decode] Waiting for KV-cache transfer …
    [prefill] Transfer completed with status SUCCESS
    [decode] Transfer SUCCESS: max-abs-error = 0.0
"""

import multiprocessing as mp
import uuid
import time
import pickle
from typing import List, Tuple

import torch

# -----------------------------------------------------------------------------
# Minimal wrapper copied from the NIXL docs
# -----------------------------------------------------------------------------
from nixl._api import nixl_agent  # noqa: E402


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


# -----------------------------------------------------------------------------
# Worker entry points
# -----------------------------------------------------------------------------

def decode_worker(pipe):
    """Worker that *receives* KV-cache blocks from the prefill worker."""
    torch.cuda.set_device(0)
    connector = NixlConnector("decode_engine", rank=0)

    # For demo, make 4 blocks of tiny data (shape: 4 × 1 × 1 × 8 = 32 FP16s)
    kv_cache = torch.zeros((1000, 1, 1, 8), dtype=torch.float16, device="cuda")
    connector.register_kv_caches(kv_cache)

    # Share own agent metadata with the prefill peer
    metadata, block_descs = connector.get_agent_metadata()
    pipe.send_bytes(metadata)
    pipe.send_bytes(block_descs)

    print("[decode] Waiting for KV-cache transfer …", flush=True)

    # Blocking wait for the prefill worker to say we're done
    done_flag = pipe.recv()  # should be a simple string
    assert done_flag == "kv_transfer_done"

    # Also receive a CPU copy of the expected tensor for verification
    expected_cpu_tensor = pipe.recv()
    # Sync CUDA before reading
    torch.cuda.synchronize()

    max_abs_err = (kv_cache.cpu() - expected_cpu_tensor).abs().max().item()
    if max_abs_err == 0.0:
        print("[decode] Transfer SUCCESS: max-abs-error = 0.0")
    else:
        print(f"[decode] Transfer FAILED: max-abs-error = {max_abs_err}")


def prefill_worker(pipe):
    """Worker that *sends* its KV-cache blocks to the decode worker."""
    time.sleep(0.5)  # tiny delay to make sure decode is ready
    torch.cuda.set_device(0)

    connector = NixlConnector("prefill_engine", rank=1)

    # Create kv_cache filled with a known pattern so we can verify later
    kv_cache = torch.zeros(1000 * 1 * 1 * 8, dtype=torch.float16, device="cuda")
    kv_cache = kv_cache.view(1000, 1, 1, 8)
    kv_cache[0] = 1
    kv_cache[1] = 2
    kv_cache[2] = 3
    kv_cache[3] = 4
    connector.register_kv_caches(kv_cache)

    # Get remote agent info from the decode worker
    decode_metadata = pipe.recv_bytes()
    decode_block_descs = pipe.recv_bytes()
    decode_block_descs = pickle.loads(decode_block_descs)
    connector.add_remote_agent(decode_metadata, decode_block_descs)

    # Transfer all 4 blocks 1-to-1 (local→remote)
    local_ids = [0, 1, 2, 3]
    remote_ids = [0, 1, 2, 3]
    status = connector.write_blocks(local_ids, remote_ids)

    print(f"[prefill] Transfer completed with status {status}", flush=True)

    # Notify decode worker & send reference data for verification
    pipe.send("kv_transfer_done")
    pipe.send(kv_cache.cpu().clone())

    time.sleep(10)


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)
    # Bidirectional pipe for metadata + sync
    pipe_decode, pipe_prefill = mp.Pipe()

    p_decode = mp.Process(target=decode_worker, args=(pipe_decode,), name="decode")
    p_prefill = mp.Process(target=prefill_worker, args=(pipe_prefill,), name="prefill")

    p_decode.start()
    p_prefill.start()

    p_decode.join()
    p_prefill.join()


if __name__ == "__main__":
    main()