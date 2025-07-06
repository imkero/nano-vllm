import pickle
from functools import cached_property
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from transformers import Qwen2Config, Qwen3Config
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if isinstance(hf_config, Qwen3Config):
            self.model = Qwen3ForCausalLM(hf_config)
        elif isinstance(hf_config, Qwen2Config):
            self.model = Qwen2ForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported architecture: {hf_config.__class__.__name__}")
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    @cached_property
    def uses_mrope(self):
        return getattr(self.model, "uses_mrope", False)

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        dummy_ids = [0] * max_model_len
        if self.uses_mrope:
            positions = torch.arange(max_model_len, device="cpu", dtype=torch.int64).unsqueeze(0).expand(3, -1)
        else:
            positions = torch.arange(max_model_len, device="cpu", dtype=torch.int64)
        seqs = [Sequence(dummy_ids, position_ids=positions) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = hf_config.get("head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        # if all seqs dont have input_embeds, we can use get_input_embeddings directly
        if all(seq.input_embeds is None for seq in seqs):
            input_ids = []
            for seq in seqs:
                seq_token_ids = seq.token_ids[seq.num_cached_tokens:] if seq.num_cached_tokens > 0 else seq.token_ids
                input_ids.extend(seq_token_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda(non_blocking=True)
            input_embeds = self.get_input_embeddings(input_ids)
        else:
            input_embeds = []
            for seq in seqs:
                if seq.input_embeds is None:
                    seq_token_ids = seq.token_ids[seq.num_cached_tokens:] if seq.num_cached_tokens > 0 else seq.token_ids
                    seq_token_ids = seq_token_ids.cuda(non_blocking=True)
                    seq_embeds = self.get_input_embeddings(seq_token_ids)
                else:
                    seq_embeds = seq.input_embeds[seq.num_cached_tokens:]
                input_embeds.append(seq_embeds)
            input_embeds = torch.cat(input_embeds).cuda(non_blocking=True)

        positions = []
        for seq in seqs:
            if self.uses_mrope:
                if seq.position_ids is not None:
                    positions.append(seq.position_ids[:, seq.num_cached_tokens:])
                else:
                    positions.append(torch.arange(seq.num_cached_tokens, len(seq), dtype=torch.int64, device="cpu").unsqueeze(0).expand(3, -1))
            else:
                if seq.position_ids is not None:
                    positions.append(seq.position_ids[seq.num_cached_tokens:])
                else:
                    positions.append(torch.arange(seq.num_cached_tokens, len(seq), dtype=torch.int64, device="cpu"))
        positions = torch.cat(positions).cuda(non_blocking=True)

        for seq in seqs:
            seqlen = len(seq)
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_embeds, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            if seq.num_completion_tokens == 0:
                raise Exception("Should not do decode on sequence having num_completion_tokens == 0.")
            input_ids.append(seq.last_token)
            positions.append(self.model.get_next_position_id(seq.position_ids, len(seq)))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda(non_blocking=True)
        input_embeds = self.model.get_input_embeddings(input_ids)
        if self.uses_mrope:
            positions = torch.stack(positions).cuda(non_blocking=True)
        else:
            positions = torch.tensor(positions, dtype=torch.int64).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_embeds, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_embeds: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_embeds.size(0) > 512:
            return self.model.compute_logits(self.model(input_embeds, positions))
        else:
            bs = input_embeds.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_embeds"][:bs] = input_embeds
            if self.uses_mrope:
                graph_vars["positions"][:, :bs] = positions
            else:
                graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool):
        input_embeds, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_embeds, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_embeds = torch.zeros(max_bs, hf_config.hidden_size)
        pos_shape = (3, max_bs) if self.uses_mrope else (max_bs,)
        positions = torch.zeros(pos_shape, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            pos_view = positions[:, :bs] if self.uses_mrope else positions[:bs]
            outputs[:bs] = self.model(input_embeds[:bs], pos_view)    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_embeds[:bs], pos_view)    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_embeds=input_embeds,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    def get_input_embeddings(self, input_ids):
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        input_ids = input_ids.cuda()
        embeds = self.model.get_input_embeddings(input_ids)
        return embeds
