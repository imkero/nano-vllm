import torch
from torch import nn
from transformers import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.models.qwen2 import Qwen2Model


class Qwen2VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen2VLConfig
    ) -> None:
        super().__init__()
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        
        rope_scaling = getattr(config, "rope_scaling", None)
        self.uses_mrope = rope_scaling is not None and isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def get_next_position_id(self, position_ids, seq_length: int):
        if self.uses_mrope:
            if position_ids is not None:
                # mrope 计算下一文本 token 位置只需取前一 token 所有维度的最大坐标 + 1
                mrope_position_delta = position_ids[:, -1].max().item() - len(position_ids) + 1
            else:
                mrope_position_delta = 0
            next_pos = mrope_position_delta + seq_length - 1
            return torch.tensor([next_pos], device="cpu", dtype=torch.int64).expand(3, -1)
        else:
            return seq_length

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.uses_mrope:
            assert positions.dim() == 2, "M-RoPE requires positions to be a 2D tensor."
            assert positions.size(0) == 3, "M-RoPE requires positions to have 3 rows."
            assert positions.size(1) == input_embeds.size(0), \
                "M-RoPE requires positions to match the sequence length of input_embeds."
            
        hidden_states = self.model(input_embeds, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
