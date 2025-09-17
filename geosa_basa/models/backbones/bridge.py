from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor

@MODELS.register_module()
class Bridge(nn.Module):
    def __init__(
        self,
        num_layers: int,
        prompt_dims:int,
        embed_dims: int,
        patch_size: int,
        token_length: int = 100,
        num_heads: int = 16,
    )  -> None:
        super().__init__()
        self.num_layers = num_layers
        self.prompt_dims = prompt_dims
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_size = self.embed_dims // num_heads
        self.q_proj = nn.Linear(self.embed_dims,self.embed_dims)
        self.k_proj = nn.Linear(self.embed_dims,self.embed_dims)
        self.v_proj = nn.Linear(self.embed_dims,self.embed_dims)
        self.in_proj = torch.nn.Linear(self.prompt_dims,self.embed_dims)
        self.out_proj = torch.nn.Linear(self.embed_dims,self.prompt_dims)
        self.dropout = nn.Dropout(0.5)
        self.create_model()

    def create_model(self):
        self.prompts = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.prompt_dims])
        )

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.prompt_dims
            )
        )

        nn.init.uniform_(self.prompts.data, -val, val)

    def get_prompts(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.prompts
        else:
            return self.prompts[layer]

    def forward(self,x,layer):
        B, N, C = x.shape
        k = self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        prompts = self.get_prompts(layer).unsqueeze(0).expand(B, -1, -1)
        prompts_in = self.in_proj(prompts)
        q = self.q_proj(prompts_in).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # B x N_heads x N_tokens x N_tokens
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        context_layer = attention_scores @ v
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(B, -1, C)
        context_layer = self.out_proj(context_layer)
        context_layer = self.dropout(context_layer)
        context_layer = context_layer + prompts
        return context_layer

@MODELS.register_module()
class LoRABridge(Bridge):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.prompts
        self.prompts_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.prompts_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.prompt_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.prompt_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.prompts_a.data, -val, val)
        nn.init.uniform_(self.prompts_b.data, -val, val)

    def get_prompts(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.prompts_a @ self.prompts_b
        else:
            return self.prompts_a[layer] @ self.prompts_b[layer]











