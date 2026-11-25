"""LoRA on the Qwen3-Embedding-8B text tower (rank 16 in the paper); unchanged from the training code."""
import math

import torch
from torch import nn

LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


class LoRALinear(nn.Module):
    """Frozen base Linear + low-rank update."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base.requires_grad_(False), alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.T.to(x.dtype)) @ self.lora_b.T.to(x.dtype) * self.scale


def add_lora(backbone, rank, alpha):
    backbone.requires_grad_(False)
    for module in list(backbone.modules()):
        for name in LORA_TARGETS:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)
