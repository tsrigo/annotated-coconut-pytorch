from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, Tensor
from torch.nn import Module, ModuleList

from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# swiglu feedforward, Shazeer et al.

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        rotary_pos_emb: RotaryEmbedding | None = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_pos_emb = rotary_pos_emb

        self.norm = nn.RMSNorm(dim)

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cached_kv: Tensor | None = None,
        return_cached_kv = False
    ):

        x = self.norm(x)

        qkv = self.to_qkv(x)

        q, k, v = self.split_heads(qkv)

        if exists(cached_kv):
            ck, cv = cached_kv

            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        if exists(self.rotary_pos_emb):
            q, k = self.rotary_pos_emb.rotate_queries_with_cached_keys(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        out = self.merge_heads(out)

        out = self.to_out(out)

        if not return_cached_kv:
            return out

        return out, stack((k, v))

# main class

class Coconut(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, rotary_pos_emb = self.rotary_emb),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.layers = layers

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        cached_kv: Tensor | None = None,
        return_cached_kv = False
    ):
        x = self.token_emb(x)

        cached_kv = default(cached_kv, [])
        cached_kv_iter = iter(cached_kv)

        next_key_values = []

        for attn, ff in self.layers:

            attn_out, key_values = attn(
                x,
                cached_kv = next(cached_kv_iter, None),
                return_cached_kv = True
            )

            x = attn_out + x

            next_key_values.append(key_values)

            x = ff(x) + x

        embed = self.norm(x)

        logits = self.to_logits(embed)

        if not return_cached_kv:
            return logits

        return logits, stack(next_key_values)

# test

if __name__ == '__main__':

    model = Coconut(
        num_tokens = 256,
        dim = 512,
        depth = 2
    )

    ids = torch.randint(0, 256, (1, 1024))

    logits, cached_kv = model(ids, return_cached_kv = True)

    logits2 = model(ids[:, -1:], cached_kv = cached_kv)

    print(logits.shape, logits2.shape)
