# improvised version of coconut that allows multiple streams during the recurrent latent reasoning
# https://arxiv.org/abs/2107.10342

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce
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

        if return_cached_kv:
            cached_kv = stack((k, v))

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

        return out, cached_kv

# main class

class Transformer(Module):
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
        self.dim = dim

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
        inp: Tensor | list[Tensor],
        cached_kv: Tensor | None = None,
        return_intermediates = False,
        return_embed_with_cache_kv = False
    ):

        # handle input, which can be a list of Tensor of Float['b n d'] or Int['b n']

        if not isinstance(inp, list):
            inp = [inp]

        inp = [self.token_emb(t) if t.dtype in (torch.int, torch.long) else t for t in inp]

        x = cat(inp, dim = -2)

        # cached key values need to be handled with priority and care for this paper

        cached_kv = default(cached_kv, [])
        cached_kv_iter = iter(cached_kv)

        next_keys_values = []

        for attn, ff in self.layers:

            attn_out, key_values = attn(
                x,
                cached_kv = next(cached_kv_iter, None),
                return_cached_kv = True
            )

            x = attn_out + x

            next_keys_values.append(key_values)

            x = ff(x) + x

        embeds = self.norm(x)

        if return_embed_with_cache_kv:
            return embeds, next_keys_values

        logits = self.to_logits(embeds)

        if not return_intermediates:
            return logits

        return logits, embeds, next_keys_values

# coconut wrapper around transformer handles recurrence with latent reasoning tokens

class Coconut(Module):
    def __init__(
        self,
        num_reasoning_steps,
        transformer: dict | Transformer,
        num_latents_per_step = 1, # extending the paper, allow for more than one "reasoning" token per step
        learn_begin_of_thought = False,
        num_hypothesis = 1, # extending the paper, allow for multiple sequence latent streams, merged at the end
        synthesize_hypothesis_per_step = False
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer)

        dim = transformer.dim

        self.model = transformer
        self.num_reasoning_steps = num_reasoning_steps

        # begin and end of thought tokens, handled external to transformer

        self.begin_of_thought = nn.Parameter(torch.zeros(num_latents_per_step, dim))
        self.end_of_thought = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(self.begin_of_thought, std = 0.02)
        nn.init.normal_(self.end_of_thought, std = 0.02)

        # whether to teach model when to begin a thought

        self.learn_begin_of_thought = learn_begin_of_thought
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # project latents to multiple streams

        self.num_hypothesis = num_hypothesis
        self.has_multiple_hypothesis = num_hypothesis > 1

        if self.has_multiple_hypothesis:
            streams = num_hypothesis
            eye = torch.eye(dim)

            self.to_streams = nn.Sequential(nn.Linear(dim, dim * streams), Rearrange('b ... (hyp d) -> (b hyp) ... d', hyp = streams))
            self.merge_streams = nn.Sequential(Rearrange('(b hyp) ... d -> b ... (hyp d)', hyp = num_hypothesis), nn.Linear(dim * streams, dim))
            self.maybe_synth_streams = nn.Identity()

            if synthesize_hypothesis_per_step:
                self.maybe_synth_streams = nn.Sequential(
                    Rearrange('(b hyp) n d -> b n (hyp d)', hyp = streams),
                    nn.Linear(dim * streams, dim * streams),
                    Rearrange('b n (hyp d) -> (b hyp) n d', hyp = streams),
                )

            # init to identity

            self.to_streams[0].weight.data.copy_(repeat(eye, 'i j -> (r i) j', r = streams))
            self.to_streams[0].bias.data.zero_()

            self.merge_streams[-1].weight.data.copy_(repeat(eye, 'i j -> i (r j)', r = streams))
            self.merge_streams[-1].bias.data.zero_()

            if synthesize_hypothesis_per_step:
                self.maybe_synth_streams[1].weight.data.copy_(repeat(eye, 'i j -> (r1 i) (r2 j)', r1 = streams, r2 = streams))
                self.maybe_synth_streams[1].bias.data.zero_()

    def forward(
        self,
        prompt,
        answer,
        return_loss = True
    ):
        """
        ein notation:
        b - batch
        h - attention heads
        n - seq (number of reasoning tokens, etc)
        d - feature dimension
        l - logits (num tokens)
        """

        batch, num_thoughts, has_multi_hyp = prompt.shape[0], self.begin_of_thought.shape[0], self.has_multiple_hypothesis

        # prepare <bot> and <eot> in paper

        begin_thought = repeat(self.begin_of_thought, 'n d -> b n d', b = batch)
        end_thought = repeat(self.end_of_thought, 'd -> b 1 d', b = batch)

        # give the model the prompt

        prompt_logits, embeds, cached_kv = self.model([prompt, begin_thought], return_intermediates = True)

        # loss for decoding a <bot>

        bot_loss = self.zero

        if self.learn_begin_of_thought:
            pred_bot_embed, rest_logits = embeds[:, -2], prompt_logits[:, -2]

            pred_bot_logits = einsum('b d, d -> b', pred_bot_embed, self.begin_of_thought[0])
            pred_bot_logits = rearrange(pred_bot_logits, 'b -> b 1')

            bot_logits = cat((pred_bot_logits, rest_logits), dim = -1)
            bot_labels = bot_logits.new_zeros((batch,), dtype = torch.long)

            bot_loss = F.cross_entropy(bot_logits, bot_labels)

        # extract latent reasoning token off <bot> position

        latent_token = embeds[:, -num_thoughts:]

        # handle maybe multiple hypothesis

        if has_multi_hyp:
            latent_token = self.to_streams(latent_token)

            cached_kv = repeat(cached_kv, '... b h n d -> ... (b hyp) h n d', hyp = self.num_hypothesis)

            num_steps_multistream = self.num_reasoning_steps - 1

        # latent reasoning is a recurrent model forward with the last hidden state being passed back in as input, while the prompt key / values are kept the same (prompt is NOT passed back in)

        latent_tokens = [latent_token]

        for _ in range(self.num_reasoning_steps - 1):
            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, return_embed_with_cache_kv = True)

            if has_multi_hyp:
                latent_token = self.maybe_synth_streams(latent_token)

            latent_tokens.append(latent_token)

        # merge hypothesis if needed

        if has_multi_hyp:
            latent_token = self.merge_streams(latent_token)

            cached_kv = stack(cached_kv)

            # for the accumulated key / values ...

            cached_kv_orig, cached_kv_multistream = cached_kv[..., -num_steps_multistream:, :], cached_kv[..., :-num_steps_multistream:, :]

            # 1. average back the original cached key / values before the latent reasoning steps

            cached_kv_orig = reduce(cached_kv_orig, '... (b hyp) h n d -> ... b h n d', 'mean', hyp = self.num_hypothesis)

            # 2. allow the <eot> and subsequent answer tokens to see all of key / values across all hypothesis

            cached_kv_multistream = rearrange(cached_kv_multistream, '... (b hyp) h n d -> ... b h (hyp n) d', hyp = self.num_hypothesis)

            # 3. concat the original cached key / values with the flattened key / values for all hypothesis

            cached_kv = cat((cached_kv_orig, cached_kv_multistream), dim = -2)

        # final step, latent token and end thought token, as well as answer sequence is appended together

        logits = self.model([latent_token, end_thought, answer[..., :-1]], cached_kv = cached_kv)

        answer_logits = logits[:, num_thoughts:]

        # concat the latent reasoning tokens to be passed out for study

        if not self.has_multiple_hypothesis:
            latent_tokens = cat(latent_tokens, dim = -2)

        intermediates = prompt_logits, latent_tokens, answer_logits

        if not return_loss:
            return intermediates

        # handle the loss on the answers

        answer_loss = F.cross_entropy(
            rearrange(answer_logits, 'b n l -> b l n'),
            answer
        )

        loss_breakdown = (answer_loss, bot_loss)

        total_loss = (answer_loss + bot_loss)

        return total_loss, (loss_breakdown, *intermediates)
