# improvised version of coconut that allows multiple streams during the recurrent latent reasoning
# https://arxiv.org/abs/2107.10342

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

from x_transformers.attend import Attend

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def append(t, value, num = 1):
    return F.pad(t, (0, num), value = value)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def min_p_filter(logits, min_p = 0.1):
    """
    https://arxiv.org/abs/2407.01082
    """
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

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
        attend_kwargs: dict = dict(),
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim)

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.attend = Attend(causal = True, **attend_kwargs)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        mask: Tensor | None = None,
        cached_kv: Tensor | None = None,
        return_cached_kv = False,
        rotary_pos_emb = None
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

        if exists(rotary_pos_emb):
            q = apply_rotary_emb(rotary_pos_emb, q)
            k = apply_rotary_emb(rotary_pos_emb, k)

        out, _ = self.attend(q, k, v, mask = mask)

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
        ff_mult = 4,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, attend_kwargs = attend_kwargs),
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

        # determine cached kv length

        cached_kv_len = cached_kv[0].shape[-2] if exists(cached_kv) else 0.

        # rotary pos emb

        total_seq_len = x.shape[-2] + cached_kv_len

        seq = torch.arange(total_seq_len, device = x.device)

        rotary_pos_emb = self.rotary_emb(seq)

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
        synthesize_hypothesis_per_step = False,
        checkpoint = False,
        begin_thought_loss_weight = 1.,
        hyp_diff_loss_weight = 1.
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer)

        dim = transformer.dim

        self.model = transformer
        self.num_reasoning_steps = num_reasoning_steps

        # begin and end of thought tokens, handled external to transformer

        self.num_thoughts = num_latents_per_step

        self.begin_of_thought = nn.Parameter(torch.zeros(num_latents_per_step, dim))
        self.end_of_thought = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(self.begin_of_thought, std = 0.02)
        nn.init.normal_(self.end_of_thought, std = 0.02)

        # whether to teach model when to begin a thought

        self.learn_begin_of_thought = learn_begin_of_thought
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # checkpoint

        self.checkpoint = checkpoint

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

        # loss related

        self.begin_thought_loss_weight = begin_thought_loss_weight
        self.hyp_diff_loss_weight = hyp_diff_loss_weight

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_length = 16,
        filter_fn = min_p_filter,
        filter_kwargs: dict = dict(),
        temperature = 1.
    ):
        prompt_logits, latent_tokens, answer_logits, cached_kv = self.forward(prompt)

        out = prompt[:, 0:0]

        def sample(logits):
            nonlocal out
            logits = filter_fn(logits[:, -1], **filter_kwargs)
            sampled = gumbel_sample(logits, temperature = temperature)
            out = cat((out, sampled), dim = -1)

        sample(answer_logits)

        for _ in range(max_length - 1):
            answer_logits = self.model(out[:, -1:], cached_kv = cached_kv)
            sample(answer_logits)

        return out

    def checkpointed_recurrent_latent_forward(
        self,
        latent_token,
        cached_kv,
        mask
    ):
        if not torch.is_tensor(cached_kv):
            cached_kv = stack(cached_kv)

        num_thoughts = self.num_thoughts
        num_recurrent_steps = self.num_reasoning_steps - 1

        # recurrent model forward with next latent token + past cached kv, but checkpointed

        latent_tokens = [latent_token, *((None,) * num_recurrent_steps)]

        def recurrent_step(step_inputs):
            i, latent_token, *latent_tokens, cached_kv, mask = step_inputs

            i += 1

            is_last_step = i == num_recurrent_steps

            mask = append(mask, True, num_thoughts)

            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, return_embed_with_cache_kv = True)

            if self.has_multiple_hypothesis and not is_last_step:
                latent_token = self.maybe_synth_streams(latent_token)

            latent_tokens[i] = latent_token

            return (i, latent_token, *latent_tokens, stack(cached_kv), mask)

        # functions

        fns = [recurrent_step] * num_recurrent_steps

        # initial input

        inputs = (0, latent_token, *latent_tokens, cached_kv, mask)

        # forward checkpoint sequential

        _, latent_token, *latent_tokens, cached_kv, mask = checkpoint_sequential(fns, 1, input = inputs, use_reentrant = False)

        return latent_token, latent_tokens, cached_kv, mask

    def recurrent_latent_forward(
        self,
        latent_token,
        cached_kv,
        mask
    ):
        # latent reasoning is a recurrent model forward with the last hidden state being passed back in as input, while the prompt key / values are kept the same (prompt is NOT passed back in)

        latent_tokens = [latent_token]

        num_recurrent_steps = self.num_reasoning_steps - 1
        num_thoughts = self.num_thoughts

        for i in range(num_recurrent_steps):
            is_last_step = i == (num_recurrent_steps - 1)

            mask = append(mask, True, num_thoughts)

            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, return_embed_with_cache_kv = True)

            if self.has_multiple_hypothesis and not is_last_step:
                latent_token = self.maybe_synth_streams(latent_token)

            latent_tokens.append(latent_token)

        return latent_token, latent_tokens, cached_kv, mask

    def forward(
        self,
        prompt: Tensor | list[Tensor],
        answer: Tensor | list[Tensor] | None = None,
        return_loss = True,
        return_intermediates = False
    ):
        """
        ein notation:
        b - batch
        h - attention heads
        n - seq (number of reasoning tokens, etc)
        d - feature dimension
        l - logits (num tokens)
        """

        # handle variable length prompts

        if isinstance(prompt, (list, tuple)):
            prompt = pad_sequence(prompt, padding_value = -1, batch_first = True)

        if exists(answer) and isinstance(answer, (list, tuple)):
            answer = pad_sequence(answer, padding_value = -1, batch_first = True)

        mask = prompt >= 0

        prompt = prompt.masked_fill(~mask, 0)

        # shapes and variables

        batch, num_thoughts, has_multi_hyp, num_hyp = prompt.shape[0], self.begin_of_thought.shape[0], self.has_multiple_hypothesis, self.num_hypothesis

        # prepare <bot> and <eot> in paper

        begin_thought = repeat(self.begin_of_thought, 'n d -> b n d', b = batch)
        end_thought = repeat(self.end_of_thought, 'd -> b 1 d', b = batch)

        # give the model the prompt

        mask = append(mask, True, num_thoughts)

        prompt_logits, embeds, cached_kv = self.model([prompt, begin_thought], return_intermediates = True)

        # loss for decoding a <bot>

        bot_loss = self.zero

        if self.learn_begin_of_thought:
            pred_bot_embed, rest_logits = embeds[:, -2], prompt_logits[:, -2]

            pred_bot_logits = einsum('b d, n d -> b', pred_bot_embed, self.begin_of_thought)
            pred_bot_logits = rearrange(pred_bot_logits, 'b -> b 1')

            bot_logits = cat((pred_bot_logits, rest_logits), dim = -1)
            bot_labels = bot_logits.new_zeros((batch,), dtype = torch.long)

            bot_loss = F.cross_entropy(bot_logits, bot_labels)

        # extract latent reasoning token off <bot> position

        latent_token = embeds[:, -num_thoughts:]

        # handle maybe multiple hypothesis

        if has_multi_hyp:
            latent_token = self.to_streams(latent_token)

            cached_kv = repeat(cached_kv, '... b h n d -> ... (b hyp) h n d', hyp = num_hyp)

            num_steps_multistream = self.num_reasoning_steps - 1

        # maybe an aux loss to encourage each hypothesis to be different

        hyp_diff_loss = self.zero

        if has_multi_hyp and self.hyp_diff_loss_weight > 0.:
            latent_token_split_hyp = rearrange(latent_token, '(b hyp) ... d -> (b ...) hyp d', hyp = num_hyp)

            normed_hyp = l2norm(latent_token_split_hyp)
            cosine_sim = einsum('... i d, ... j d -> ... i j', normed_hyp, normed_hyp)

            upper_right_tri = cosine_sim.triu(1)                      # only consider in aux loss cosine sim of latent token to all other latents, not to self
            weight = (num_hyp ** 2) / ((num_hyp ** 2 - num_hyp) / 2)  # omit diagonal then divide by 2 as symmetric

            hyp_diff_loss = upper_right_tri.mean() * weight

        # whether to checkpoint or not

        should_checkpoint = self.training and self.checkpoint and latent_token.requires_grad

        # route to right functions

        latent_reasoning_fn = self.checkpointed_recurrent_latent_forward if should_checkpoint else self.recurrent_latent_forward

        latent_token, latent_tokens, cached_kv, mask = latent_reasoning_fn(latent_token, cached_kv, mask = mask)

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

        final_model_forward = [latent_token, end_thought]

        if exists(answer):
            answer_input = answer[..., :-1]
            answer_input_mask = answer_input >= 0

            answer_input = answer_input.masked_fill(~answer_input_mask, 0)

            mask = torch.cat((mask, answer_input_mask), dim = -1)

            final_model_forward.append(answer_input)

        logits = self.model(final_model_forward, cached_kv = cached_kv)

        answer_logits = logits[:, num_thoughts:]

        # concat the latent reasoning tokens to be passed out for study

        if not self.has_multiple_hypothesis:
            latent_tokens = cat(latent_tokens, dim = -2)

        intermediates = prompt_logits, latent_tokens, answer_logits, cached_kv

        if not return_loss or not exists(answer):
            return intermediates

        # handle the loss on the answers

        answer_loss = F.cross_entropy(
            rearrange(answer_logits, 'b n l -> b l n'),
            answer
        )

        loss_breakdown = (answer_loss, bot_loss, hyp_diff_loss)

        total_loss = (
            answer_loss +
            bot_loss * self.begin_thought_loss_weight +
            hyp_diff_loss * self.hyp_diff_loss_weight
        )

        if not return_intermediates:
            return total_loss

        return total_loss, (loss_breakdown, *intermediates)
