from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, einsum, Tensor
from torch.nn import Module, ModuleList

from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
from transformers import GPT2LMHeadModel, GPT2Config
from torch.nn import CrossEntropyLoss

from x_transformers.attend import Attend

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def append(t, value, num = 1):
    '''
    :param t: Tensor, 要进行填充的张量
    :param value: Tensor, 用于填充的值
    :param num: int, 填充的数量，默认为 1
    :return: Tensor
    
    功能：在张量 t 的最后一个维度上填充指定数量 num 的值 value
    '''

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
            q = apply_rotary_emb(rotary_pos_emb, q, freqs_seq_dim = -2)
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
        mask: Tensor | None = None,
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

        if exists(mask):
            seq = mask.cumsum(dim = -1)
            seq = rearrange(seq, 'b n -> b 1 n')

        else:
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
                rotary_pos_emb = rotary_pos_emb,
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


class CustomGPT2LMHeadModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True  # 确保返回字典格式的结果
        )
        self.model = GPT2LMHeadModel(config)
        self.dim = hidden_size

    def forward(
        self,
        inp,
        cached_kv=None,
        mask=None,
        return_intermediates=False,
        return_embed_with_cache_kv=False
    ):
        """
        Forward pass through the model.
        
        Parameters:
        - inp: Tensor 或多个 Tensor 的列表
        - cached_kv: 前一次缓存的键值对
        - mask: 可选的mask，用于防止模型不必要的关注
        - return_intermediates: 是否返回(logits, 中间状态, cached_kv)
        - return_embed_with_cache_kv: 是否只返回(embeds, cached_kv)
        """
        # 如果输入是单个 Tensor，则转为列表
        if not isinstance(inp, list):
            inp = [inp]

        # 将列表中的 int/long 类型输入转换为 GPT-2 可处理的形式
        # 这里只做简单拼接处理，实际应用可根据需要进一步定制
        x = []
        for t in inp:
            if t.dtype in (torch.int, torch.long):
                x.append(t)
            else:
                # 模拟直接使用 embedding 后的张量
                # 如果需要更复杂的embedding逻辑，可在此添加
                x.append(t.argmax(dim=-1) if t.dim() == 3 else t)

        # 将所有输入拼接到一起
        in_ids = torch.cat(x, dim=1)

        # huggingface GPT2LMHeadModel forward 参数与本地命名差异
        # past_key_values对应 cached_kv
        # attention_mask对应 mask
        outputs = self.model(
            input_ids=in_ids,
            past_key_values=cached_kv,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
            # return_dict=True  # Ensure dictionary output format
        )
        # GPT2LMHeadModel 返回:
        # logits, past_key_values, hidden_states, ...
        logits = outputs.logits
        next_cached_kv = outputs.past_key_values  # 下次可复用
        hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏层输出
        

        if return_embed_with_cache_kv:
            # 返回 (embeddings, cached_kv)
            return hidden_states, next_cached_kv

        if return_intermediates:
            # 返回 (logits, embeds, cached_kv)
            return logits, hidden_states, next_cached_kv

        # 默认仅返回 logits
        return logits


# coconut wrapper around transformer handles recurrence with latent reasoning tokens

class Coconut(Module):
    def __init__(
        self,
        num_reasoning_steps,        # 表示模型进行 recurrent reasoning 的步数或轮次。换句话说，它是模型在生成最终输出之前，将输入通过自身处理多少次。扩展了论文。
        transformer: dict | Transformer,
        num_latents_per_step = 1,   # 每个推理步骤生成的 latent thought 数量；这个应该是论文中的 c
        learn_begin_of_thought = False,  # 是否学习 <bot> token
        begin_thought_loss_weight = 1.,  # 思考开始的损失权重
        checkpoint = False,         # 是否使用检查点优化内存
        model = 'gpt2',
    ):
        super().__init__()

        # # 如果传入的 transformer 是字典形式，则实例化 Transformer 对象
        # if isinstance(transformer, dict):
        #     transformer = Transformer(**transformer)

        # dim = transformer.dim  # transformer 的嵌入维度
        if model == 'transformer':
            print("transformer")
            if isinstance(transformer, dict):
                transformer = Transformer(**transformer)
            dim = transformer.dim
            self.model = transformer
        elif model == 'gpt2':
            print("gpt2")
            self.model = CustomGPT2LMHeadModel()
            dim = self.model.dim
        else:
            raise ValueError("model must be 'transformer' or 'gpt2'")
                
        self.num_reasoning_steps = num_reasoning_steps

        # <bot> 和 <eot> token，论文中用于标记思考开始和结束
        self.begin_of_thought = nn.Parameter(torch.zeros(num_latents_per_step, dim))  # 可训练的思考开始 token
        self.end_of_thought = nn.Parameter(torch.zeros(dim))  # 可训练的思考结束 token

        # 使用正态分布初始化这两个 token
        nn.init.normal_(self.begin_of_thought, std=0.02)
        nn.init.normal_(self.end_of_thought, std=0.02)

        # 是否学习思考开始标记
        self.learn_begin_of_thought = learn_begin_of_thought
        self.begin_thought_loss_weight = begin_thought_loss_weight

        # 用于存储不需要持久化的零向量
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        # 是否使用检查点进行优化
        self.checkpoint = checkpoint

        # self.base_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_length = 16,
        filter_fn = min_p_filter,
        filter_kwargs: dict = dict(),
        temperature = 1.
    ):
        '''
        这段代码实现了自回归（autoregressive）的生成方式。
        0. 应用 forward 函数，获取 answer_logits
        1. 初始化输出
        2. 定义 sample 函数，用于从 logits 中采样
            2.1 初始采样，从 answer_logits 中采样
            2.2 逐步采样，将 answer_logits 作为 self.model 的输入，获取下一个 token 的 logits，然后 sample
        3. 逐步生成，直到达到最大长度

        流程图：
        prompt -> forward -> answer_logits -> sample -> out -> answer_logits -> sample -> out -> ... -> out
        '''
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

        num_thoughts = latent_token.shape[-2]
        num_recurrent_steps = self.num_reasoning_steps - 1

        # recurrent model forward with next latent token + past cached kv, but checkpointed

        latent_tokens = [latent_token, *((None,) * num_recurrent_steps)]

        def recurrent_step(step_inputs):
            i, latent_token, *latent_tokens, cached_kv, mask = step_inputs

            i += 1

            mask = append(mask, True, num_thoughts)
            # return_embed_with_cache_kv = True，返回 embeds 和 cached_kv
            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, mask = mask, return_embed_with_cache_kv = True)

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
        num_thoughts = latent_token.shape[-2]

        # latent reasoning is a recurrent model forward with the last hidden state being passed back in as input, while the prompt key / values are kept the same (prompt is NOT passed back in)
        # 潜在推理是一个循环模型，其中最后一个隐藏状态被回传作为输入，而提示键/值保持不变（提示不会被回传）。
        latent_tokens = [latent_token]

        # 在这里，latent_token 的数量增加了 num_reasoning_steps - 1 个
        # 也就是论文当中的添加隐含推理的过程
        # 为什么要减 1 呢？因为在 forward 函数中，已经进行了一次推理（生成了一个latent_token），所以这里只需要进行 num_reasoning_steps - 1 次推理
        # 或者说因为输入的 latent_token 已经是第一个 token 了，最后返回的 latent_token 数量就是 num_reasoning_steps
        for _ in range(self.num_reasoning_steps - 1):

            mask = append(mask, True, num_thoughts)

            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, mask = mask, return_embed_with_cache_kv = True)

            latent_tokens.append(latent_token)

        return latent_token, latent_tokens, cached_kv, mask

    def forward(
        self,
        prompt: Tensor | list[Tensor],
        answer: Tensor | list[Tensor] | None = None,
        return_loss=True,
        return_intermediates=False
    ):
        """
        前向传播方法，处理输入 prompt 和 answer，执行推理逻辑。
        输入：prompt - 输入序列；answer - 目标序列（可选）；return_loss - 是否返回损失；return_intermediates - 是否返回中间结果。
        """

        # 处理可变长度的 prompt 序列
        if isinstance(prompt, (list, tuple)):
            prompt = pad_sequence(prompt, padding_value=-1, batch_first=True)

        if exists(answer) and isinstance(answer, (list, tuple)):
            answer = pad_sequence(answer, padding_value=-1, batch_first=True)

        # 创建一个掩码，标记哪些位置是有效的（prompt >= 0）
        mask = prompt >= 0
        prompt = prompt.masked_fill(~mask, 0)  # 将无效位置填充为 0

        # 获取输入的 batch 大小和潜在变量的维度
        batch, num_thoughts = prompt.shape[0], self.begin_of_thought.shape[0]

        # 准备 <bot> 和 <eot> token
        begin_thought = repeat(self.begin_of_thought, 'n d -> b n d', b=batch)
        end_thought = repeat(self.end_of_thought, 'd -> b 1 d', b=batch)

        # 给模型输入 prompt 和 <bot> token
        mask = append(mask, True, num_thoughts)
        prompt_logits, embeds, cached_kv = self.model([prompt, begin_thought], mask=mask, return_intermediates=True)

        # 如果需要，计算 <bot> token 的损失
        bot_loss = self.zero
        if self.learn_begin_of_thought:
            pred_bot_embed, rest_logits = embeds[:, -2], prompt_logits[:, -2]
            pred_bot_logits = einsum('b d, n d -> b', pred_bot_embed, self.begin_of_thought)
            pred_bot_logits = rearrange(pred_bot_logits, 'b -> b 1')
            bot_logits = cat((pred_bot_logits, rest_logits), dim=-1)
            bot_labels = bot_logits.new_zeros((batch,), dtype=torch.long)
            bot_loss = F.cross_entropy(bot_logits, bot_labels)

        # 提取潜在的推理 token
        latent_token = embeds[:, -num_thoughts:]

        # 如果使用检查点，则选择相应的前向函数
        should_checkpoint = self.training and self.checkpoint and latent_token.requires_grad
        latent_reasoning_fn = self.checkpointed_recurrent_latent_forward if should_checkpoint else self.recurrent_latent_forward

        # 通过循环推理过程更新 latent token
        # 当 c = 1 时，latent_tokens = [latent_token]
        latent_token, latent_tokens, cached_kv, mask = latent_reasoning_fn(latent_token, cached_kv, mask=mask)

        # 最终前向输入，合并 latent token 和 <eot> token
        # TODO: 为什么只用了一个 latent_token，而不是 latent_tokens？
        final_forward = [latent_token, end_thought]
        # final_forward = latent_tokens + [end_thought]
        
        mask = append(mask, True, 1 + num_thoughts)

        if exists(answer):
            answer_input = answer[..., :-1] # TODO: 最后一个 token 是？
            answer_input_mask = answer_input >= 0
            answer_input = answer_input.masked_fill(~answer_input_mask, 0)
            mask = torch.cat((mask, answer_input_mask), dim=-1)
            final_forward.append(answer_input)

        # 最终模型计算输出 logits
        logits = self.model(final_forward, cached_kv=cached_kv, mask=mask)
        answer_logits = logits[:, num_thoughts:]

        # 合并潜在推理 tokens 以供进一步研究
        latent_tokens = cat(latent_tokens, dim=-2)

        # 返回中间结果
        intermediates = prompt_logits, latent_tokens, answer_logits, cached_kv

        if not return_loss or not exists(answer):
            return intermediates

        # 计算 answer 的损失
        answer_loss = F.cross_entropy(
            rearrange(answer_logits, 'b n l -> b l n'),
            answer,
            ignore_index=-100
        )

        # 返回总损失
        loss_breakdown = (answer_loss, bot_loss)
        total_loss = answer_loss + bot_loss * self.begin_thought_loss_weight

        if not return_intermediates:
            return total_loss

        return total_loss, (loss_breakdown, *intermediates)
    
    def compute_loss(self, input_ids, labels, position_ids=None, output_attentions=False):
        loss, (loss_breakdown, *intermediates) = self.forward(input_ids, labels, return_intermediates = True)
        prompt_logits, latent_tokens, answer_logits, cached_kv = intermediates
        logits = answer_logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # outputs.loss = loss
        # outputs.token_accuracy = token_accuracy
        # outputs.total_correct = correct_tokens
        # outputs.total_loss = loss * total_tokens
        # outputs.total_tokens = total_tokens
        
        outputs = {}
        outputs['loss'] = loss
        outputs['token_accuracy'] = token_accuracy
        outputs['total_correct'] = correct_tokens
        outputs['total_loss'] = loss * total_tokens
        outputs['total_tokens'] = total_tokens
        
        return outputs

