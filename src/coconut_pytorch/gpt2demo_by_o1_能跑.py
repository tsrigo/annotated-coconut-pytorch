import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class CustomGPT2LMHeadModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            use_cache=True,
            output_hidden_states=True
        )
        self.model = GPT2LMHeadModel(config)

    def forward(
        self,
        inp,
        cached_kv=None,
        mask=None,
        return_intermediates=False,
        return_embed_with_cache_kv=False
    ):
        """
        inp: Tensor 或多个 Tensor 的列表
        cached_kv: 前一次缓存的键值对
        mask: 可选的mask，用于防止模型不必要的关注
        return_intermediates: 是否返回(logits, 中间状态, cached_kv)
        return_embed_with_cache_kv: 是否只返回(embeds, cached_kv)
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
            output_hidden_states=True
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

def main():
    # 简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机构造输入
    vocab_size = 1000
    batch_size = 2
    seq_len = 3
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    model = CustomGPT2LMHeadModel(vocab_size=vocab_size).to(device)

    # 1) 仅返回最后一步logits
    answer_logits = model(dummy_input[:, -1:], cached_kv=None)
    print("answer_logits shape:", answer_logits.shape)

    # 2) 返回 (latent_token, cached_kv)
    latent_token, cached_kv = model(dummy_input, return_embed_with_cache_kv=True)
    print("latent_token shape:", latent_token.shape)

    # 3) 再次调用，传入上一步缓存
    latent_token, cached_kv = model(dummy_input, cached_kv=cached_kv, return_embed_with_cache_kv=True)
    print("latent_token with cache shape:", latent_token.shape)

    # 4) 同时返回logits, embeds, cached_kv
    prompt_logits, embeds, cached_kv = model(
        [dummy_input, dummy_input],
        mask=None,
        return_intermediates=True
    )
    print("prompt_logits shape:", prompt_logits.shape, "embeds shape:", embeds.shape)

    # 5) 仅返回logits (不带中间输出)
    final_logits = model(dummy_input, cached_kv=cached_kv)
    print("final_logits shape:", final_logits.shape)

if __name__ == "__main__":
    main()