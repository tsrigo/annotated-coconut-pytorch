import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from coconut import CustomGPT2LMHeadModel

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