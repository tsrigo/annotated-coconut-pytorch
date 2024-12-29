import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits
print(loss, logits)

# self.model = transformer
# answer_logits = self.model(out[:, -1:], cached_kv = cached_kv) # 方法1：传入输入，得到 answer_logits
# latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, mask = mask, return_embed_with_cache_kv = True)
# latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, mask = mask, return_embed_with_cache_kv = True)
# prompt_logits, embeds, cached_kv = self.model([prompt, begin_thought], mask=mask, return_intermediates=True)
# logits = self.model(final_forward, cached_kv=cached_kv, mask=mask)

# 需要从 GPT2 （base model） 获取的信息：
# 1. answer_logits
# 2. latent_token (embeds)
# 3. cached_kv
# 4. prompt_logits
# 5. logits

# 需要传入 GPT2 的信息：
# 1. inp: Tensor | list[Tensor],
# 2. cached_kv: Tensor | None = None,
# 3. mask: Tensor | None = None,
# 4. return_intermediates = False,
# 5. return_embed_with_cache_kv = False