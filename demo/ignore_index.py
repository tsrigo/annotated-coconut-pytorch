from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F
from einops import rearrange

# 初始化 GPT-2 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 检查当前 tokenizer 的特殊标记
print("Current special tokens:", tokenizer.special_tokens_map)

# 如果没有 pad_token，添加一个
if 'pad_token' not in tokenizer.special_tokens_map:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 再次检查特殊标记
print("Updated special tokens:", tokenizer.special_tokens_map)

# 示例句子
sentence = "This is a sample sentence."

# 编码句子
inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

# 获取 pad_token_id
pad_token_id = tokenizer.pad_token_id

# 假设 answer_logits 是模型预测的概率分布，形状为 (batch_size, num_classes, sequence_length)
answer_logits = torch.randn(1, tokenizer.vocab_size, inputs['input_ids'].shape[1])

# 创建一个示例 answer 张量，包含有效的类别索引
answer = torch.randint(0, tokenizer.vocab_size, inputs['input_ids'].shape)

# 将 attention_mask 为 0 的位置设为 pad_token_id
answer[inputs['attention_mask'] == 0] = pad_token_id

# 计算交叉熵损失，忽略目标值为 pad_token_id 的位置
loss = F.cross_entropy(
    rearrange(answer_logits, 'b n l -> b l n'),  # 将 logits 转换为 (batch_size, sequence_length, num_classes)
    answer.squeeze(),  # 确保 answer 是一维的
    ignore_index=pad_token_id  # 忽略目标值为 pad_token_id 的位置
)

print(loss)
