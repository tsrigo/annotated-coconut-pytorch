import torch
from einops import repeat

# 示例参数
batch = 2  # 批量大小
n = 3      # begin_of_thought 的维度
d = 4      # 特征维度

# 初始化 begin_of_thought 和 end_of_thought 使用固定数值
begin_of_thought = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])  # 形状: (3, 4)

end_of_thought = torch.tensor([0, 0, 0, 0])  # 形状: (4,)

print("Old begin_of_thought shape:", begin_of_thought.shape)  # 输出: torch.Size([3, 4])
print("Old end_of_thought shape:", end_of_thought.shape)      # 输出: torch.Size([4])
print("begin_of_thought:")
print(begin_of_thought)
print("end_of_thought:")
print(end_of_thought)

# 使用 einops.repeat 扩展张量
begin_thought = repeat(begin_of_thought, 'n d -> b n d', b=batch)   # 将原始的 (n, d) 形状扩展为 (b, n, d)
# begin_thought 张量被扩展为包含 batch 个相同的 (n, d) 张量，形成一个三维张量 (b, n, d)。
# 这在批处理操作中非常有用，因为它允许在一个批次中同时处理多个样本，每个样本都拥有相同的“开始思考”向量。
end_thought = repeat(end_of_thought, 'd -> b 1 d', b=batch)         # 将原始的 (d,) 形状扩展为 (b, 1, d)
# end_thought 被扩展为一个三维张量 (b, 1, d)，其中每个批次中的样本都有一个独立的“结束思考”向量。
# 这为后续的计算提供了统一的结束标记,确保每个样本都能正确地结束其思考过程。
print("New begin_thought shape:", begin_thought.shape)  # 输出: torch.Size([2, 3, 4])
print("New end_thought shape:", end_thought.shape)      # 输出: torch.Size([2, 1, 4])
print("begin_thought:")
print(begin_thought)
print("end_thought:")
print(end_thought)