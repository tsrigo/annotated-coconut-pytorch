import torch

prompt = torch.tensor([[ 1, -1,  2, -3,  0],
                       [ 0,  5, -2,  3, -4]])  # 示例张量

mask = prompt >= 0
prompt = prompt.masked_fill(~mask, 0)  # 不满足条件的位置填充为 0
print(prompt)
# output: tensor([[1, 0, 2, 0, 0],
#                 [0, 5, 0, 3, 0]])