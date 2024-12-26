import torch

# 创建一个形状为 (4,) 的一维张量
tensor_1d = torch.tensor([1, 2, 3, 4])
print("一维张量 (d,):", tensor_1d)
print("形状:", tensor_1d.shape)

# 创建一个形状为 (1, 4) 的二维张量
tensor_2d_row = torch.tensor([[1, 2, 3, 4]])
print("二维张量 (1, d):", tensor_2d_row)
print("形状:", tensor_2d_row.shape)

# 创建一个形状为 (4, 1) 的二维张量
tensor_2d_col = torch.tensor([[1], [2], [3], [4]])
print("二维张量 (d, 1):")
print(tensor_2d_col)
print("形状:", tensor_2d_col.shape)