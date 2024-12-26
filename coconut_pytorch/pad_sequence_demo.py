import torch
from torch.nn.utils.rnn import pad_sequence

def exists(x):
    return x is not None

# 示例输入：三个长度不一的张量
answer = [
    torch.tensor([1, 2]),
    torch.tensor([3, 4, 5]),
    torch.tensor([6])
]

if exists(answer) and isinstance(answer, (list, tuple)):
    answer = pad_sequence(answer, padding_value=-1, batch_first=True)

print(answer)
# 输出中，较短的张量在末尾被补上 -1，以对齐长度
# tensor([[ 1,  2, -1],
#         [ 3,  4,  5],
#         [ 6, -1, -1]])
```
