import torch
from transformers import StoppingCriteria, LogitsProcessor


def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def batch_ids(input_ids_list, pad_token_id, device, dtype):
    max_seq_len = max([len(item) for item in input_ids_list])
    batch_size = len(input_ids_list)
    input_ids = torch.Tensor(batch_size, max_seq_len).to(dtype).to(device)
    input_ids.fill_(pad_token_id)
    for batch_id in range(batch_size):
        input_ids[batch_id, :len(input_ids_list[batch_id])] = input_ids_list[batch_id]
    return input_ids


def get_sep_position(input_ids, sep_id, skip=0):
    """
    获取输入序列中分隔符的位置。具体来说，返回每个批次中，第 skip 个分隔符的位置。

    参数:
        input_ids (torch.Tensor): 输入序列的ID张量，形状为(batch_size, sequence_length)。
        sep_id (int): 分隔符的ID。
        skip (int, optional): 在找到最后一个分隔符后跳过的分隔符数量。默认为0。

    返回:
        torch.Tensor: 每个批次中分隔符的位置索引，形状为(batch_size,)。

    例子:
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 2, 3, 4, 1]])
        >>> get_sep_position(input_ids, 5)
        tensor([4, 0])

        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [5, 2, 3, 4, 1, 5, 2, 3, 4, 1]])
        >>> get_sep_position(input_ids, 5)
        tensor([4, 0])

        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [5, 2, 3, 4, 1, 5, 2, 3, 4, 1]])
        >>> get_sep_position(input_ids, 5, skip=1)
        tensor([9, 5])

    """
    batch_size = input_ids.shape[0] # 获取批次大小
    sep_positions = input_ids.new_zeros(batch_size).long()  # 初始化存储分隔符位置的张量
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)       # 获取分隔符的掩码  
        sep_position = mask.nonzero()[0, -1].item() # 获取最后一个分隔符的位置
        for _ in range(skip):
            mask[sep_position] = False              # 将最后一个分隔符的位置设置为False
            sep_position = mask.nonzero()[0, -1].item() # 更新最后一个分隔符的位置
        sep_positions[batch_id] = sep_position      # 存储分隔符的位置
    return sep_positions


# Stop generation only after generating two EOSs, such as  z <eos> y <eos>
class DoubleEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False
    
    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id] = 0
        return scores
