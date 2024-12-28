# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from coconut_pytorch import Coconut
import time

# 定义训练的设备
device = torch.device("cpu")

prompt = torch.randint(0, 256, (2, 1024))   # 大小为 (2, 1024) 的随机张量, 相当于 2 个 prompt，每个 1024 个 token
answer = torch.randint(0, 256, (2, 64))  
train_data = TensorDataset(prompt, answer)
prompt = torch.randint(0, 256, (2, 1024))   # 大小为 (2, 1024) 的随机张量, 相当于 2 个 prompt，每个 1024 个 token
answer = torch.randint(0, 256, (2, 64))  
test_data = TensorDataset(prompt, answer)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)


model = Coconut(
    num_reasoning_steps = 3,
    num_latents_per_step = 1,
    transformer = dict(
        num_tokens = tokenizer.vocab_size,
        dim = 512,
        depth = 6
    )
)
model = model.to(device)
# 损失函数
None
# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

st = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        prompt, answer = data
        prompt = prompt.to(device)
        answer = answer.to(device)    
        loss, (loss_breakdown, *intermediates) = model(prompt, answer, return_intermediates = True)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练{}次所用时间：{}".format(total_train_step, time.time()-st))
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            prompt, answer = data
            prompt = prompt.to(device)
            answer = answer.to(device)
            loss, (loss_breakdown, *intermediates) = model(prompt, answer, return_intermediates = True)
            outputs = model.generate(prompt, max_length = 64) # (2, 64)
            total_test_loss = total_test_loss + loss.item()
            print("Answer: ", answer)
            print("Output: ", outputs)
            accuracy = 0
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(model, "tudui_{}.pth".format(i))
    # print("模型已保存")

writer.close()