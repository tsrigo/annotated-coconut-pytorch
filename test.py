import torch
from src.coconut_pytorch import Coconut

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model = Coconut(
    num_reasoning_steps = 3,
    num_latents_per_step = 2,
    transformer = dict(
        num_tokens = 256,
        dim = 512,
        depth = 6
    ),
    model = 'gpt2'
).to(device)


prompt = torch.randint(0, 256, (1, 54))   # 大小为 (2, 1024) 的随机张量, 相当于 2 个 prompt，每个 1024 个 token
answer = torch.randint(0, 256, (1, 54))   
prompt, answer = prompt.to(device), answer.to(device)

loss = model(prompt, answer)
loss.backward()

# after much training

answer = model.generate(prompt, max_length = 64) # (2, 64)
print(prompt)
print(answer)