import torch
from coconut_pytorch import Coconut
from transformers import GPT2Tokenizer

# Initialize tokenizer and device
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Coconut model
model = Coconut(
    num_reasoning_steps=3,  # Number of reasoning steps
    num_latents_per_step=2,  # Number of latent tokens per step
    transformer=dict(
        num_tokens=tokenizer.vocab_size,  # Vocabulary size
        dim=512,  # Embedding dimension
        depth=6   # Transformer depth
    )
).to(device)

# Synthesized single data point
question = "What is 2 + 3 + 4?"
steps = ["2 + 3 = 5", "5 + 4 = 9"]
answer = "9"

# Tokenize the data
question_tokens = tokenizer(question, return_tensors="pt").input_ids.to(device)
steps_tokens = [tokenizer(step, return_tensors="pt").input_ids.to(device) for step in steps]
answer_tokens = tokenizer(answer, return_tensors="pt").input_ids.to(device)

# Training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
stages = 3  # Number of training stages

# Multi-stage training
for stage in range(stages):
    print(f"Training Stage {stage + 1}/{stages}")
    model.train()

    for epoch in range(num_epochs):
        # Prepare inputs and targets for the current stage
        if stage == 0:
            # Stage 0: Full reasoning steps
            prompt = question_tokens
            target = torch.cat(steps_tokens + [answer_tokens], dim=1)
        elif stage == 1:
            # Stage 1: Replace Step 1 with [Thought]
            prompt = torch.cat([question_tokens, tokenizer("<bot> [Thought] <eot>", return_tensors="pt").input_ids.to(device)], dim=1)
            target = torch.cat(steps_tokens[1:] + [answer_tokens], dim=1)
        elif stage == 2:
            # Stage 2: Replace all steps with [Thought]
            prompt = torch.cat([question_tokens, tokenizer("<bot> [Thought] [Thought] <eot>", return_tensors="pt").input_ids.to(device)], dim=1)
            target = answer_tokens
        
        print("*" * 25 + f"Stage: {stage}, Epoch: {epoch}" + "*" * 25 + '\n')
        print("Prompt:", tokenizer.decode(prompt[0], skip_special_tokens=True))
        print("Target:", tokenizer.decode(target[0], skip_special_tokens=True))
        # Forward pass
        loss = model(prompt, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Example generation
model.eval()
with torch.no_grad():
    generated_tokens = model.generate(question_tokens, max_length=64)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"Generated Answer: {generated_text}")
