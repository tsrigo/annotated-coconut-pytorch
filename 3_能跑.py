import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from coconut_pytorch import Coconut
from transformers import set_seed, GPT2Tokenizer

# Set random seed for reproducibility
set_seed(42)

# Initialize tokenizer and device
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Coconut model
model = Coconut(
    num_reasoning_steps=3,  # Number of reasoning steps
    num_latents_per_step=1,  # Number of latent tokens per step
    transformer=dict(
        num_tokens=tokenizer.vocab_size,  # Vocabulary size
        dim=512,  # Embedding dimension
        depth=6   # Transformer depth
    )
).to(device)

# Synthesize a single example training data
def create_synthetic_data():
    question = "What is 2 + 3?"
    steps = ["Step 1: Add 2 and 3.", "Step 2: The result is 5."]
    answer = "5"

    # Combine into training stages
    training_stages = []
    for i in range(len(steps) + 1):
        input_sequence = (
            [question] + ["<bot>"] + steps[:i] + ["<eot>"]
        )
        target_sequence = steps[i:] + [answer]

        input_tokens = tokenizer(
            " ".join(input_sequence),
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).input_ids.squeeze(0)
        target_tokens = tokenizer(
            " ".join(target_sequence),
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).input_ids.squeeze(0)

        training_stages.append((input_tokens, target_tokens))

    return training_stages

# Prepare synthetic dataset
synthetic_data = create_synthetic_data()

# DataLoader preparation
def collate_fn(batch):
    """
    Collates a batch of tokenized examples, padding sequences to the same length.
    """
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)
    return inputs_padded.to(device), targets_padded.to(device)

dataloader = DataLoader(synthetic_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 20
stages = len(synthetic_data)  # Number of training stages

for stage in range(stages):
    print(f"Training Stage {stage + 1}/{stages}")
    model.train()

    # Reset optimizer state for each stage
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Prepare inputs for the current stage
            if stage == 0:
                prompts = inputs
                targets = targets
            else:
                prompts = inputs
                targets = targets  # Targets remain unchanged

            # Forward pass
            loss = model(prompts, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 1 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")

    print(f"Stage {stage + 1} completed.\n")

# Save the trained model
torch.save(model.state_dict(), "coconut_model_single_data.pth")

# Example generation
model.eval()
with torch.no_grad():
    example_prompt = "What is 4 + 3?"
    prompt_tokens = tokenizer(example_prompt, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)

    generated_tokens = model.generate(prompt_tokens, max_length=64)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print(f"Generated Answer: {generated_text}")
