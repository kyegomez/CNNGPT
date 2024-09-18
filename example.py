import torch
from cnngpt.main import CNNLanguageModel

# Hyperparameters
vocab_size = 5000
embedding_dim = 256
num_layers = 4
kernel_size = 3
hidden_dim = 256
max_seq_len = 100

# Initialize model
model = CNNLanguageModel(
    vocab_size,
    embedding_dim,
    num_layers,
    kernel_size,
    hidden_dim,
    max_seq_len,
)

# Dummy input (batch_size=32, sequence_length=50)
x = torch.randint(
    0, vocab_size, (32, 50)
)  # Random integers as token IDs

# Forward pass
logits = model(x)  # [batch_size, seq_len, vocab_size]

# Output shape
print("Logits shape:", logits.shape)
