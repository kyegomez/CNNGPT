import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_layers,
        kernel_size,
        hidden_dim,
        max_seq_len,
    ):
        super(CNNLanguageModel, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional Encoding (learnable)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embedding_dim)
        )

        # Convolutional Layers
        self.conv_layers = nn.ModuleList()

        for layer in range(num_layers):
            dilation = 2**layer  # Exponentially increasing dilation
            padding = (
                kernel_size - 1
            ) * dilation  # Calculate padding for causal convolutions
            self.conv_layers.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            in_channels=embedding_dim,
                            out_channels=hidden_dim * 2,  # For GLU
                            kernel_size=kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        "layer_norm": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        # Output Layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, sequence_length]
        """
        batch_size, seq_len = x.size()

        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Add positional encoding
        x = (
            x + self.positional_encoding[:, :seq_len, :]
        )  # [batch_size, seq_len, embedding_dim]

        # Transpose for convolutional layers
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]

        # Convolutional Blocks
        for conv_block in self.conv_layers:
            residual = x  # Residual connection
            conv = conv_block["conv"]
            layer_norm = conv_block["layer_norm"]

            # Convolution
            out = conv(
                x
            )  # [batch_size, hidden_dim * 2, seq_len + padding]

            # Remove padding on the right
            padding = conv.padding[0]
            if padding > 0:
                out = out[:, :, :-padding]

            # GLU Activation
            out = F.glu(
                out, dim=1
            )  # [batch_size, hidden_dim, seq_len]

            # Layer Normalization
            out = layer_norm(out.transpose(1, 2)).transpose(1, 2)

            # Residual Connection
            x = out + residual  # [batch_size, embedding_dim, seq_len]

        # Transpose back
        x = x.transpose(1, 2)  # [batch_size, seq_len, embedding_dim]

        # Output Layer
        logits = self.fc_out(x)  # [batch_size, seq_len, vocab_size]

        return logits
