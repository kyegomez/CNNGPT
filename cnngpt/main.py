import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class CNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        kernel_size: int,
        hidden_dim: int,
        max_seq_len: int,
    ) -> None:
        """
        Initializes the CNN-based language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            num_layers (int): Number of convolutional layers.
            kernel_size (int): Size of the convolutional kernel.
            hidden_dim (int): Number of hidden units for each convolutional layer.
            max_seq_len (int): Maximum sequence length.
        """
        super(CNNLanguageModel, self).__init__()

        logger.info("Initializing CNNLanguageModel...")

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        logger.debug(
            f"Embedding layer initialized with vocab_size={vocab_size} and embedding_dim={embedding_dim}"
        )

        # Positional Encoding (learnable)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embedding_dim)
        )
        logger.debug(
            f"Positional encoding initialized with max_seq_len={max_seq_len}"
        )

        # Convolutional Layers
        self.conv_layers = nn.ModuleList()
        logger.debug(
            f"Creating {num_layers} convolutional layers with exponentially increasing dilation."
        )

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
            logger.debug(
                f"Layer {layer}: Conv1D with dilation={dilation}, padding={padding}"
            )

        # Output Layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        logger.debug(
            f"Output fully connected layer initialized with hidden_dim={hidden_dim} and vocab_size={vocab_size}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, sequence_length, vocab_size].
        """
        batch_size, seq_len = x.size()
        logger.info(
            f"Forward pass with batch_size={batch_size} and seq_len={seq_len}"
        )

        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        logger.debug("Embedding layer applied.")

        # Add positional encoding
        x = (
            x + self.positional_encoding[:, :seq_len, :]
        )  # [batch_size, seq_len, embedding_dim]
        logger.debug("Positional encoding added.")

        # Transpose for convolutional layers
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        logger.debug("Tensor transposed for convolutional layers.")

        # Convolutional Blocks
        for i, conv_block in enumerate(self.conv_layers):
            residual = x  # Residual connection
            conv = conv_block["conv"]
            layer_norm = conv_block["layer_norm"]

            # Convolution
            out = conv(
                x
            )  # [batch_size, hidden_dim * 2, seq_len + padding]
            logger.debug(
                f"Layer {i}: Convolution applied with output shape {out.shape}."
            )

            # Remove padding on the right
            padding = conv.padding[0]
            if padding > 0:
                out = out[:, :, :-padding]
                logger.debug(
                    f"Layer {i}: Padding removed with final shape {out.shape}."
                )

            # GLU Activation
            out = F.glu(
                out, dim=1
            )  # [batch_size, hidden_dim, seq_len]
            logger.debug(
                f"Layer {i}: GLU activation applied with output shape {out.shape}."
            )

            # Layer Normalization
            out = layer_norm(out.transpose(1, 2)).transpose(1, 2)
            logger.debug(f"Layer {i}: Layer normalization applied.")

            # Residual Connection
            x = out + residual  # [batch_size, embedding_dim, seq_len]
            logger.debug(f"Layer {i}: Residual connection added.")

        # Transpose back
        x = x.transpose(1, 2)  # [batch_size, seq_len, embedding_dim]
        logger.debug("Tensor transposed back for output layer.")

        # Output Layer
        logits = self.fc_out(x)  # [batch_size, seq_len, vocab_size]
        logger.info(
            f"Output layer applied with logits shape {logits.shape}."
        )

        return logits
