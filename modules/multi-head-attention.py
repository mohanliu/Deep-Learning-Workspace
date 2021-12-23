import torch.nn as nn
from torch import Tensor
from typing import Optional, Any


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.

    Args:
        embed_dim (int): Total dimension of the model (required).
        num_heads (int): Parallel attention heads (default=8).
        qkv_bias (bool): Whether to add bias in attention part of the model
            (default=False).
        attention_dropout (float): A dropout layer on attention outputs
            (default= 0.0).
        projection_dropout (float): A dropout layer on the final output
            (default= 0.0).
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        qkv_bias=False,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): the input sequence to the attention layer
                (required).
        Shape:
            - x: :math:`(B, N, C)` where `B` is the batch size, `N` is the
             sequence length, and `C` is the embedding dimension.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.projection_dropout(x)
        return x


if __name__ == "__main__":
    m = MultiheadAttention(32)
