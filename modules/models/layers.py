# based on Ross Wightman's PyTorch Image Models

import torch.nn as nn
from torch import Tensor
from typing import Optional, Any


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding.
    based on: timm/models/layers/patch_embed.py

    Args:
        in_channels (int): Number of input channels (default=3).
        image_size (int): Size of the input image (default=224).
        patch_size (int): Size of the input image patches (default=16).
        dim_embed (int): Dimension of the patch embedding (default=768).
        embed_norm (nn.Module): Patch embedding normalization layer
            (default=None).
        flatten (bool): Whether to flatten the patch embedding (default=True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        dim_embed: int = 768,
        embed_norm: Optional[Any] = None,
        flatten: bool = True,
    ) -> None:
        super(PatchEmbed, self).__init__()

        self.image_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.project_patch = nn.Conv2d(
            in_channels, dim_embed, kernel_size=patch_size, stride=patch_size
        )
        self.norm = embed_norm(dim_embed) if embed_norm else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert (
            H == self.image_size[0] and W == self.image_size[1]
        ), "[X] Image size does not match the model configuration"
        x = self.project_patch(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


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
    ) -> None:
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor) -> Tensor:
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


class MLP(nn.Module):
    """The MLP/Feed-Forward block used in a Transformer's encoder layer followed
        by an attnetion block.
    Args:
        in_features (int): The number of input features (required).
        hidden_features (int): The number of hidden features (default=None).
            If None, hidden_features=in_features.
        out_features (int): The number of output features (default=None).
            If None, out_features=in_features.
        activation (nn.Module): The activation layer (default: nn.GELU).
        dropout (float): The dropout rate (default: 0.0).
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=nn.GELU,
        dropout=0.0,
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
