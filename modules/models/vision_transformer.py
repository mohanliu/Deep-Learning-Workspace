# based on Ross Wightman's PyTorch Image Models

import copy
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from .initializers import lecun_normal_, trunc_normal_
from .layers import MLP, MultiheadAttention, PatchEmbed


class VisionTransformerEncoderLayer(nn.Module):
    """VisionTransformerEncoderLayer is made up of self-attn and mlp blocks.

    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the MultiHeadAttention block
            (required).
        attention_dropout (float): The dropout value for attention outputs in
            the MultiHeadAttention block (default=0.0).
        qkv_bias (bool): Whether to add bias in attention part of the model
            (default=False).
        projection_dropout: The dropout value for the final output of the
            attention block (default=0.0).
        dim_mlp (int): The dimension of the mlp block (default=3072).
            Also known as MLP size in the original paper.
        mlp_dropout (float): the dropout value for the mlp block (default=0.0).
        mlp_activation (nn.Module): The activation function to use in the
            mlp block (default=nn.GELU).
        layer_norm_eps (float): Epsilon for the normalization layer
            (default=1e-6).
        norm_first (bool): if ``True``, layer norm is done prior to attention
            and mlp operations, respectivaly. Otherwise it's done after
            (default=True).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        attention_dropout: float = 0.0,
        qkv_bias: bool = False,
        projection_dropout: float = 0.0,
        dim_mlp: int = 3072,
        mlp_dropout: float = 0.0,
        mlp_activation: Union[str, Callable[[Tensor], Tensor]] = nn.GELU,
        layer_norm_eps: float = 1e-6,
        norm_first: bool = True,
    ) -> None:
        super(VisionTransformerEncoderLayer, self).__init__()

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)

        if isinstance(mlp_activation, str):
            mlp_activation = _get_activation_fn(mlp_activation)
        else:
            mlp_activation = mlp_activation

        self.self_attn = MultiheadAttention(
            d_model,
            num_heads=nhead,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=dim_mlp,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

    def forward(
        self,
        embedding: Tensor,
    ) -> Tensor:

        """Pass the input through the encoder layer.

        Args:
            embedding (torch.Tensor): the sequence to the encoder layer
                (required).
        """

        x = embedding
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn(x))
            x = self.norm2(x + self.mlp(x))

        return x


class VisionTransformerEncoder(nn.Module):
    """VisionTransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer (VisionTransformerEncoder): An instance of the
            VisionTransformerEncoderLayer() class.
        num_layers (int): The number of sub-encoder-layers in the encoder
            (required).
        norm (nn.Module): The normalization layer to use for the output of
            encoder (default=LayerNorm()).
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(VisionTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        embedding: Tensor,
    ) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            embedding (torch.Tensor): the sequence of patch embedding to the
                encoder (required).

        Returns:
            torch.Tensor: the output of the encoder.

        Shape:
            - embedding: :math:`(B, num_patches, d_model)`
            - output: :math:`(B, num_patches, d_model)`
        """
        output = embedding

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class VisionTransformer(nn.Module):
    """A vision transformer model. User is able to modify the attributes as
    needed. The architecture is based on the paper "An Image is Worth 16x16
    Words: Transformers for Image Recognition at Scale". Alexey Dosovitskiy,
    Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, et al.
    2021. Users can build a vision transformer encoder with the following
    parameters:

    Args:
        in_channels (int): Number of input channels (default=3).
        image_size (int): Size of the input image (default=224).
        patch_size (int): Size of the input image patches (default=16).
        embed_dropout (float): Dropout probability for the input embedding
            (default=0.0).
        num_classes (int): Number of classes in the dataset (default=1000).
        d_model (int): The constant latent vector size through all of encoder
            layers (default=768). Also known as Hidden size D in the original
            paper.
        num_encoder_layers (int): Number of layers in the encoder (default=12).
        nhead (int): The number of heads in the multiheadattention models
            (default=12).
        attention_dropout (float): The dropout value for attention outputs in
            encoder's MultiHeadAttention blocks (default=0.0).
        qkv_bias (bool): Whether to add bias in attention part of the model
            (default=True).
        projection_dropout (float): The dropout value for the final output of
            the attention block (default=0.0).
        dim_mlp (int): The dimension of the mlp blocks in the encoder
            (default=3072) Also known as {MLP size} in the original paper.
        mlp_dropout (float): The dropout value for the encoder's mlp blocks
            (default=0.0).
        mlp_activation (str or callable): Activation function to use for the
            encoder's mlp blocks (default=nn.GELU).
        encoder_norm (bool): Whether to normalize the final output of the
            encoder (default=True).
        pool (str): Encoder's output pooling method. Can be either "MEAN" or
            "CLS" (default=CLS).
        custom_encoder (nn.Module): Custom encoder to use instead of the
            default vision transformer encoder (default=None).
        custom_pre_logit (nn.Module): Custom pre_logit layer to use for the
            vision transformer. The pre_logit layer will project the pooled
            output of the encoder to a new desired dimension (rep_size) before
            calculating logits.
        custom_patch_embedding (nn.Module): Custom patch embedding layer to use
            instead of the default vision transformer patch embedding
            (default=None).
        layer_norm_eps (float): Epsilon value for the layer normalization
            layers (default=1e-6).
        encoder_norm_first (bool): If True, then the layer normalization layers
            are applied before the activation function in the encoder
            (default=True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dropout: float = 0.0,
        num_classes: int = 1000,
        d_model: int = 768,
        num_encoder_layers: int = 12,
        nhead: int = 12,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        projection_dropout: float = 0.0,
        dim_mlp: int = 3072,
        mlp_dropout: float = 0.0,
        mlp_activation: Union[str, Callable[[Tensor], Tensor]] = nn.GELU,
        encoder_norm: bool = True,
        pool: str = "CLS",
        layer_norm_eps: float = 1e-6,
        encoder_norm_first: bool = True,
        custom_patch_embedding: Optional[Any] = None,
        custom_encoder: Optional[Any] = None,
        custom_pre_logit: Optional[Any] = None,
    ) -> None:
        super(VisionTransformer, self).__init__()

        # Patch embeddings, class tokens, and positional embeddings
        if custom_patch_embedding is not None:
            self.patch_embedding = custom_patch_embedding
        else:
            self.patch_embedding = PatchEmbed(
                in_channels=in_channels,
                image_size=image_size,
                patch_size=patch_size,
                dim_embed=d_model,
            )
        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.embed_dropout = Dropout(embed_dropout)

        # Encoder
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = VisionTransformerEncoderLayer(
                d_model,
                nhead,
                attention_dropout,
                qkv_bias,
                projection_dropout,
                dim_mlp,
                mlp_dropout,
                mlp_activation,
                layer_norm_eps,
                encoder_norm_first,
            )
            encoder_norm = (
                LayerNorm(d_model, eps=layer_norm_eps) if encoder_norm else None
            )
            self.encoder = VisionTransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        # Pooling layer
        # verify_str_arg(pool, arg="pool", valid_values=("CLS", "MEAN"))
        self.pool = pool

        # Representation layer (pre_logit)
        # refer to docs for the details.
        if custom_pre_logit is not None:
            self.pre_logit = custom_pre_logit["layer"]
            rep_size = custom_pre_logit["rep_size"]
        else:
            self.pre_logit = nn.Identity()
            rep_size = d_model

        # Classification head
        self.head = Linear(rep_size, num_classes)

        # Initialize weight parameters
        self._reset_parameters()

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass through the vision transformer and get the output
        features before classification.

        Args:
            x (torch.Tensor): a batch of images to the vision transformer
            (required).

        Returns:
            torch.Tensor: the output features of the vision transformer.

        Shape:
            - output: :math:`(B, num_patches, d_model)`
        """

        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.embed_dropout(x + self.pos_embed)

        x = self.encoder(x)

        x = x.mean(dim=1) if self.pool == "MEAN" else x[:, 0]
        x = self.pre_logit(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the vision transformer and get the output
        classification raw scores.

        Args:
            x (torch.Tensor): a batch of images to the vision transformer
            (required).

        Returns:
            torch.Tensor: the output classification raw scores.

        Shape:
            - x: :math:`(B, C, H, W)`
            - output (x): :math:`(B, num_classes)`
        """

        x = self.forward_features(x)
        x = self.head(x)

        return x

    def _reset_parameters(self):
        """Initiate parameters in the vision transformer model."""

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)


def _init_vit_weights(
    module: nn.Module,
    name: str = "",
    head_bias: float = 0.0,
):
    """ViT weight initialization"""
    if isinstance(module, Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("representation"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    # NOTE: conv is left to pytorch default!
    elif isinstance(module, (LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    # verify_str_arg(activation, arg="activation", valid_values=["RELU", "GELU", "GLU"])
    if activation == "RELU":
        return nn.RELU
    if activation == "GELU":
        return nn.GELU
    if activation == "GLU":
        return nn.GLU
