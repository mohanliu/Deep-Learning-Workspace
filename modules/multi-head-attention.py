import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Callable, List, Optional, Tuple

"""
Pytorch Official Implementation:

https://github.com/pytorch/pytorch/blob/d35fc409ad84c1a837e7e07ffe3f4e4942538e50/torch/nn/modules/activation.py#L862
https://github.com/pytorch/pytorch/blob/d35fc409ad84c1a837e7e07ffe3f4e4942538e50/torch/nn/functional.py#L5067

"""


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Official scaled dot product attention:
    https://github.com/pytorch/pytorch/blob/d35fc409ad84c1a837e7e07ffe3f4e4942538e50/torch/nn/functional.py#L4974
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


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

        self.qkv = nn.Linear(
            embed_dim, embed_dim * 3, bias=qkv_bias
        )  # map input embedding into QKV vectors (x3 dimension)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(
            embed_dim, embed_dim
        )  # a linear layer after concat all attention heads
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
            self.qkv(x)  # map input to QKV vectors
            .reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )  # reshape channels into multiple heads: 3C --> 3 x h x C/h
            .permute(2, 0, 3, 1, 4)  # final order: qkv, B, h, N, C/h
        )
        q, k, v = qkv.unbind(0)  # q,k,v: B, h, N, C/h

        attn = (q @ k.transpose(-2, -1)) * self.scale  # QK^T/sqrt(d)
        attn = attn.softmax(dim=-1)  # global softmax: softmax(QK^T/sqrt(d))
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # softmax(QK^T/sqrt(d))*v
        x = self.proj(x)  # a linear layer for multi-head attention
        x = self.projection_dropout(x)
        return x


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    m = MultiheadAttention(32)
    console.rule("Model summary")
    console.print(m)
    console.rule("Weights")
    for k, v in m.named_parameters():
        console.print("{}: {}".format(k, str(v.shape)))
