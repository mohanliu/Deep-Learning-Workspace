# --------------------------------------------------------
# Position embedding utils
# Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------

import numpy as np
import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, omega_denom=10000):
    """
    grid_size: int of the grid height or width (usually, the height is equal to the width)
        grid_size = int(self.patch_embed.num_patches ** .5),
        e.g. grid_size = 14 (for 16x16 patches of 224x224 image)

    return:
        pos_embed:
            [grid_size * grid_size, embed_dim] (w/o cls_token)
            or
            [1 + grid_size * grid_size, embed_dim] (w/ cls_token)
    """
    # initailize the position indices (integter values but converted to float)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    # `meshigrid` returns a 2D array of repeated vectors
    # grid_x, grid_y = np.meshgrid(grid_w, grid_h)
    #   where
    #     - grid_x = np.stack([grid_h] * len(grid_w))
    #     - grid_y = np.stack([grid_w] * len(grid_h))
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first

    # stack the grid_x and grid_y together
    grid = np.stack(grid, axis=0)  # shape: (2, grid_size, grid_size)

    # unsqueeze the first dimension to make it 4D: (2, 1, grid_size, grid_size)
    # equivalent to: grid = np.expand_dims(grid, axis=1)
    # Note: this line seems optional, more testing is needed.
    grid = grid.reshape([2, 1, grid_size, grid_size])

    # get the position embedding
    #   shape: (grid_size * grid_size, D), (196, D) for 16x16 patches of 224x224 image
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, omega_denom)

    # add cls_token if needed
    if cls_token:
        pos_embed = np.concatenate(
            [
                np.zeros([1, embed_dim]),  # all zeros for cls_token, shape: (1, D)
                pos_embed,  # shape: (grid_size * grid_size, D)
            ],
            axis=0,
        )  # shape: (grid_size * grid_size + 1, D)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, omega_denom=10000):
    # embeded_dim must be multiple of 4
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h and the other half to encode grid_w
    # each dimension, we are flatten the grid to a 1D vector
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], omega_denom
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], omega_denom
    )  # (H*W, D/2)

    # concatenate the embedding of grid_h and grid_w
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, omega_denom=10000):
    """
    embed_dim: output dimension for each position (d_model, has to be even number)
    pos: a list of positions to be encoded: size (M,) (sequnce length)
    out: (M, D), same dimension as patch_embed

    p(t)_i = sin(omega_k * t)  if i = 2k,
    p(t)_i = cos(omega_k * t)  if i = 2k+1,
        where omega_k = 1/((10000) ** (2k/D)), k = 1, ..., D/2

    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)  # k = 1, ..., D/2
    omega /= embed_dim / 2.0  # 2k/D
    omega = 1.0 / omega_denom ** omega  # shape: (D/2,), 1/10000 ** (2k/D)

    # flatten the position indices
    # - stays the same for 1d pos vector
    # - flattened to (M,) for 2d pos vector (1, grid_size, grid_size): where M = 1*H*W
    pos = pos.reshape(-1)  # (M,)

    # get the phase in radians, used for sin and cos
    out = np.einsum(
        "m,d->md", pos, omega
    )  # (M, D/2), outer product (using einstein sum)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


if __name__ == "__main__":
    output = get_1d_sincos_pos_embed_from_grid(8, np.arange(16))
    print(output.shape)
