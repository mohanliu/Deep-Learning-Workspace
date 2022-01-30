import torch
import torch.nn as nn


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
        in_channels=3,
        image_size=224,
        patch_size=16,
        dim_embed=768,
        embed_norm=None,
        flatten=True,
    ):
        super(PatchEmbed, self).__init__()

        self.in_channels = in_channels
        self.image_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # use a convolutional layer to project the image into a patch embedding
        self.proj = nn.Conv2d(
            in_channels, dim_embed, kernel_size=patch_size, stride=patch_size
        )

        self.norm = embed_norm(dim_embed) if embed_norm else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape  # x: B x C x H x W
        assert (
            H == self.image_size[0] and W == self.image_size[1]
        ), "[X] Image size does not match the model configuration"

        x = self.proj(x)  # B x D x H/p x W/p (B, 768, 14, 14)
        if self.flatten:
            x = x.flatten(start_dim=2)  # B x D x (H/p) x (W/p) -> B x D x N
            x = x.transpose(1, 2)  # B x D x N -> B x N x D
        x = self.norm(x)
        return x


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    input_tensor = torch.randn(1, 3, 224, 224)
    console.print("Input tensor shape: {}".format(input_tensor.shape))

    pe = PatchEmbed(in_channels=3, image_size=224, patch_size=16, dim_embed=768)

    output = pe(input_tensor)
    console.print("Output tensor shape: {}".format(output.shape))
