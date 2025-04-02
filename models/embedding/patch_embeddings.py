from torch import nn
import torch

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    """Convert input to tuple pair if not already."""
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbedding(nn.Module):
    """3D Patch embedding for video input with positional encoding.

    Args:
        image_size (int | tuple): Input image size
        image_patch_size (int | tuple): Size of each image patch
        max_frames (int): Maximum number of video frames
        frame_patch_size (int): Number of frames per temporal patch
        dim (int): Output embedding dimension
        channels (int, optional): Number of input channels. Default: 1
        drop_prob (float, optional): Dropout rate. Default: 0.1
    """

    def __init__(
        self,
        image_size=(48, 768),
        image_patch_size=(24, 24),
        dim=256,
        channels=1,
        drop_prob=0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b c h w (p1 p2)",

                # "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
            ),
            Rearrange("b c h w p -> b c (w h) p"),
            Rearrange("b c n p -> b n (p c)"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        nums_patch = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, nums_patch, dim))
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """Convert video to patch embeddings with positional encoding.

        Args:
            x (Tensor): Video tensor (B, C, F, H, W)

        Returns:
            Tensor: Patch embeddings (B, N, D)
        """
        patch_emb = self.to_patch_embedding(x)
        pos_emb = self.pos_embedding[:, : patch_emb.size(1), :]
        return self.drop_out(patch_emb + pos_emb)
