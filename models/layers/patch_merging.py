import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    """
    Args:
        dim: The dimension of the input data.
        merge_sizes(List[int]): The size of the merge operation.
        norm_layer: The normalization layer to be used (default is nn.LayerNorm).
    """

    def __init__(self, dim, merge_sizes=[4, 8, 16], norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.merge_sizes = merge_sizes

        # Reduce frame dimension and double feature dim
        self.norms = nn.ModuleList([
            norm_layer(ms * dim) for ms in merge_sizes
        ])
        self.reductions = nn.ModuleList([
            nn.Linear(ms * dim, 2 * dim, bias=False) for ms in merge_sizes
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor shaped as  (B, F, C) where B is batch size, F is number of frames, and C is number of channel.

        """
        B, F, C = x.shape
        outputs = []

        for i, merge_size in enumerate(self.merge_sizes):
            assert F % merge_size == 0, f"F={F} must be divisible by merge_size={merge_size}"
            x_merge = x.view(B, F // merge_size, merge_size * C)

            # Apply normalization and reduction
            x_merge = self.norms[i](x_merge)
            x_merge = self.reductions[i](x_merge)  # (B, F//merge_size, 2*C)
            outputs.append(x_merge)

        # combine the patch
        x = torch.cat(outputs, dim=1)

        return x
