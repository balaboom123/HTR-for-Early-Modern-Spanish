from torch import nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.mlp import Mlp


class EncoderLayer(nn.Module):
    """Transformer encoder layer with windowed self-attention and FFN blocks.

    Args:
        dim (int): Feature dimension
        ffn_hidden_ratio (int): FFN hidden layer size multiplier
        n_head (int): Number of attention heads
        drop_prob (float): Dropout rate
    """

    def __init__(self, dim, ffn_hidden_ratio, n_head, drop_prob):
        super().__init__()
        self.window_attention = MultiHeadAttention(
            dim=dim, num_heads=n_head, drop_prob=drop_prob
        )
        self.norm1 = nn.LayerNorm(normalized_shape=dim)

        self.mlp = Mlp(
            in_features=dim, hidden_features=ffn_hidden_ratio * dim, drop=drop_prob
        )
        self.norm2 = nn.LayerNorm(normalized_shape=dim)

    def forward(self, x):
        """Process input through windowed attention and FFN blocks.

        Args:
            x (Tensor): Input features (B, N, D)
            window_size (int): Size of local attention window
            src_mask (Tensor): Attention mask (B, N, N)

        Returns:
            Tensor: Processed features (B, N, D)
        """
        _, N, _ = x.shape

        # compute self attention
        _x = x
        x = self.norm1(x)
        x = self.window_attention(q=x, k=x, v=x)
        x = x + _x

        # positionwise feed forward network
        _x = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + _x

        return x
