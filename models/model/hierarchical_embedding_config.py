from transformers import PretrainedConfig


class HierarchicalEmbeddingConfig(PretrainedConfig):
    """
    HierarchicalEmbeddingConfig class for configuring a hierarchical Marian encoder.

    Args:
        image_size (tuple): Tuple representing the size of the image. Default is (1, 255).
        image_patch_size (tuple): Tuple representing the size of image patches. Default is (1, 255).
        dim (int): Tuple of dimensions. Default is (128, 256, 512).
        ffn_hidden_ratio (int): Ratio used in the feedforward network. Default is 4.
        n_heads (int): Tuple representing the number of heads. Default is (2, 4, 8).
        drop_prob (float): Dropout probability. Default is 0.1.
        vit_block (int): Tuple representing the number of encoder layers. Default is 2.

    Attributes:
        image_size (tuple): Tuple representing the size of the image.
        image_patch_size (tuple): Tuple representing the size of image patches.
        dim (tuple): Tuple of dimensions.
        ffn_hidden_ratio (int): Ratio used in the feedforward network.
        n_heads (tuple): Tuple representing the number of heads.
        drop_prob (float): Dropout probability.
        vit_block (tuple): Tuple representing the number of encoder layers.
    """

    model_type = "hierarchical_T5"

    def __init__(
        self,
        image_size=(1, 255),
        image_patch_size=(1, 255),
        dim: int = 256,
        ffn_hidden_ratio: int = 4,
        n_heads: int = 4,
        drop_prob=0.1,
        vit_block: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_patch_size = image_patch_size
        self.dim = dim
        self.ffn_hidden_ratio = ffn_hidden_ratio
        self.n_heads = n_heads
        self.drop_prob = drop_prob
        self.vit_block = vit_block
