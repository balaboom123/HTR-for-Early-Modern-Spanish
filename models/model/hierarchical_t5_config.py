from transformers import PretrainedConfig


class HierarchicalT5Config(PretrainedConfig):

    model_type = "hierarchical_T5"

    def __init__(
        self,
        pad_token_id=0,
        eos_token_id=1,
        image_size=(1, 255),
        image_patch_size=(1, 255),
        channels=1,
        dim: int = 256,
        vit_block: int = 2,
        n_heads: int = 4,
        ffn_hidden_ratio: int = 4,
        drop_prob=0.1,
        max_output=128,
        pretrained_path=None,
        dec_voc_size=7000,
        model_name="t5-small",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.image_size = tuple(image_size)
        self.image_patch_size = tuple(image_patch_size)
        self.channels = channels
        self.dim = dim
        self.vit_block = vit_block
        self.n_heads = n_heads
        self.ffn_hidden_ratio = ffn_hidden_ratio
        self.drop_prob = drop_prob
        self.max_output = max_output
        self.pretrained_path = pretrained_path
        self.dec_voc_size = dec_voc_size
        self.model_name = model_name

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.update(
            {
                "pad_token_id": self.pad_token_id,
                "image_size": self.image_size,
                "image_patch_size": self.image_patch_size,
                "channels": self.channels,
                "dim": self.dim,
                "vit_block": self.vit_block,
                "n_heads": self.n_heads,
                "ffn_hidden_ratio": self.ffn_hidden_ratio,
                "drop_prob": self.drop_prob,
                "max_output": self.max_output,
                "pretrained_path": self.pretrained_path,
                "dec_voc_size": self.dec_voc_size,
                "model_name": self.model_name,
            }
        )
        return config_dict
