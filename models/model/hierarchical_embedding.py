from torch import nn
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.patch_embeddings import PatchEmbedding
from models.model.hierarchical_embedding_config import HierarchicalEmbeddingConfig


# not to use this class
# split the embedding and encoder layer
class HierarchicalEmbedding(PreTrainedModel):
    """
    HierarchicalEmbedding class for custom hierarchical encoder model.

    Args:
        config (HierarchicalEmbeddingConfig): Configuration for the encoder model.

    Attributes:
        config_class (HierarchicalEmbeddingConfig): Configuration class for the encoder model.
        base_model_prefix (str): Prefix for the base model of the encoder.

    Methods:
        __init__(self, config: HierarchicalEncoderConfig): Constructor to initialize the encoder.
        forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None, create_attn: bool = True, **kwargs) -> BaseModelOutput: Forward pass for the encoder.
        make_src_mask(self, src): Generate source mask to avoid attending to padding tokens.

    Returns:
        BaseModelOutput: Contains the last hidden state and optionally attentions.
    """
    config_class = HierarchicalEmbeddingConfig
    base_model_prefix = "hierarchical_T5"

    def __init__(self, config: HierarchicalEmbeddingConfig):
        super().__init__(config)

        self.emb = PatchEmbedding(
            image_size=config.image_size,
            image_patch_size=config.image_patch_size,
            dim=config.dim,
            drop_prob=config.drop_prob,
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        **kwargs
    ) -> BaseModelOutput:
        """
        Forward pass for the custom encoder.

        Args:
            input_ids (torch.LongTensor): Token indices representing the input sequence.
            attention_mask (torch.FloatTensor, optional): Mask to avoid attending to padding tokens.

        Returns:
            BaseModelOutput: Contains the last hidden state and optionally attentions.
        """
        x = self.emb(input_ids)

        return x
