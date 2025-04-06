import torch
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import PreTrainedModel
from safetensors.torch import load_file

from models.model.hierarchical_embedding import HierarchicalEmbedding
from models.model.hierarchical_embedding_config import HierarchicalEmbeddingConfig
from models.model.hierarchical_t5_config import HierarchicalT5Config


class HierarchicalT5(PreTrainedModel):
    config_class = HierarchicalT5Config
    base_model_prefix = "hierarchical_T5"

    def __init__(self, config: HierarchicalT5Config):
        super().__init__(config)

        self.pad_token_id = config.pad_token_id

        # Initialize custom encoder configuration
        encoder_config = HierarchicalEmbeddingConfig(
            image_size=config.image_size,
            image_patch_size=config.image_patch_size,
            dim=config.dim,
            ffn_hidden_ratio=config.ffn_hidden_ratio,
            n_heads=config.n_heads,
            drop_prob=config.drop_prob,
            vit_block=config.vit_block,
        )

        # Initialize the custom encoder
        self.custom_embed = HierarchicalEmbedding(config=encoder_config)

        # Load Marian Decoder
        print(f"Loading T5 model: {config.model_name}")
        t5_config = T5Config.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            dropout_rate=config.drop_prob,
        )
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            config=t5_config,
        )

        # Example usage before load_state_dict
        prefix_map = {
            "custom_embed.": "",
            "t5.": "",
        }

        if config.pretrained_path != None:
            weights = load_file(config.pretrained_path + "/model.safetensors")
            weights = self.rename_keys(weights, prefix_map)
            self.custom_embed.load_state_dict(weights, strict=False)
            self.t5.load_state_dict(weights, strict=False)
            # print(f"Missing keys: {missing_keys}")
            # print(f"Unexpected keys: {unexpected_keys}")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        decoder_attention_mask: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = self.make_trg_mask(labels)

        # Generate embeddings using custom embedder
        embeddings = self.custom_embed(
            input_ids,
        )

        return self.t5(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,  # for cross attention
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        beam_size: int = 3,
        **generate_kwargs,
    ):

        # Get encoder embeddings
        embeddings = self.custom_embed(
            input_ids,
            attention_mask=attention_mask,
        )

        return self.t5.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            num_beams=beam_size,
            do_sample=False,
            # bad_words_ids=[[6999]],
            max_length=128,  # Limit the output length
            decoder_start_token_id=self.t5.config.decoder_start_token_id,
            eos_token_id=self.t5.config.eos_token_id,
            return_dict_in_generate=True,
            **generate_kwargs,
        )

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_token_id).to(self.device)  # Shape: (B, 512)

        return trg_pad_mask

    def rename_keys(self, weights, prefix_map):
        new_weights = {}
        for key, value in weights.items():
            for old_prefix, new_prefix in prefix_map.items():
                if key.startswith(old_prefix):
                    new_weights[new_prefix + key[len(old_prefix) :]] = value
                    break
            else:
                new_weights[key] = value
        return new_weights
