import torch
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import PreTrainedModel
from safetensors.torch import load_file

from models.model.hierarchical_embedding import HierarchicalEmbedding
from models.model.hierarchical_embedding_config import HierarchicalEmbeddingConfig
from models.model.hierarchical_t5_config import HierarchicalT5Config


class HierarchicalT5EncoderOnly(PreTrainedModel):
	config_class = HierarchicalT5Config
	base_model_prefix = "hierarchical_T5_encoder"

	def __init__(self, config: HierarchicalT5Config):
		super().__init__(config)

		self.pad_token_id = config.pad_token_id
		self.frame_patch_size = config.frame_patch_size

		# Initialize custom encoder configuration
		encoder_config = HierarchicalEmbeddingConfig(
			image_size=config.image_size,
			image_patch_size=config.image_patch_size,
			max_frames=config.max_frames,
			frame_patch_size=config.frame_patch_size,
			window_size=config.window_size,
			dim=config.dim,
			ffn_hidden_ratio=config.ffn_hidden_ratio,
			n_heads=config.n_heads,
			drop_prob=config.drop_prob,
			vit_block=config.vit_block,
		)

		# Initialize the custom encoder
		self.custom_embed = HierarchicalEmbedding(config=encoder_config)

		# Load T5 model but we'll only use the encoder
		print(f"Loading T5 encoder from: {config.model_name}")
		t5_config = T5Config.from_pretrained(
			pretrained_model_name_or_path=config.model_name,
			dropout_rate=config.drop_prob,
		)

		# Load the full T5 model temporarily
		full_t5_model = T5ForConditionalGeneration.from_pretrained(
			pretrained_model_name_or_path=config.model_name,
			config=t5_config,
		)

		# Extract just the encoder
		self.t5_encoder = full_t5_model.encoder

		# Ensure the encoder and custom embedder have compatible hidden sizes
		if (self.custom_embed.config.dim * 2) != t5_config.d_model:
			raise ValueError(
				f"HierarchicalEmbedding d_model {encoder_config.dim}*2 does not match T5 encoder d_model {t5_config.d_model}."
			)

		# Add a projection layer if needed for your downstream task
		# For example, if you need to project encoder outputs to a specific dimension
		self.output_projection = nn.Linear(t5_config.d_model, config.output_dim) if hasattr(config,
		                                                                                    'output_dim') else None

		# Load pretrained weights if provided
		prefix_map = {
			"custom_embed.": "",
			"t5_encoder.": "",
		}

		if config.pretrained_path is not None:
			weights = load_file(config.pretrained_path + "/model.safetensors")
			weights = self.rename_keys(weights, prefix_map)
			self.custom_embed.load_state_dict(weights, strict=False)
			self.t5_encoder.load_state_dict(weights, strict=False)

	def forward(
			self,
			input_ids: torch.LongTensor,
			attention_mask: torch.FloatTensor = None,
			output_hidden_states: bool = False,
			output_attentions: bool = False,
	):
		# Create attention mask if not provided
		if attention_mask is None:
			attention_mask = self.make_src_mask(input_ids)

		# Generate embeddings using custom embedder
		embeddings = self.custom_embed(
			input_ids,
			attention_mask=attention_mask,
		)

		# Get encoder outputs only
		encoder_outputs = self.t5_encoder(
			inputs_embeds=embeddings,
			attention_mask=attention_mask,
			output_hidden_states=output_hidden_states,
			output_attentions=output_attentions,
			return_dict=True,
		)

		# Get the encoder's last hidden state
		encoder_last_hidden_state = encoder_outputs.last_hidden_state

		# Apply output projection if defined
		if self.output_projection is not None:
			encoder_last_hidden_state = self.output_projection(encoder_last_hidden_state)

		# Return a dictionary with all the encoder outputs
		result = {
			"last_hidden_state": encoder_last_hidden_state,
		}

		if output_hidden_states:
			result["hidden_states"] = encoder_outputs.hidden_states

		if output_attentions:
			result["attentions"] = encoder_outputs.attentions

		return result

	def make_src_mask(self, src):
		batch_size, _, frames, _, width = src.shape

		non_pad_elements = (src != 0).sum(dim=-1)  # Shape: (B, 1, max_frame, 1)
		src_mask = (
			(non_pad_elements > 0).squeeze(-1).squeeze(1)
		)  # Shape: (B, max_frame)

		src_mask = src_mask.view(
			batch_size, frames // (4 * self.frame_patch_size), 4 * self.frame_patch_size
		)
		src_mask = src_mask.max(dim=-1)[0]

		return src_mask

	def rename_keys(self, weights, prefix_map):
		new_weights = {}
		for key, value in weights.items():
			for old_prefix, new_prefix in prefix_map.items():
				if key.startswith(old_prefix):
					new_weights[new_prefix + key[len(old_prefix):]] = value
					break
			else:
				new_weights[key] = value
		return new_weights