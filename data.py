from conf import *

# from util.data_loader import DataLoader
from utils.OcrDataset import OcrDataset
import sentencepiece as spm

from torch.utils.data import DataLoader, ConcatDataset
import evaluate
from transformers import AutoConfig, AutoTokenizer
from models.model.hierarchical_embedding_config import HierarchicalEmbeddingConfig

# Register custom config
AutoConfig.register("hierarchical_T5", HierarchicalEmbeddingConfig)

tokenizer = AutoTokenizer.from_pretrained(model_name)  # Helsinki-NLP/opus-mt-de-en
dec_voc_size = tokenizer.vocab_size
print(f"Vocabulary size of the model: {dec_voc_size}")

pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
print(f"Pad token id: {pad_token_id}")
print(f"EOS token id: {eos_token_id}")

# initialize dataset
test_dataset = OcrDataset(
    files_dir=test_dir,
    image_size=image_size,
    max_output=max_output,
    tokenizer=tokenizer,
)
train_dataset = OcrDataset(
    files_dir=train_dir,
    image_size=image_size,
    max_output=max_output,
    tokenizer=tokenizer,
)

train_iter = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=40,
)
test_iter = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
