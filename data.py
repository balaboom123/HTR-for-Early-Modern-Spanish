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

# get_sp(csv_file="dataset/how2sign/train_3D_keypoints_85/how2sign_realigned_train.csv")

if tokenizer_type == "hf":
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Helsinki-NLP/opus-mt-de-en
    dec_voc_size = tokenizer.vocab_size
    print(f"Vocabulary size of the model: {dec_voc_size}")

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    print(f"Pad token id: {pad_token_id}")
    print(f"EOS token id: {eos_token_id}")

elif tokenizer_type == "sp":
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("vocab.model")
    dec_voc_size = tokenizer.get_piece_size()

    print(f"Vocabulary size of the model: {dec_voc_size}")
    pad_token_id = tokenizer.piece_to_id('<pad>')
    eos_token_id = tokenizer.piece_to_id('</s>')
    print(f"Pad token id: {pad_token_id}")
    print(f"EOS token id: {eos_token_id}")

else:
    raise ValueError(f"Invalid tokenizer type: {tokenizer_type}, must be 'hf' or 'sp'")


# initialize dataset
test_dataset = OcrDataset(
    files_dir=test_dir,
    image_size=image_size,
    max_output=max_output,
    tokenizer=tokenizer,
    tokenizer_type=tokenizer_type,
)
train_dataset = OcrDataset(
    files_dir=train_dir,
    image_size=image_size,
    max_output=max_output,
    tokenizer=tokenizer,
    tokenizer_type=tokenizer_type,
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
