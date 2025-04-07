import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from conf import *
from models.model.hierarchical_t5 import HierarchicalT5
from models.model.hierarchical_t5_config import HierarchicalT5Config
from models.model.hierarchical_embedding_config import HierarchicalEmbeddingConfig
from utils.OcrDataset import OcrDataset
from utils.epoch_timer import epoch_time

# Register custom config
AutoConfig.register("hierarchical_T5", HierarchicalEmbeddingConfig)

# Ensure device is set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model configuration and model
model_config = HierarchicalT5Config.from_pretrained(pretrained_path)
model_config.pretrained_path = pretrained_path
model = HierarchicalT5(config=model_config)

# Load the best model checkpoint (if available)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
dec_voc_size = tokenizer.vocab_size
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
print(f"Vocabulary size of the model: {dec_voc_size}")

# initialize dataset
test_dataset = OcrDataset(
    files_dir=test_dir,
    image_size=model_config.image_size,
    max_output=model_config.max_output,
    tokenizer=tokenizer,
)

test_iter = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

def evaluate(model, iterator, tokenizer, beam_size=5):
    """
    Evaluates the model on the given dataset iterator.

    Args:
        model (nn.Module): The trained model to evaluate.
        iterator (DataLoader): DataLoader for the evaluation dataset.
        tokenizer: Tokenizer for converting indices to text.
        beam_size (int): Beam size for generation; if 0, only acc is computed.

    Returns:
        tuple: Average acc and word accuracy.
    """
    model.eval()
    total_loss = []
    total_correct_words, total_words = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            src = x.to(device, non_blocking=True)
            trg = y.to(device, non_blocking=True)

            # Compute acc
            loss = model(input_ids=src, labels=trg).loss
            total_loss.append(loss.item())

            if beam_size != 0:
                output = model.generate(input_ids=src, beam_size=beam_size)
            else:
                continue

            trg_words = tokenizer.batch_decode(y, skip_special_tokens=True)
            output_words = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

            # Calculate word-level accuracy for each batch
            for pred, ref in zip(output_words, trg_words):
                pred_words = pred.strip(" ").split(" ")
                ref_words = ref.strip(" ").split(" ")
                correct = sum(p == r for p, r in zip(pred_words, ref_words))
                total_correct_words += correct
                total_words += len(ref_words)

            if i % 25 == 0:
                print(
                    f"Batch {i}: {round((i / len(iterator)) * 100, 2)}% complete\n"
                    f"Predicted: {output_words[0]}\nTarget: {trg_words[0]}"
                )

    avg_loss = sum(total_loss) / len(total_loss) if total_loss else 0.0
    word_accuracy = total_correct_words / total_words if total_words > 0 else 0.0

    return avg_loss, word_accuracy

if __name__ == "__main__":
    start_time = time.time()
    eval_loss, word_acc = evaluate(model, test_iter, tokenizer, beam_size=5)
    end_time = time.time()
    elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)

    print("Evaluation Results:")
    print(f"Average Loss: {eval_loss:.3f}")
    print(f"Word Accuracy: {word_acc:.4f}")
    print(f"Time Taken: {elapsed_mins}m {elapsed_secs}s")
