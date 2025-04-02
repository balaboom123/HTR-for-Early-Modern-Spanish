import os, shutil
import torch

# List to store the best model file names and their validation losses
best_models = []


def save_best_models(model, loss, step, save_dir="result", max_models=3):
    """Saves and maintains top N models based on BLEU scores.

    Args:
        model: PyTorch model to save
        bleu: Model's BLEU score
        step: Training step
        save_dir: Save directory. Default: "result"
        max_models: Maximum models to keep. Default: 3
    """
    global best_models

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_filename = f"model-{step}-{loss:.4f}"
    model_path = os.path.join(save_dir, model_filename)

    model.save_pretrained(model_path, config=model.config)

    best_models.append((model_filename, loss))
    best_models = sorted(
        best_models, key=lambda x: x[1], reverse=True
    )  # Sort by bleu in descending order

    if len(best_models) > max_models:
        worst_model = best_models.pop(0)  # Remove the first (worst) model
        worst_model_path = os.path.join(save_dir, worst_model[0])
        if os.path.exists(worst_model_path):
            if os.path.isdir(worst_model_path):
                shutil.rmtree(worst_model_path)  # Remove folder and its contents
            else:
                os.remove(worst_model_path)  # Remove file


def get_best_models(save_dir="result"):
    """Returns path of best performing model.

    Args:
        save_dir: Models directory. Default: "result"

    Returns:
        str: Path to best model
    """
    global best_models

    best_models = sorted(
        best_models, key=lambda x: x[1], reverse=True
    )  # Sort by bleu in descending order
    best_model_path = os.path.join(save_dir, best_models[-1][0])
    return best_model_path
