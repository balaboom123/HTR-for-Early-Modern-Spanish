from typing import Union, List, Dict, Tuple
import os.path
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from functools import lru_cache
from transformers import PreTrainedTokenizer
from PIL import Image, ImageOps


class OcrDataset(Dataset):
    def __init__(
        self,
        files_dir: str,
        image_size: Tuple[int, int],
        max_output: int,
        tokenizer: PreTrainedTokenizer,
        use_cache: bool = False,
    ) -> None:

        if not hasattr(OcrDataset, "_shared_labels_cache"):
            OcrDataset._shared_labels_cache = {}

        self.image_size = image_size
        self.max_output_length = max_output
        self.files_dir = files_dir
        self.img_files = self.find_files(files_dir, pattern="**/*.jpg")
        self.csv_file = self.find_files(files_dir, pattern="**/*.csv")[0]
        self.sentence_dict = self.load_labels()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

        # Pre-tokenize all labels once
        for img_file in self.img_files:
            # base_name = os.path.relpath(img_file, start="data")
            if img_file not in OcrDataset._shared_labels_cache:
                processed_label = self.process_label(img_file)
                OcrDataset._shared_labels_cache[img_file] = processed_label

        del self.sentence_dict, self.tokenizer

        if use_cache:
            self.__getitem__ = lru_cache()(self.__getitem__)

    def find_files(self, directory: str, pattern: str) -> List[str]:
        return glob(os.path.join(directory, pattern), recursive=True)

    def process_label(self, sentence_name: str) -> torch.Tensor:
        sentence = self.sentence_dict.get(sentence_name, "")

        if pd.isna(sentence):
            sentence = ""
        else:
            sentence = str(sentence)

        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            max_length=self.max_output_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )

        return encoding["input_ids"].squeeze()

    def resize_with_padding(self, image: Image.Image) -> Image.Image:
        """
        Args:
            image: PIL.Image
        Returns:
            resized_padded_image: PIL.Image of size target_size
        """
        target_h, target_w = self.image_size
        original_w, original_h = image.size

        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h

        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        image = ImageOps.expand(
            image, (pad_left, pad_top, pad_right, pad_bottom), fill=0
        )

        return image

    def get_features(self, image_path: str) -> torch.Tensor:
        """
        Load an image, resize it, and return it as a NumPy array.

        Args:
            image_path (str): Path to the image.

        Returns:
            np.ndarray: Image array with shape (C, H, W), dtype=np.uint8 or float32 if normalized.
        """
        image = Image.open(image_path).convert("L")
        image = self.resize_with_padding(image)
        image_array = np.array(image)

        return torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    def load_labels(self) -> Dict[str, str]:
        data_frame = pd.read_csv(self.csv_file, delimiter=",", on_bad_lines="skip")
        df = data_frame[["IMAGE_NAME", "SENTENCE"]]

        df["IMAGE_NAME"] = df["IMAGE_NAME"].apply(
            lambda x: os.path.join(self.files_dir, x)
        )

        sentence_dict = pd.Series(df.SENTENCE.values, index=df.IMAGE_NAME).to_dict()

        return sentence_dict

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_file_path = self.img_files[idx]
        features_tensor = self.get_features(feature_file_path)

        # Remove root directory (e.g., "data/")
        # image_name = os.path.relpath(feature_file_path, start="data")
        label_tensor = OcrDataset._shared_labels_cache[feature_file_path]

        return features_tensor, label_tensor
