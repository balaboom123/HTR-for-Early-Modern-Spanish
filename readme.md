# Handwritten OCR for Early Modern Spanish

This repository contains an implementation of an Optical Character Recognition (OCR) system specifically designed for early modern Spanish handwritten texts. The system uses a hierarchical T5 model to recognize and transcribe characters from images.


## Data
### Spanish Text
The wordlist comes from the [OpenSLR Spanish Word list](https://openslr.org/21/).

### Images Gerenation
The dataset class in `utils/OcrDataset.py` is designed for loading data generated from SynthTIGER.
To train with default dataset, you may get the data from [SynthTIGER](https://github.com/clovaai/synthtiger) or you should make a custom dataset class.

## Train
please set up the hyperparameters for model in `conf.py` then use `train.py`

constructing...

## Test
1. download the weights from [Google Drive](https://drive.google.com/drive/folders/1pEAr8v9swyyCxX8qleVAV3FQV1jEMdZx?usp=sharing). 
2. make sure the **pretrained_path** in `conf.py` is match to the folder path.
3. make sure the **test_dir** in `conf.py` is set to the folder where the test data are stored. default is the test data i used in the result.
4. run `test.py`

## Experiment
### Result
| Metric        | Value  |
|---------------|--------|
| Train Loss    | 1.91   |
| Test Loss     | 6.90   |
| Word Accuracy | 67.05% |

### Model Specification
 - total parameters = 60,573,824 parameters
 - input size = (32, 128) # height, width
 - max output tokens = 16
 - lr scheduling : LinearLR

### Configuration
 - batch_size = 128
 - image_patch_size = (8, 8)
 - channels = 1  # gray scale
 - dim = 512
 - ffn_hidden_ratio = 4


 - init_lr = 1e-3
 - end_lr = 1e-7
 - betas = (0.9, 0.999)
 - warmup = 5
 - epoch = 50
 - clip = 1.0
 - weight_decay = 0
 - drop_prob = 0.1
 - label_smoothing = 0.1


## License

This project is licensed under the [LICENSE](LICENSE) file.
