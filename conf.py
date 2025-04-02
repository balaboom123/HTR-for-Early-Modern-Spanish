# load pre-trained model or not
load_pretrained = False
pretrained_path = "result/save/model-62-7.5961"
tokenizer_type = "hf"

# model parameter setting
# input shape -> (batch_size, channels, height, width)
batch_size = 16
image_size = (768, 48)  # height, width
image_patch_size = (24, 24)
channels = 1
dim = 256  # half of the dimension of the t5
vit_block = 4
model_name = "google/t5-v1_1-small" #
encoder_layers = 6  # deprecated
decoder_layers = 6  # deprecated
n_heads = 4  # half of the n_heads of the t5
ffn_hidden_ratio = 4
drop_prob = 0.2
max_output = 32

# optimizer parameter setting
init_lr = 1e-3
betas = (0.9, 0.95)
warmup = 10
epoch = 100
clip = 1.0
weight_decay = 1e-1
inf = float("inf")

# loss
label_smoothing = 0.2

# lr_scheduler
T_max = epoch - warmup
end_lr = 1e-5

# file path setting
train_dir = "data/cc100"
test_dir = "data/source"

# Prepare the information as a formatted string
info = f"""
# load pre-trained model or not
load_pretrained = {load_pretrained}
pretrained_path = {pretrained_path}

# Model Parameter Settings
batch_size = {batch_size}
image_size = {image_size}
image_patch_size = {image_patch_size}
max_frames = {channels}
dim = {dim}
model_name = {model_name}
encoder_layers = {encoder_layers}
decoder_layers = {decoder_layers}
n_heads = {n_heads}
ffn_hidden_ratio = {ffn_hidden_ratio}
drop_prob = {drop_prob}
max_output = {max_output}

# loss
label_smoothing = {label_smoothing}

# lr_scheduler
T_max = {T_max}
end_lr = {end_lr}

# Optimizer Parameter Settings
init_lr = {init_lr}
betas = {betas}
warmup = {warmup}
epoch = {epoch}
clip = {clip}
weight_decay = {weight_decay}
inf = {inf}
"""
