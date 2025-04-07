# load pre-trained model or not
load_pretrained = False
pretrained_path = "result/best"

# model parameter setting
# input shape -> (batch_size, channels, height, width)
batch_size = 16
image_size = (32, 128)  # height, width
image_patch_size = (8, 8)
channels = 1
dim = 512
vit_block = 4
model_name = "t5-small"
n_heads = 4  # deprecated
ffn_hidden_ratio = 4
drop_prob = 0.2
max_output = 16

# optimizer parameter setting
init_lr = 1e-3
betas = (0.9, 0.999)
warmup = 10
epoch = 100
clip = 1.0
weight_decay = 1e-4
inf = float("inf")

# acc
label_smoothing = 0.2

# lr_scheduler
T_max = epoch - warmup
end_lr = 1e-5

# file path setting
train_dir = "data/cc100"
test_dir = "data/source_vocab"

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
