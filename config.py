import torch

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数设置
batch_size = 128
max_length = 512
d_model = 512
n_layers = 6
n_heads = 8
d_ff = 2048
dropout = 0.1

# 训练设置
epoch = 50
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 100
clip = 1.0
weight_decay = 5e-4
inf = float('inf')


