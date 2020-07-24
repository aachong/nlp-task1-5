
fix_length = 600
max_vocab_size = 5000
batch_size = 64

embedding_size = 128
dropout = 0.5
label_size = 10

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
