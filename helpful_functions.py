import torch
import torch.nn as nn
import torch.nn.init as init

def get_device():
    if torch.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
