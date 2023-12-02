import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, **batch):
        raise NotImplementedError()
