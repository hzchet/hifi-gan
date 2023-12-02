from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class SubMPD(nn.Module):
    def __init__(
        self,
        period: int,
        channels: List[int],
        kernel_sizes: int,
        strides: int,
        paddings: int
    ):
        super().__init__()
        self.period = period
        conv_blocks = []
        for in_channels, out_channels, kernel_size, stride, padding in zip(
            channels[:-1], channels[1:], kernel_sizes, strides, paddings
        ):
            conv_blocks.append(weight_norm(nn.Conv2d(
                in_channels, 
                out_channels, 
                (kernel_size, 1), 
                (stride, 1), 
                padding=(padding, 0)
            )))

        self.conv_blocks = nn.ModuleList(conv_blocks)
    
    def forward(self, x):
        feature_map = []
        B, C, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, (0, self.period - (T % self.period)), mode="reflect")
            T += self.period - T % self.period
        
        x  = x.view(B, C, T // self.period, self.period)
        for i, conv_block in enumerate(self.conv_blocks):
            if i != len(self.conv_blocks) - 1:
                x = F.leaky_relu(conv_block(x), 0.1)
            else:
                x = conv_block(x)
            
            feature_map.append(x)
        
        return torch.flatten(x, 1, -1), feature_map


class SubMSD(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        groups_list: List[int],
        paddings: List[int],
        use_spectral_norm: bool
    ):
        super().__init__()
        conv_blocks = []
        for in_channels, out_channels, kernel_size, stride, groups, padding in zip(
            channels[:-1], channels[1:], kernel_sizes, strides, groups_list, paddings
        ):
            if use_spectral_norm:
                conv_blocks.append(spectral_norm(nn.Conv1d(
                    in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding
                )))
            else:
                conv_blocks.append(weight_norm(nn.Conv1d(
                    in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding
                )))
        self.conv_blocks = nn.ModuleList(conv_blocks)
    
    def forward(self, x):
        feature_map = []
        for i, conv_block in enumerate(self.conv_blocks):
            if i != len(self.conv_blocks) - 1:
                x = F.leaky_relu(conv_block(x), 0.1)
            else:
                x = conv_block(x)
            
            feature_map.append(x)
        
        x = torch.flatten(x, 1 ,-1)
        return x, feature_map
