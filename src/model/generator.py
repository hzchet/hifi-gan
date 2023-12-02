from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: List[int]
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList([weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding='same')) for d in dilations
        ])
        
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = x + conv_block(F.leaky_relu(x, 0.1))

        return x
    
    def remove_weight_norm(self):
        for block in self.conv_blocks:
            remove_weight_norm(block)


class MRF(nn.Module):
    """
    Multi Receptive Field Fusion
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int],
        dilation_list: List[int]
    ):
        super().__init__()
        res_blocks = []
        for kernel_size in kernel_sizes:
            res_blocks.append(ResBlock(channels, kernel_size, dilation_list))
        self.res_blocks = nn.ModuleList(res_blocks)
        
    def forward(self, x):
        res_stream = None
        for res_block in self.res_blocks:
            if res_stream is None:
                res_stream = res_block(x)
            else:
                res_stream = res_stream + res_block(x)
        
        return res_stream / len(self.res_blocks)
    
    def remove_weight_norm(self):
        for block in self.res_blocks:
            block.remove_weight_norm()


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_kernel_sizes: List[int],
        upsample_channels: List[int],
        upsample_strides: List[int],
        mrf_kernel_sizes: List[int],
        mrf_dilation_list: List[int]
    ):
        super().__init__()
        self.input_conv = weight_norm(nn.Conv1d(in_channels, upsample_channels[0], kernel_size=7,
                                                padding='same'))
        
        upsample_blocks, mrf_blocks = [], []
        for in_channels, out_channels, kernel_size, stride in zip(
            upsample_channels[:-1], upsample_channels[1:], upsample_kernel_sizes, upsample_strides
        ):
            upsample_blocks.append(nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, 
                padding=(kernel_size - stride) // 2
            ))
            mrf_blocks.append(MRF(out_channels, mrf_kernel_sizes, mrf_dilation_list))
            
        self.upsample_blocks = nn.ModuleList(upsample_blocks)
        self.mrf_blocks = nn.ModuleList(mrf_blocks)
        
        self.output_conv = weight_norm(nn.Conv1d(upsample_channels[-1], 1, kernel_size=7, 
                                                 padding='same'))

    def forward(self, x):
        x = self.input_conv(x)
        
        for upsample_block, mrf_block in zip(self.upsample_blocks, self.mrf_blocks):
            x = F.leaky_relu(x, 0.1)
            x = upsample_block(x)
            x = mrf_block(x)
        
        x = F.leaky_relu(x, 0.1)
        x = self.output_conv(x)
        
        return torch.tanh(x)

    def remove_weight_norm(self):
        for block in self.upsample_blocks:
            remove_weight_norm(block)
        for block in self.mrf_blocks:
            block.remove_weight_norm()

        remove_weight_norm(self.input_conv)
        remove_weight_norm(self.output_conv)
