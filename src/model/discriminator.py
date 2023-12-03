from typing import List

import torch.nn as nn

from src.model.discriminator_blocks import SubMPD, SubMSD


class MPD(nn.Module):
    def __init__(
        self,
        periods: List[int],
        channels: List[int],
        kernel_sizes: int,
        strides: int,
        paddings: int
    ):
        super().__init__()
        self.net = nn.ModuleList([
            SubMPD(p, channels, kernel_sizes, strides, paddings) for p in periods
        ])
    
    def _forward_impl(self, x):
        outputs, feature_maps = [], []
        for discriminator in self.net:
            output, feature_map = discriminator(x)
            outputs.append(output)
            feature_maps.extend(feature_map)

        return outputs, feature_maps
    
    def forward(self, audio, audio_gen, **batch):
        outputs, feature_maps = self._forward_impl(audio)
        outputs_gen, feature_maps_gen = self._forward_impl(audio_gen)
        
        return {
            "mpd_logits": outputs,
            "mpd_feature_maps": feature_maps,
            "mpd_logits_gen": outputs_gen,
            "mpd_feature_maps_gen": feature_maps_gen
        }


class MSD(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        groups_list: List[int],
        paddings: List[int]
    ):
        super().__init__()
        self.net = nn.ModuleList([
            SubMSD(channels, kernel_sizes, strides, groups_list, paddings, use_spectral_norm=False),
            SubMSD(channels, kernel_sizes, strides, groups_list, paddings, use_spectral_norm=False),
            SubMSD(channels, kernel_sizes, strides, groups_list, paddings, use_spectral_norm=False),
        ])

    def _forward_impl(self, x):
        outputs, feature_maps = [], []
        for discriminator in self.net:
            output, feature_map = discriminator(x)
            outputs.append(output)
            feature_maps.extend(feature_map)

        return outputs, feature_maps
    
    def forward(self, audio, audio_gen, **batch):
        outputs, feature_maps = self._forward_impl(audio)
        outputs_gen, feature_maps_gen = self._forward_impl(audio_gen)
        
        return {
            "msd_logits": outputs,
            "msd_feature_maps": feature_maps,
            "msd_logits_gen": outputs_gen,
            "msd_feature_maps_gen": feature_maps_gen
        }
