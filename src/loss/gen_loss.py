import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    def __init__(self, alpha: float = 2, beta: float = 45):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        
    def forward(
        self,
        mpd_feature_maps,
        mpd_feature_maps_gen,
        msd_feature_maps,
        msd_feature_maps_gen,
        spectrogram,
        spectrogram_gen,
        mpd_logits_gen,
        msd_logits_gen,
        **batch
    ):
        mel_loss = self.l1_loss(spectrogram, spectrogram_gen)
        
        feature_loss = 0
        for feature_maps, feature_maps_gen in zip(mpd_feature_maps, mpd_feature_maps_gen):
            feature_loss = feature_loss + self.l1_loss(feature_maps, feature_maps_gen)
        for feature_maps, feature_maps_gen in zip(msd_feature_maps, msd_feature_maps_gen):
            feature_loss = feature_loss + self.l1_loss(feature_maps, feature_maps_gen)
        
        adv_loss = 0
        for logit in mpd_logits_gen:
            adv_loss = adv_loss + torch.mean((logit - 1) ** 2)
        for logit in msd_logits_gen:
            adv_loss = adv_loss + torch.mean((logit - 1) ** 2)
            
        full_loss = adv_loss + self.alpha * feature_loss + self.beta * mel_loss
        
        return full_loss, adv_loss, mel_loss, feature_loss
