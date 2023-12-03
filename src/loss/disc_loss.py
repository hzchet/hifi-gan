import torch
import torch.nn as nn



class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mpd_logits,  msd_logits, mpd_logits_gen, msd_logits_gen, **batch):
        total_disc_loss = 0
        for mpd_real, mpd_gen in zip(mpd_logits, mpd_logits_gen):
            total_disc_loss += torch.mean((mpd_real - 1) ** 2) + torch.mean(mpd_gen ** 2)
        for msd_real, msd_gen in zip(msd_logits, msd_logits_gen):
            total_disc_loss += torch.mean((msd_real - 1) ** 2 + torch.mean(msd_gen ** 2))
            
        return total_disc_loss
