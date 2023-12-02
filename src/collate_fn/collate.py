from typing import List

import torch.nn as nn

from src.collate_fn.mels import MelSpectrogram, MelSpectrogramConfig


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.T for item in batch]
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def get_collate():
    config = MelSpectrogramConfig
    mel_spec = MelSpectrogram(config)
    
    def collate_fn(dataset_items: List[dict]):
        """
        Collate and pad fields in dataset items
        """
        audio = [item['audio'] for item in dataset_items]
        audio = pad_sequence(audio)
        spectrograms = mel_spec(audio)

        return {
            'spectrogram': spectrograms,
            'audio': audio
        }

    return collate_fn
