import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            limit=None,
            max_audio_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs

        self._assert_index_is_valid(index)
        index = self._filter_records(index, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self.max_audio_length = max_audio_length
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_samples"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            
            if audio_tensor_wave.shape[1] > self.max_audio_length:
                audio_tensor_wave = audio_tensor_wave[:, :self.max_audio_length]
            
            return audio_tensor_wave

    @staticmethod
    def _filter_records(
            index: list, limit: int
    ) -> list:
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_samples" in entry, (
                "Each dataset item should include field 'audio_samples'"
                " - duration of audio (in samples)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
