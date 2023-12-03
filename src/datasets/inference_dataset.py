import os

import torchaudio

from src.base.base_dataset import BaseDataset


class InferenceDataset(BaseDataset):
    def __init__(self, path_to_audios: str, *args, **kwargs):
        assert os.path.exists(path_to_audios)
        index = self._get_index(path_to_audios)
        super().__init__(index, *args, **kwargs)
        
    def _get_index(self, path_to_audios):
        index = []
        audios = os.listdir(path_to_audios)
        for audio_path in audios:
            audio_path = os.path.join(path_to_audios, audio_path)
            t_info = torchaudio.info(str(audio_path))
            index.append({
                "path": audio_path,
                "audio_samples": t_info.num_frames
            })
            
        return index
