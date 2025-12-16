import os

import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torchaudio


class Dataset(torch_data.Dataset):
    def __init__(
        self, datadir: str, feats: nn.Module, positive_audio_ids: set[str] = None
    ):
        self._pathes = []
        self._feats = feats
        self._positive_audio_ids = (
            positive_audio_ids if positive_audio_ids is not None else set()
        )

        if os.path.exists(datadir):
            audios = os.listdir(datadir)
            for audio_path in audios:
                if audio_path.endswith(".opus"):
                    self._pathes.append(os.path.join(datadir, audio_path))

    def __getitem__(self, index):
        path = self._pathes[index]
        waveform, sample_rate = torchaudio.load(path)
        assert sample_rate == 48000
        assert waveform.shape[0] == 1
        feats = self._feats(waveform)[0]

        audio_id = audio_id_from_path(path)
        label = 1 if audio_id in self._positive_audio_ids else 0

        return feats, label, path

    def __len__(self) -> int:
        return len(self._pathes)


def audio_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    max_length = max(item[0].shape[1] for item in batch)
    X = torch.zeros((len(batch), batch[0][0].shape[0], max_length))
    for idx, item in enumerate(batch):
        X[idx, :, : item[0].shape[1]] = item[0]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    pathes = [item[2] for item in batch]
    return (X, targets, pathes)
