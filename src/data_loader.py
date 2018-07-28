import torch
from torch.utils.data import Dataset

import numpy as np

from audio_util import get_mix_audio


class AudioDataset(Dataset):
    def __init__(self, recipe_path):
        with open(recipe_path, 'r') as f:
            strings = f.readlines()

        list = []
        for string in strings:
            if string.find('#') != 0:
                columns = string.rstrip().split(' ')
                path1 = columns[0] + columns[1]
                path2 = columns[0] + columns[2]

                list.append([path1, path2])

        self.audio_pairs = list

    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):
        audio_pair = self.audio_pairs[index]
        mixed, wave1, wave2 = get_mix_audio(audio_pair[0], audio_pair[1])

        input = torch.from_numpy(mixed / 65536).float()
        target = torch.from_numpy(np.stack((wave1 / (65536 * 2), wave2 / (65536 * 2)), axis=0)).float()

        return mixed, wave1, wave2, input, target
