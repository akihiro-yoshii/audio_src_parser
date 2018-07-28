import torch
from torch.utils.data import Dataset

import numpy as np

from audio_util import get_mix_audio


class AudioDataset(Dataset):
    def __init__(self, recipe_path):

        list = []

        with open("./recipe/others_test.txt", 'r') as f:
            others_strings = f.readlines()

        with open("./recipe/flute_test.txt", 'r') as f:
            flute_strings = f.readlines()

        for other in others_strings:
            for flute in flute_strings:
                path1 = "./data/nsynth-train/audio/" + other.rstrip()
                path2 = "./data/nsynth-train/audio/" + flute.rstrip()

                list.append([path1, path2])

        for pair in list:
            print(pair)

        self.audio_pairs = list

    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):
        audio_pair = self.audio_pairs[index]
        mixed, wave1, wave2 = get_mix_audio(audio_pair[0], audio_pair[1])

        input = torch.from_numpy(mixed / 65536).float()
        target = torch.from_numpy(np.stack((wave1 / (65536 * 2), wave2 / (65536 * 2)), axis=0)).float()

        return mixed, wave1, wave2, input, target
