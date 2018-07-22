from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audio_util import get_mix_audio, save_wav, get_frames, print_wave_info
from model import Sequence, Sequence2
import argparse

import time

def get_arguments():
    parser = argparse.ArgumentParser(description='Audio parser')
    parser.add_argument('--recipe', type=str, required=True,
                        help='name of recipe file')
    parser.add_argument('--steps', type=int, default=15, metavar='N',
                        help='number of steps to train (default: 15)')
    parser.add_argument('--cycles', type=int, default=20, metavar='N',
                        help='number of cycles for LBFGS optimizer (default: 20)')
    return parser.parse_args()

def list_audio(recipe_path):
    with open(recipe_path, 'r') as f:
        strings = f.readlines()

    list = []
    for string in strings:
        if string.find('#') != 0:
            columns = string.rstrip().split(' ')
            path1 = columns[0] + columns[1]
            path2 = columns[0] + columns[2]

            list.append([path1, path2])

    return list


def main():

    args = get_arguments()

    list = list_audio(args.recipe)
    # print(list)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set

    # data preparation
    mixed_list = np.empty((0, 8000), int)
    wave1_list = np.empty((0, 8000), int)
    wave2_list = np.empty((0, 8000), int)
    for pair in list:
        mixed, wave1, wave2 = get_mix_audio(pair[0], pair[1])
        mixed_list = np.append(mixed_list, np.expand_dims(mixed, axis=0), axis=0)
        wave1_list = np.append(wave1_list, np.expand_dims(wave1, axis=0), axis=0)
        wave2_list = np.append(wave2_list, np.expand_dims(wave2, axis=0), axis=0)

    input = torch.from_numpy(mixed_list / 65536).float()
    target = torch.from_numpy(np.stack((wave1_list / (65536 * 2), wave2_list / (65536 * 2)), axis=1)).float()

    # test_input = torch.from_numpy(np.expand_dims(mixed / 65536, axis=0)).float()
    # test_target = torch.from_numpy(np.expand_dims(np.stack((wave1 / 65536, wave2 / 65536)), axis=0)).float()
    test_input = input
    test_target = target

    # build the model
    # seq = Sequence()
    seq = Sequence()
    seq.float()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.8, max_iter=args.cycles)
    optimizer = optim.Adam(seq.parameters(), lr=0.001)
    #begin to train
    for i in range(args.steps):
        print('==== STEP{} ===='.format(i))
        start_time = time.time()
        # def closure():
        for j in range(args.cycles):
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            # return loss
            optimizer.step()
        # optimizer.step(closure)
        end_time = time.time()
        print("[Step Info]")
        print("Time: {}".format(end_time - start_time))

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = seq(test_input)
            loss = criterion(pred, test_target)
            print('test loss:', loss.item())
            y = (pred.detach().numpy() * 65536).astype('int16')

            for i in range(input.size(0)):
                save_wav(wave1_list[i].astype(np.int16), './out/sample{}_wave0.wav'.format(i))
                save_wav(wave2_list[i].astype(np.int16), './out/sample{}_wave1.wav'.format(i))
                save_wav(mixed_list[i].astype(np.int16), './out/sample{}_mixed.wav'.format(i))
                save_wav(y[i][0], './out/sample{}_0.wav'.format(i))
                save_wav(y[i][1], './out/sample{}_1.wav'.format(i))
            print()


if __name__ == '__main__':
    main()
