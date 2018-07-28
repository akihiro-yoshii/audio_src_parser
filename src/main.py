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

def save_checkpoint(state, is_best, filename='out/model/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("update best score")
        torch.save(state, 'out/model/best.pth.tar')

def recipe2data(recipe_path):
    list = list_audio(recipe_path)
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

    return mixed_list, wave1_list, wave2_list, input, target


def train(args):
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    mixed_list, wave1_list, wave2_list, input, target = recipe2data(args.recipe)

    # build the model
    # seq = Sequence()
    seq = Sequence2()
    seq.float()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.8, max_iter=args.cycles)
    optimizer = optim.Adam(seq.parameters(), lr=0.001)
    #begin to train
    last_loss = float("inf")

    for i in range(args.steps):
        print('==== STEP{} ===='.format(i))
        start_time = time.time()
        # def closure():
        loss_list = np.empty(0)
        for j in range(args.cycles):
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            loss_list = np.append(loss_list, [loss.item()])
            # return loss
            optimizer.step()
        # optimizer.step(closure)
        end_time = time.time()
        print("[Step Info]")
        print("Time: {}".format(end_time - start_time))
        print("Loss: {}".format(np.mean(loss_list)))
        save_checkpoint({
            'state_dict': seq.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=last_loss > np.mean(loss_list))
        if last_loss > np.mean(loss_list):
            last_loss = np.mean(loss_list)
        print()

def eval(args):
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    mixed_list, wave1_list, wave2_list, input, target = recipe2data(args.recipe)

    # build the model
    seq = Sequence2()
    seq.float()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.8, max_iter=args.cycles)
    optimizer = optim.Adam(seq.parameters(), lr=0.001)

    checkpoint = torch.load('out/model/checkpoint.pth.tar')
    seq.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    with torch.no_grad():
        pred = seq(input)
        loss = criterion(pred, target)
        print('test loss:', loss.item())
        y = (pred.detach().numpy() * 65536).astype('int16')

        for i in range(input.size(0)):
            save_wav(wave1_list[i].astype(np.int16), './out/wav/sample{}_wave0.wav'.format(i))
            save_wav(wave2_list[i].astype(np.int16), './out/wav/sample{}_wave1.wav'.format(i))
            save_wav(mixed_list[i].astype(np.int16), './out/wav/sample{}_mixed.wav'.format(i))
            save_wav(y[i][0], './out/wav/sample{}_0.wav'.format(i))
            save_wav(y[i][1], './out/wav/sample{}_1.wav'.format(i))

def get_arguments():
    parser = argparse.ArgumentParser(description='Audio parser')
    subparsers = parser.add_subparsers(help='sub command help', title='subcommands')

    train_parser = subparsers.add_parser('train', help='train help')
    train_parser.add_argument('--recipe', type=str, required=True,
                              help='name of recipe file')
    train_parser.add_argument('--steps', type=int, default=15, metavar='N',
                              help='number of steps to train (default: 15)')
    train_parser.add_argument('--cycles', type=int, default=20, metavar='N',
                              help='number of cycles for LBFGS optimizer (default: 20)')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval', help='eval help')
    eval_parser.add_argument('--recipe', type=str, required=True,
                              help='name of recipe file')
    eval_parser.set_defaults(func=eval)


    args = parser.parse_args()
    args.func(args)

def main():
    args = get_arguments()


if __name__ == '__main__':
    main()
    # eval()
