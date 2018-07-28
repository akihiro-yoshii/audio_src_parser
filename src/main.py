from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audio_util import save_wav
from model import Sequence, Sequence2
from data_loader import AudioDataset
import argparse

import time


def save_checkpoint(state, is_best, filename='out/model/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("update best score")
        torch.save(state, 'out/model/best.pth.tar')


def train(args):
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # build the model
    # seq = Sequence()
    seq = Sequence2()
    seq.float()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(seq.parameters(), lr=0.001)
    #begin to train

    dataset = AudioDataset(args.recipe)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    last_loss = float("inf")
    for i in range(args.epochs):
        print('==== STEP{} ===='.format(i))
        start_time = time.time()
        # def closure():
        loss_list = np.empty(0)
        for batch_id, (_, _, _, input, target) in enumerate(dataloader):
            print("Epoch{}_batch{}".format(i, batch_id))
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)

            loss.backward()
            loss_list = np.append(loss_list, [loss.item()])
            # return loss
            optimizer.step()

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

    # build the model
    seq = Sequence2()
    seq.float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.001)

    checkpoint = torch.load('out/model/best.pth.tar')
    seq.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    dataset = AudioDataset(args.recipe)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_id, (mixed_list, wave1_list, wave2_list, input, target) in enumerate(dataloader):
        with torch.no_grad():
            pred = seq(input)
            loss = criterion(pred, target)
            print('test loss:', loss.item())
            y = (pred.detach().numpy() * 65536).astype('int16')

            save_wav(wave1_list[0].numpy().astype(np.int16), './out/wav/sample{}_wave0.wav'.format(batch_id))
            save_wav(wave2_list[0].numpy().astype(np.int16), './out/wav/sample{}_wave1.wav'.format(batch_id))
            save_wav(mixed_list[0].numpy().astype(np.int16), './out/wav/sample{}_mixed.wav'.format(batch_id))
            save_wav(y[0][0], './out/wav/sample{}_0.wav'.format(batch_id))
            save_wav(y[0][1], './out/wav/sample{}_1.wav'.format(batch_id))

def get_arguments():
    parser = argparse.ArgumentParser(description='Audio parser')
    subparsers = parser.add_subparsers(help='sub command help', title='subcommands')

    train_parser = subparsers.add_parser('train', help='train help')
    train_parser.add_argument('--recipe', type=str, required=True,
                              help='name of recipe file')
    train_parser.add_argument('--epochs', type=int, default=15, metavar='N',
                              help='number of epochs (default: 20)')
    train_parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                              help='number of batch_size (default: 6)')
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
