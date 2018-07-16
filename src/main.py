from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audio_util import get_mix_audio, save_wav, get_frames, print_wave_info
import argparse

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.linear = nn.Linear(51, 2)

    def forward(self, input):
        outputs = []
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 2)
        return outputs

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

    # path1, path2 = list_audio(args.recipe)
    list = list_audio(args.recipe)
    print(list)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    # data = torch.load('traindata.pt')

    mixed, wave1, wave2 = get_mix_audio(list[0][0], list[0][1])
    input = torch.from_numpy(np.expand_dims(mixed / 65536, axis=0)).float()
    target = torch.from_numpy(np.expand_dims(np.stack((wave1 / 65536, wave2 / 65536)), axis=0)).float()
    # target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(np.expand_dims(mixed / 65536, axis=0)).float()
    test_target = torch.from_numpy(np.expand_dims(np.stack((wave1 / 65536, wave2 / 65536)), axis=0)).float()

    # build the model
    seq = Sequence()
    seq.float()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8, max_iter=args.cycles)
    #begin to train
    for i in range(args.steps):
        print('==== STEP{} ===='.format(i))
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            print('[TEST]')
            pred = seq(test_input)
            loss = criterion(pred, test_target)
            print('test loss:', loss.item())
            y = (pred.detach().numpy() * 65536).astype('int16')

            save_wav(y[0][0], './data/pred{}_0.wav'.format(i))
            save_wav(y[0][1], './data/pred{}_1.wav'.format(i))
            print()


if __name__ == '__main__':
    main()
