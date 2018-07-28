import torch
import torch.nn as nn

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input):
        outputs = []
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            output = self.linear(h_t)
            output = torch.cat([input_t - output, output], dim=1)
            outputs += [output]
        outputs = torch.stack(outputs, 2)
        return outputs

class Sequence2(nn.Module):
    def __init__(self):
        super(Sequence2, self).__init__()

        def Down(ic, oc):
            return torch.nn.Sequential(
                nn.Conv1d(ic, oc, 3, padding=1),
                nn.Conv1d(oc, oc, 3, padding=1),
                nn.Conv1d(oc, oc, 3, padding=1),
            )

        def Up(ic, oc):
            return torch.nn.Sequential(
                nn.Conv1d(ic, ic, 3, padding=1),
                nn.Conv1d(ic, ic, 3, padding=1),
                nn.Conv1d(ic, oc, 3, padding=1),
            )

        self.down1 = Down(1, 16)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.down2 = Down(16, 32)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.down3 = Down(32, 64)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.down4 = Down(64, 128)

        self.up4 = Up(128, 64)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.up3 = Up(64, 32)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = Up(32, 16)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = Up(16, 1)


    def forward(self, input):
        input = torch.unsqueeze(input, 1)
        out = self.down1(input)
        conv1 = out
        out = self.pool1(out)
        out = self.down2(out)
        conv2 = out
        out = self.pool2(out)
        out = self.down3(out)
        conv3 = out
        out = self.pool3(out)
        out = self.down4(out)

        out = self.up4(out)
        out = self.upsample3(out)
        out = self.up3(out + conv3)
        out = self.upsample2(out)
        out = self.up2(out + conv2)
        out = self.upsample1(out)
        out = self.up1(out + conv1)

        base = input - out
        out = torch.cat([base, out], dim=1)

        return out
