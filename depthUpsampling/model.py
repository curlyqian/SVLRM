import torch
import torch.nn as nn
from math import sqrt


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class SVLRM(nn.Module):
    def __init__(self):
        super(SVLRM, self).__init__()
        self.medium_layer = self.make_layer(Conv_ReLU_Block, 10)
        self.input = nn.Conv2d(
            in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.output = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        #权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.medium_layer(out)
        out = self.output(out)
        return out