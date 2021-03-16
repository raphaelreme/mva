import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, sizes=(3, 6, 3), activation=nn.ReLU):
        super().__init__()

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)
