import torch
import torch.nn as nn
import arglist
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, -1)


class Dense2Conv(nn.Module):
    def forward(self, input):
        out = torch.repeat_interleave(input, arglist.FEAT2DSIZE * arglist.FEAT2DSIZE)
        out = out.view(-1, input.shape[1], arglist.FEAT2DSIZE, arglist.FEAT2DSIZE)
        return out
