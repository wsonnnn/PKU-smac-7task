import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, -1)


class Dense2Conv(nn.Module):
    def forward(self, input):
        out = torch.repeat_interleave(input, 64 * 64)
        out = out.view(-1, input.shape[1], 64, 64)
        return out


def mask_redundant_actions(valid_actions, name):
    masker = np.zeros_like(valid_actions)
    if name == "FindAndDefeatZerglings":
        masker[331] = valid_actions[331]
        masker[12] = valid_actions[12]
        masker[1] = valid_actions[1]
        masker[7] = valid_actions[7]
    else:
        masker = valid_actions
    
    return masker