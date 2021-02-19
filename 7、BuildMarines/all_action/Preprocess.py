from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

import torch
import torch.nn as nn

# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def preprocess_minimap(minimap, embedding_func):
    layers = []
    assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
    for i in range(len(features.MINIMAP_FEATURES)):
        if features.MINIMAP_FEATURES[i].type == features.FeatureType.CATEGORICAL:
            layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
            for j in range(features.MINIMAP_FEATURES[i].scale):
                indy, indx = (minimap[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
        else:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)

    return np.concatenate(layers, axis=0)


def preprocess_screen(screen, embedding_func):
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)
    for i in range(len(features.SCREEN_FEATURES)):
        if features.SCREEN_FEATURES[i].type == features.FeatureType.CATEGORICAL:
            layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
            for j in range(features.SCREEN_FEATURES[i].scale):
                indy, indx = (screen[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
        else:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
    return np.concatenate(layers, axis=0)


def minimap_channel():
    n_channels = 0
    for i in range(len(features.MINIMAP_FEATURES)):
        if features.MINIMAP_FEATURES[i].type == features.FeatureType.CATEGORICAL:
            n_channels += 10
        else:
            n_channels += 1
    return n_channels


def screen_channel():
    n_channels = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if features.SCREEN_FEATURES[i].type == features.FeatureType.CATEGORICAL:
            n_channels += 10
        else:
            n_channels += 1
    return n_channels


def get_feature_embed(feat_list):
    embed = {}
    for feat in feat_list:
        if feat.type == features.FeatureType.CATEGORICAL:
            embed[feat.name] = nn.Embedding(feat.scale, 10)
    embed = nn.ModuleDict(embed)
    return embed
