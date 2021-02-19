from abc import ABC

import torch
import torch.nn as nn
import numpy as np
import arglist
from utils import Dense2Conv, Flatten


minimap_channel = 11
screen_channel = 27

nonspatial_dim = arglist.NUM_ACTIONS

minimap_conv = nn.Sequential(nn.Conv2d(minimap_channel, 16, 5, 1, 2),
                             nn.ReLU(),
                             nn.Conv2d(16, 32, 3, 1, 1),
                             nn.ReLU())

screen_conv = nn.Sequential(nn.Conv2d(screen_channel, 16, 5, 1, 2),
                            nn.ReLU(),
                            nn.Conv2d(16, 32, 3, 1, 1),
                            nn.ReLU())

nonspatial_dense = nn.Sequential(nn.Linear(nonspatial_dim, 32),
                                 nn.ReLU(),
                                 Dense2Conv())

class Actor_FullyConv(nn.Module):

    def __init__(self):
        super(Actor_FullyConv, self).__init__()

        # shape of the minimap : (batch_size, minimap_channel, 64, 64)
        # 64 x 64 -> 64 x 64 -> 64 x 64
        # output of minimap_conv : (batch_size, 32, 64, 64)
        #
        # shape of screen : (batch_size, screen_channel, 84, 84)
        # 84 x 84 -> 84 x 84 -> 84 x 84
        # output of screen_conv : (batch_size, 32, 84, 84)
        #
        # output of convolution : (size - kernel_size + 2 x Padding)/Stride + 1

        # when using FullyConv, padding as "same"

        self.minimap_conv = minimap_conv
        self.screen_conv = screen_conv
        self.nonspatial_dense = nonspatial_dense

        self.latent_dense = nn.Sequential(nn.Conv2d(32 * 3, 64, 3, 1, 1),
                                          nn.ReLU())

        # then concatenate the outputs of (minimap, screen, non_spatial)
        self.action_dense = nn.Sequential(nn.Conv2d(64, 1, 1),
                                          nn.ReLU(),
                                          Flatten(),
                                          nn.Linear(arglist.FEAT2DSIZE * arglist.FEAT2DSIZE, arglist.NUM_ACTIONS))

        self.screen1_conv = nn.Conv2d(64, 1, 1)
        self.screen2_conv = nn.Conv2d(64, 1, 1)

    def forward(self, minimap, screen, non_spatial):
        """
        input:
            minimap : in shape (batch_size, minimap_channle, H, W)
            screen : im shape (batch_size, screen_channel, H, W)
            non_spatial : in shape (batch_size, nonspatial_dim)
            valid_actions : masks for valid actions in the batch
                            in shape (batch_size, actions)
        return:
            values :  batch of (batch, 1)
            base_action_prob : (batch_size, n_actions)
            prob_dict : a dictionary,
                        for each arg args : (batch, size) (a probability distribution)
        """

        minimap_out = self.minimap_conv(minimap)
        screen_out = self.screen_conv(screen)
        info_out = self.nonspatial_dense(non_spatial)

        state_h = self.latent_dense(torch.cat([minimap_out, screen_out, info_out], dim=1))
        prob_categorical = self.action_dense(state_h)

        prob_screen1 = self.screen1_conv(state_h)
        prob_screen2 = self.screen2_conv(state_h)

        return {'categorical': prob_categorical, 'screen1': prob_screen1, 'screen2': prob_screen2}


class Critic_FullyConv(nn.Module):

    def __init__(self):
        super(Critic_FullyConv, self).__init__()

        # shape of the minimap : (batch_size, minimap_channel, 64, 64)
        # 64 x 64 -> 64 x 64 -> 64 x 64
        # output of minimap_conv : (batch_size, 32, 64, 64)
        #
        # shape of screen : (batch_size, screen_channel, 84, 84)
        # 84 x 84 -> 84 x 84 -> 84 x 84
        # output of screen_conv : (batch_size, 32, 84, 84)
        #
        # output of convolution : (size - kernel_size + 2 x Padding)/Stride + 1

        self.minimap_conv = minimap_conv
        self.screen_conv = screen_conv
        self.nonspatial_dense = nonspatial_dense

        # spatial actions
        self.conv_action = nn.ModuleList([nn.Sequential(nn.Conv2d(2, 16, 5, 1, 2),
                                         nn.ReLU(),
                                         nn.Conv2d(16, 32, 3, 1, 1),
                                         nn.ReLU()) for _ in range(2)])
        # non-spatial actions
        self.action_dense = nn.ModuleList([nn.Sequential(nn.Linear(arglist.NUM_ACTIONS, 32),
                                          nn.ReLU(),
                                          Dense2Conv()) for _ in range(2)])

        self.latent_dense = nn.ModuleList([nn.Sequential(nn.Conv2d(32 * 5, 64, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 1, 1),
                                          nn.ReLU(),
                                          Flatten()) for _ in range(2)])

        self.value_dense = nn.ModuleList([nn.Linear(arglist.FEAT2DSIZE * arglist.FEAT2DSIZE, 1) for _ in range(2)])

    def Q_value(self, minimap, screen, non_spatial, action):
        minimap = self.minimap_conv(minimap)
        screen = self.screen_conv(screen)
        info = self.nonspatial_dense(non_spatial)

        act_categorical = action['categorical']
        act_screen1 = action['screen1']
        act_screen2 = action['screen2']

        act_spatial = torch.cat([act_screen1, act_screen2], dim=1)
        act_spatial = self.conv_action[0](act_spatial)
        act_non = self.action_dense[0](act_categorical)

        latent = torch.cat([minimap, screen, info, act_spatial, act_non], dim=1)
        latent = self.latent_dense[0](latent)

        q = self.value_dense[0](latent)   

        return q

    def forward(self, minimap, screen, non_spatial, action):
        """
        input:
            minimap : in shape (batch_size, minimap_channle, H, W)
            screen : im shape (batch_size, screen_channel, H, W)
            non_spatial : in shape (batch_size, nonspatial_dim)
            valid_actions : masks for valid actions in the batch
                            in shape (batch_size, actions)
        return:
            values :  batch of (batch, 1)
            base_action_prob : (batch_size, n_actions)
            prob_dict : a dictionary,
                        for each arg args : (batch, size) (a probability distribution)
        """
        minimap_out = self.minimap_conv(minimap)
        screen_out = self.screen_conv(screen)
        info_out = self.nonspatial_dense(non_spatial)

        act_categorical = action['categorical']
        act_screen1 = action['screen1']
        act_screen2 = action['screen2']

        act_spatial = torch.cat([act_screen1, act_screen2], dim=1)

        a_spatial1, a_spatial2 = [self.conv_action[_](act_spatial) for _ in range(2)]
        a_non1, a_non2 = [self.action_dense[_](act_categorical) for _ in range(2)]

        latent1 = torch.cat([minimap_out, screen_out, info_out, a_spatial1, a_non1], dim=1)
        latent2 = torch.cat([minimap_out, screen_out, info_out, a_spatial2, a_non2], dim=1)

        latent1, latent2 = self.latent_dense[0](latent1), self.latent_dense[1](latent2)

        q1, q2 = self.value_dense[0](latent1), self.value_dense[1](latent2)
        return q1, q2