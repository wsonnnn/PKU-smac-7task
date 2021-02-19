import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
sys.path.append("..") 
import utils

from utils import Dense2Conv, Flatten

class FullyConvNet(nn.Module):
    
    def __init__(self, minimap_channel, screen_channel, nonspatial_dim, n_actions):

        super(FullyConvNet, self).__init__()

        # state encoding
        # can switch to average pooling anyhow
        nonspatial_dim = nonspatial_dim + 16

        self.action_embedding = nn.Embedding(n_actions, 16)
        self.mini_conv = nn.Sequential(nn.Conv2d(minimap_channel, 16, 5, 1, 2),
                                       nn.ReLU(),
                                       nn.Conv2d(16, 32, 3, 1, 1),
                                       nn.ReLU())
        # (B, C, 64, 64) -> (B, 32, 64, 64)

        self.screen_conv = nn.Sequential(nn.Conv2d(screen_channel, 16, 5, 1, 2),
                                         nn.ReLU(),
                                         nn.Conv2d(16, 32, 3, 1, 1),
                                         nn.ReLU())
        # (B, C, 64, 64) -> (B, 32, 64, 64)

        self.nonspatial_conv = Dense2Conv()

        self.nonspatial_dense = nn.Sequential(nn.Linear(nonspatial_dim, 128),
                                              nn.ReLU(),
                                              nn.Linear(128, 32))

        # (B, C) -> (B, 32) -> (B, 32, 64, 64)

        # concatenate mini_conv, screen_conv, nonspatial_conv
        self.mini_out = nn.Sequential(nn.Conv2d(32, 1, 1, 1),
                                      Flatten())

        self.screen_out = nn.Sequential(nn.Conv2d(32, 1, 1, 1),
                                        Flatten())
 
        self.nonspatial_MLP = nn.Sequential(nn.Linear(2*64*64 + 32, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 256))
    
    def forward(self, minimap, screen, player, last_action):

        last_action = self.action_embedding(last_action)

        nonspatial = torch.cat([player, last_action], dim=-1)
        nonspatial_latent = self.nonspatial_dense(nonspatial)

        mini_conv = self.mini_conv(minimap)
        screen_conv = self.screen_conv(screen)
        nonspatial_conv = self.nonspatial_conv(nonspatial_latent)

        # spatial_out : in shape (B, 96, 64, 64)
        spatial_out = torch.cat([mini_conv, screen_conv, nonspatial_conv], dim=1)

        mini_out = self.mini_out(mini_conv)
        screen_out = self.screen_out(screen_conv)
        nonspatial_out = torch.cat([mini_out, screen_out, nonspatial_latent], dim=-1)

        # nonspatial_out : in shape (B, 512)
        nonspatial_out = self.nonspatial_MLP(nonspatial_out)

        return spatial_out, nonspatial_out