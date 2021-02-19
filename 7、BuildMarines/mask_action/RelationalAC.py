from Relational import SC2Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import arglist


minimap_channel = 11
screen_channel = 27

debug = False

nonspatial = 11 #player
n_actions = arglist.NUM_ACTIONS

latent_net = SC2Net(minimap_channel, screen_channel, nonspatial, n_actions)


class RelationalActor(nn.Module):

    def __init__(self, nonspatial_dim=576, hidden_dim=256, screen_channel=64):

        super(RelationalActor, self).__init__()

        self.latent_net = latent_net

        self.base_policy_logit = nn.Sequential(nn.Linear(nonspatial_dim, hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(256, n_actions))
        
        self.screen1_logit = nn.Sequential(nn.ConvTranspose2d(screen_channel, 16, 4, 2, 1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                           nn.Conv2d(16, 1, 1, 1),
                                           nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.screen2_logit = nn.Sequential(nn.ConvTranspose2d(screen_channel, 16, 4, 2, 1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                           nn.Conv2d(16, 1, 1, 1),
                                           nn.Upsample(scale_factor=2, mode='nearest'))
    
    def forward(self, minimaps, screens, valid_actions, players, last_actions):

        spatial, nonspatial = self.latent_net(minimaps, screens, players, last_actions)

        prob_categorical = self.base_policy_logit(nonspatial)
        prob_screen1 = self.screen1_logit(spatial).view(-1, 64*64)
        prob_screen2 = self.screen2_logit(spatial).view(-1, 64*64)

        return {'categorical': prob_categorical, 'screen1': prob_screen1, 'screen2': prob_screen2}


class RelationalCritic(nn.Module):

    def __init__(self, nonspatial_dim=576, n_actions=n_actions, hidden_dim=256):

        super(RelationalCritic, self).__init__()
        self.latent_net = latent_net

        self.q1_net = nn.Sequential(nn.Linear(nonspatial_dim+n_actions+64*64*2, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        
        self.q2_net = nn.Sequential(nn.Linear(nonspatial_dim+n_actions+64*64*2, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
    
    def Q_value(self, minimaps, screens, valid_actions, players, last_actions, actions):
        Q, _ = self.forward(minimaps, screens, valid_actions, players, last_actions, actions)
        return Q
    
    def forward(self, minimaps, screens, valid_actions, players, last_actions, actions):

        screen1_actions = actions['screen1'].view(-1, 64*64)
        screen2_actions = actions['screen2'].view(-1, 64*64)

        _, nonspatial = self.latent_net(minimaps, screens, players, last_actions)

        x = torch.cat([nonspatial, actions['categorical'], screen1_actions, screen2_actions], dim=-1)
        Q1 = self.q1_net(x)
        Q2 = self.q2_net(x)

        return Q1, Q2

        
        
        
