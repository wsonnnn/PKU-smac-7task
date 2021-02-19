from Relational import SC2Net

import torch
import torch.nn as nn
import torch.nn.functional as F

import arglist
from Preprocess import get_feature_embed
from utils import Flatten

from pysc2.lib import actions, features
from torch.distributions import Categorical

import numpy as np

from torch.autograd import Variable


debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


spatial_features = ['screen', 'minimap', 'screen2']

nonspatial_features = [arg.name for arg in actions.TYPES if arg.name not in spatial_features]

total_args = [arg.name for arg in actions.TYPES]

minimap_channel = 11
screen_channel = 27
EPS = 1e-20
nonspatial = 11
n_actions = arglist.NUM_ACTIONS

BASELINE_SCALE = 0.1
ENTROPY_SCALE = 1e-3


latent_net = SC2Net(minimap_channel, screen_channel, nonspatial, n_actions)

class A2CActor(nn.Module):

    def __init__(self, nonspatial_dim=576, hidden_dim=256, screen_channel=64):

        super(A2CActor, self).__init__()

        self.latent_net = latent_net

        self.base_policy_logit = nn.Sequential(nn.Linear(nonspatial_dim, hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(256, n_actions),
                                               nn.Softmax(dim=-1))

        spatial_args = {"screen": 0, "minimap": 0, "screen2": 0}
        
        # calculate spatial_args dict
        for (key, value) in spatial_args.items():
            spatial_args[key] = nn.Sequential(nn.ConvTranspose2d(screen_channel, 16, 4, 2, 1),
                                              nn.ReLU(),
                                              nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                              nn.ReLU(),
                                              nn.Conv2d(16, 1, 1, 1),
                                              nn.Upsample(scale_factor=2, mode='nearest'),
                                              Flatten(),
                                              nn.Softmax(dim=-1))
        
        self.spatial_args = nn.ModuleDict(spatial_args)

        mask = np.eye(64 * 64)
        self.spatial_mask = torch.cat([torch.zeros(64 * 64).view(1, -1), torch.FloatTensor(mask)], dim=0).to(device)

        # calculate non spatial args
        nonspatial = {}
        self.nonspatial_mask = {}
        for arg in actions.TYPES:
            if arg.name not in spatial_features:
                size = arg.sizes[0]
                nonspatial[arg.name] = nn.Sequential(nn.Linear(nonspatial_dim, hidden_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(hidden_dim, size),
                                                     nn.Softmax(dim=-1))
                mask = torch.FloatTensor(np.eye(size))
                temp = torch.cat([torch.zeros(size).view(1, -1), mask], dim=0).to(device)
                self.nonspatial_mask[arg.name] = temp
                
        self.nonspatial_args = nn.ModuleDict(nonspatial)
    
    def forward(self, minimaps, screens, players, last_actions):
        """
        return the base_action_prob, spatial_prob (as dict), nonspatial_prob (as dict)
        """

        spatial, nonspatial = self.latent_net(minimaps, screens, players, last_actions)

        base_prob = self.base_policy_logit(nonspatial)

        spatial_prob = {}
        for feature in spatial_features:
            spatial_prob[feature] = self.spatial_args[feature](spatial)
        
        nonspatial_prob = {}
        for feature in nonspatial_features:
            nonspatial_prob[feature] = self.nonspatial_args[feature](nonspatial)
        
        return base_prob, spatial_prob, nonspatial_prob
    
    def choose_action(self, minimap, screen, player, last_action, valid_action):

        base_prob, spatial_prob, nonspatial_prob = self.forward(minimap, screen, player, last_action)

        masked_base_prob = base_prob * valid_action
        sum_prob = masked_base_prob.sum(-1).view(-1, 1)
        masked_base_prob = masked_base_prob / sum_prob

        dice = torch.distributions.Categorical

        base_action = dice(masked_base_prob.detach()).sample().item()

        required_args = [arg.name for arg in actions.FUNCTIONS[base_action].args]

        arg_list, spatial_args, nonspatial_args = [], [], []

        picked_up = {}
        for name in required_args:
            if name in spatial_features:
                pos = dice(spatial_prob[name].detach()).sample().item()
                arg_list.append([pos//64, pos%64])
            else:
                pos = dice(nonspatial_prob[name].detach()).sample().item()
                arg_list.append([pos])
            picked_up[name] = pos
        
        for feat in total_args:
            if feat in spatial_features:
                if feat in required_args:
                    pos = picked_up[feat]
                else:
                    pos = -1
                spatial_args.append(pos)
            else:
                if feat in required_args:
                    pos = picked_up[feat]
                else:
                    pos = -1
                nonspatial_args.append(pos)
        
        func_action = actions.FunctionCall(base_action, arg_list)

        return func_action, base_action, spatial_args, nonspatial_args

    def loss_fn(self, minimaps, screens, players, 
                      last_actions, valid_actions, 
                      base_action, spatial_args, nonspatial_args, advantage):

        batch_size = len(base_action)

        base_prob, spatial_prob, nonspatial_prob = self.forward(minimaps, screens, players, last_actions)
        
        masked_base_prob = base_prob * valid_actions
        scaled = torch.sum(masked_base_prob, dim=-1).view(-1, 1)
        scaled_prob = masked_base_prob / scaled
        

        # figure out the log_prob
        index = torch.LongTensor(np.arange(batch_size)).to(device)
        actor_logits = torch.log(torch.clamp(scaled_prob, EPS, 1.0))[index, base_action]

        #spatial arguments
        for i in range(len(spatial_features)):
            name = spatial_features[i]
            actor_logits = actor_logits + (torch.log(torch.clamp(spatial_prob[name], EPS, 1.0)) * self.spatial_mask[spatial_args[i] + 1]).sum(-1)
        
        for i in range(len(nonspatial_features)):
            name = nonspatial_features[i]
            actor_logits = actor_logits + (torch.log(torch.clamp(nonspatial_prob[name], EPS, 1.0)) * self.nonspatial_mask[name][nonspatial_args[i] + 1]).sum(-1)

        actor_loss = -1 * actor_logits * advantage
        
        '''
        calculate entropy_loss
        '''
 
        entropy_loss = torch.sum(scaled_prob * torch.log(torch.clamp(scaled_prob, EPS, 1.0)), dim=-1)
        
        for feat in spatial_features:
            entropy_loss = entropy_loss + (spatial_prob[feat] * \
                             torch.clamp(spatial_prob[feat], EPS, 1.0)).sum(-1)
        
        for feat in nonspatial_features:
            entropy_loss = entropy_loss + (nonspatial_prob[feat] * \
                             torch.clamp(nonspatial_prob[feat], EPS, 1.0)).sum(-1)

        loss = (actor_loss + 1e-3 * entropy_loss)
        return loss.mean()

# (base_action, arg1, arg2, ... )
# screen, minimap, screen2 
# [0]
# 

class A2CCritic(nn.Module):

    def __init__(self, nonspatial_dim=576, hidden_dim=256):

        super(A2CCritic, self).__init__()
        self.gamma = 0.99

        self.latent_net = latent_net

        self.value_net = nn.Sequential(nn.Linear(nonspatial_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
    
    def forward(self, minimaps, screens, players, last_actions):

        _, nonspatial = self.latent_net(minimaps, screens, players, last_actions)
        values = self.value_net(nonspatial)
        return values

    def loss_fn(self, minimaps, screens, player, last_action, 
                      n_minimap, n_screen, n_player, n_laction, rewards):

        # the last state as boostrapping
        with torch.no_grad():
            boostrapping = self.forward(n_minimap, n_screen, n_player, n_laction)

        batch_q = self._get_q_value(rewards, boostrapping)
        batch_q = torch.FloatTensor(batch_q).view(-1, 1).to(device)

        values = self.forward(minimaps, screens, player, last_action)
        
        advantage = batch_q - values
        critic_loss = advantage.pow(2).mean()

        return advantage, critic_loss
    
    def _get_q_value(self, rewards, boostrap):

        run_q = boostrap
        batch_q = []
        for reward in rewards[::-1]:
            run_q = reward + run_q * self.gamma
            batch_q.append(run_q)

        batch_q.reverse()

        batch_q = np.array(batch_q)
        batch_q = (batch_q - batch_q.mean())/(1e-8 + batch_q.std())

        return batch_q

class ActorCritic(nn.Module):

    def __init__(self, minimap_channel, screen_channel, nonspatial, n_actions):

        super(ActorCritic, self).__init__()

        self.gamma = 0.99
        self.LAMBDA= 0.95

        self.latent_net = SC2Net(minimap_channel, screen_channel, nonspatial, n_actions)

        self.base_policy_logit = nn.Sequential(nn.Linear(576, 256),
                                               nn.ReLU(),
                                               nn.Linear(256, n_actions),
                                               nn.Softmax(dim=-1))


        self.mini_embed = get_feature_embed(features.MINIMAP_FEATURES).to(device)
        self.screen_embed = get_feature_embed(features.SCREEN_FEATURES).to(device)

        spatial_args = {"screen": 0, "minimap": 0, "screen2": 0}
        
        # calculate spatial_args dict
        for (key, value) in spatial_args.items():
            spatial_args[key] = nn.Sequential(nn.ConvTranspose2d(64, 16, 4, 2, 1),
                                              nn.ReLU(),
                                              nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                              nn.ReLU(),
                                              nn.Conv2d(16, 1, 1, 1),
                                              nn.Upsample(scale_factor=2, mode='nearest'),
                                              Flatten(),
                                              nn.Softmax(dim=-1))
        
        self.spatial_args = nn.ModuleDict(spatial_args)

        mask = np.eye(64 * 64)
        self.spatial_mask = torch.cat([torch.zeros(64 * 64).view(1, -1), torch.FloatTensor(mask)], dim=0).to(device)

        # calculate non spatial args
        nonspatial = {}
        self.nonspatial_mask = {}
        for arg in actions.TYPES:
            if arg.name not in spatial_features:
                size = arg.sizes[0]
                nonspatial[arg.name] = nn.Sequential(nn.Linear(576, 256),
                                                     nn.ReLU(),
                                                     nn.Linear(256, size),
                                                     nn.Softmax(dim=-1))
                mask = torch.FloatTensor(np.eye(size))
                temp = torch.cat([torch.zeros(size).view(1, -1), mask], dim=0).to(device)
                self.nonspatial_mask[arg.name] = temp
                
        self.nonspatial_args = nn.ModuleDict(nonspatial)

        self.value_net = nn.Sequential(nn.Linear(576, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 1))
    
    def encode_screen(self, screen):
        layers = []
        for i in range(len(features.SCREEN_FEATURES)):
            if features.SCREEN_FEATURES[i].type == features.FeatureType.CATEGORICAL:
                name = features.SCREEN_FEATURES[i].name
                layer = screen[:,i].type(torch.int64).to(device)
                layer = self.screen_embed[name](layer).permute(0, 3, 1, 2)
            else:
                layer = torch.log(1.0 + screen[:,i:i + 1]).to(device)
                # layer = screen[:, i:i+1] / features.SCREEN_FEATURES[i].scale
                # layer = screen[:,i:i+1]
            layers.append(layer)
        return torch.cat(layers, dim=1).to(device)
    
    def encode_minimap(self, minimap):
        layers = []
        for i in range(len(features.MINIMAP_FEATURES)):
            if features.MINIMAP_FEATURES[i].type == features.FeatureType.CATEGORICAL:
                name = features.MINIMAP_FEATURES[i].name
                layer = minimap[:,i].type(torch.int64).to(device)
                if debug:
                    print("categorical shape", layer.shape)
                layer = self.mini_embed[name](layer).permute(0, 3, 1, 2)
            else:
                if debug:
                    print("Minimap shape", minimap.shape)
                    print("non categorical shape", minimap[:,i:i+1].shape)
                layer = torch.log(1.0 + minimap[:,i:i+1]).to(device)
                # layer = minimap[:,i:i+1] / features.MINIMAP_FEATURES[i].scale
                # layer = minimap[:,i:i+1]
            layers.append(layer.to(device))
        return torch.cat(layers, dim=1).to(device)

    def forward(self, minimaps, screens, players, last_actions):
        """
        return the base_action_prob, spatial_prob (as dict), nonspatial_prob (as dict)
        """

        spatial, nonspatial = self.latent_net(minimaps, screens, players, last_actions)

        base_prob = self.base_policy_logit(nonspatial)

        spatial_prob = {}
        for feature in spatial_features:
            spatial_prob[feature] = self.spatial_args[feature](spatial)
        
        nonspatial_prob = {}
        for feature in nonspatial_features:
            nonspatial_prob[feature] = self.nonspatial_args[feature](nonspatial)
        
        values = self.value_net(nonspatial)
        
        return base_prob, spatial_prob, nonspatial_prob, values
    
    def choose_action(self, minimap, screen, player, last_action, valid_action, israndom=True):

        minimap = self.encode_minimap(minimap)
        screen = self.encode_screen(screen)

        base_prob, spatial_prob, nonspatial_prob, _ = self.forward(minimap, screen, player, last_action)

        masked_base_prob = base_prob * valid_action
        sum_prob = masked_base_prob.sum(-1).view(-1, 1)
        masked_base_prob = masked_base_prob / sum_prob

        arg_list, spatial_args, nonspatial_args = [], [], []

        dice = torch.distributions.Categorical

        if israndom:
            base_action = dice(masked_base_prob.detach()).sample().item()
        else:
            base_action = torch.argmax(masked_base_prob,dim=1).item()
        
        required_args = [arg.name for arg in actions.FUNCTIONS[base_action].args]

        picked_up = {}
        for name in required_args:
            if name in spatial_features:
                if israndom:
                    pos = dice(spatial_prob[name]).sample().item()
                else:
                    pos = torch.argmax(spatial_prob[name], dim=1).item()
                arg_list.append([pos//64, pos%64])
            else:
                if israndom:
                    pos = dice(nonspatial_prob[name].detach()).sample().item()
                else:
                    pos = torch.argmax(nonspatial_prob[name], dim=1).item()
                arg_list.append([pos])
            picked_up[name] = pos
        
        for feat in total_args:
            if feat in spatial_features:
                if feat in required_args:
                    pos = picked_up[feat]
                else:
                    pos = -1
                spatial_args.append(pos)
            else:
                if feat in required_args:
                    pos = picked_up[feat]
                else:
                    pos = -1
                nonspatial_args.append(pos)
        
        func_action = actions.FunctionCall(base_action, arg_list)

        return func_action, base_action, spatial_args, nonspatial_args

    def loss_fn(self, minimaps, screens, players, 
                      last_actions, valid_actions, 
                      base_action, spatial_args, nonspatial_args, rewards,
                      n_minimap, n_screen, n_player, n_laction):

        # a batch is [trace1, trace2, trace3, ... trace_m], m is batch_size, we let m = 8
        batch_size = len(base_action)

        minimaps = self.encode_minimap(minimaps)
        screens = self.encode_screen(screens)

        base_prob, spatial_prob, nonspatial_prob, values= self.forward(minimaps, screens, players, last_actions)

        with torch.no_grad():
            n_minimap = self.encode_minimap(n_minimap)
            n_screen = self.encode_screen(n_screen)
            base_, spa_, non_, boostrapping = self.forward(n_minimap, n_screen, n_player, n_laction)

        '''
        batch_q = torch.FloatTensor(rewards).to(device) + 
                  self.gamma * torch.cat([values[1:].detach(), boostrapping.detach()], dim=0).view(-1).to(device)
        '''
        batch_q = self._get_q_value(rewards, boostrapping.item())
        batch_q = torch.FloatTensor(batch_q).to(device)

        # critic loss
        advantage = batch_q - values.view(-1)
        critic_loss = advantage.pow(2)
        
        # acotr loss
        masked_base_prob = base_prob * valid_actions
        scaled = torch.sum(masked_base_prob, dim=-1).view(-1, 1)
        scaled_prob = masked_base_prob / scaled

        # figure out the log_prob
        index = torch.LongTensor(np.arange(batch_size)).to(device)
        actor_logits = torch.log(torch.clamp(scaled_prob, EPS, 1.0))[index, base_action]

        #spatial arguments
        for i in range(len(spatial_features)):
            name = spatial_features[i]
            actor_logits = actor_logits + (torch.log(torch.clamp(spatial_prob[name], EPS, 1.0)) * self.spatial_mask[spatial_args[i] + 1]).sum(-1)
        
        for i in range(len(nonspatial_features)):
            name = nonspatial_features[i]
            actor_logits = actor_logits + (torch.log(torch.clamp(nonspatial_prob[name], EPS, 1.0)) * self.nonspatial_mask[name][nonspatial_args[i] + 1]).sum(-1)

        actor_loss = -1 * actor_logits * advantage.detach()
        
        
        # entropy loss
        entropy_loss = torch.sum(scaled_prob * torch.log(torch.clamp(scaled_prob, EPS, 1.0)), dim=-1)
        
        for feat in spatial_features:
            entropy_loss = entropy_loss + (spatial_prob[feat] * \
                             torch.clamp(spatial_prob[feat], EPS, 1.0)).sum(-1)
        
        for feat in nonspatial_features:
            entropy_loss = entropy_loss + (nonspatial_prob[feat] * \
                             torch.clamp(nonspatial_prob[feat], EPS, 1.0)).sum(-1)

        # entropy_loss = 0
        loss = (actor_loss + BASELINE_SCALE * critic_loss + ENTROPY_SCALE * entropy_loss).mean()
        return loss
    
    def _get_q_value(self, rewards, boostrap):

        run_q = boostrap
        batch_q = []
        for reward in rewards[::-1]:
            run_q = reward + run_q * self.gamma
            batch_q.append(run_q)

        batch_q.reverse()

        batch_q = np.array(batch_q)
        return batch_q

    def _get_gae(self, advantages):
        gae = 0
        gae_lst = []
        for i in range(advantages.shape[0]):
            gae = advantages[-i].item() + gae * self.LAMBDA
            gae_lst.append(gae)
        
        gae_lst.reverse()
        gae_lst = torch.FloatTensor(np.array(gae_lst) * (1 - self.LAMBDA)).to(device)
        return gae_lst