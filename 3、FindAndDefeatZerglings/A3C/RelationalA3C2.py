from Relational import SC2Net

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from Preprocess import get_feature_embed
from utils import Flatten

from pysc2.env import sc2_env
from pysc2.lib import actions, features

import torch.multiprocessing as mp
from torch.distributions import Categorical

import numpy as np


debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

spatial_features = ['screen', 'minimap', 'screen2']
nonspatial_features = [arg.name for arg in actions.TYPES if arg.name not in spatial_features]
total_args = [arg.name for arg in actions.TYPES]

minimap_channel = 11
screen_channel = 27
EPS = 1e-20
nonspatial = 11
n_actions = len(actions.FUNCTIONS)

BASELINE_SCALE = 0.1
ENTROPY_SCALE = 1e-3

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


        self.mini_embed = get_feature_embed(features.MINIMAP_FEATURES)
        self.screen_embed = get_feature_embed(features.SCREEN_FEATURES)

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
                #layer = screen[:,i:i+1]
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
                # layer = torch.log(1.0 + minimap[:,i:i+1]).to(device)
                # layer = minimap[:,i:i+1] / features.MINIMAP_FEATURES[i].scale
                layer = minimap[:,i:i+1]
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

# nonspatial 11
class Worker(object):
    def __init__(self, 
                 gnet, 
                 gopt,
                 process_id,
                 map_name,
                 max_step,
                 POLICY_UPDATE,
                 TEST_FREQUENCY,
                 minimap_channel,
                 screen_channel,
                 nonspatial_dim,
                 n_actions,
                 global_nep,
                 global_epr,
                 res_queue):

        # super(Worker, self).__init__()
        torch.manual_seed(30+process_id)
        
        self.name = "w%02d" % process_id
        self.max_step = max_step
        self.gnet = gnet
        self.gopt = gopt

        self.map_name = map_name
        self.lnet = ActorCritic(minimap_channel, screen_channel, nonspatial_dim, n_actions).to(device)
        self.lnet.load_state_dict(self.gnet.state_dict())

        self.env = self.init_env(map_name)

        self.global_nep = global_nep
        self.global_epr = global_epr
        self.res_queue = res_queue

        self.POLICY_UPDATE = POLICY_UPDATE
        self.TEST_FREQUENCY = TEST_FREQUENCY

        self.memory = []
        self.learning_count = 1
        self.update_count = 0

    def _onehot1d(self, x):
        y = np.zeros((n_actions, ), dtype='float32')
        y[x] = 1.
        return y

    def select_action(self, obs, valid_actions, last_action, israndom=True):

        minimap = obs[0].astype('float32')
        screen = obs[1].astype('float32')

        minimap = torch.FloatTensor(np.expand_dims(minimap, axis=0)).to(device)
        screen = torch.FloatTensor(np.expand_dims(screen, axis=0)).to(device)
        valid_actions = torch.from_numpy(valid_actions).unsqueeze(0).to(device)
        player = torch.FloatTensor(obs[3]).unsqueeze(0).to(device)
        last_action = torch.LongTensor(last_action).to(device)

        act_func, base_action, spatial_args, nonspatial_args = self.lnet.choose_action(minimap, 
                                                                                        screen, 
                                                                                        player, 
                                                                                        last_action, 
                                                                                        valid_actions,
                                                                                        israndom=israndom)

        return act_func, base_action, spatial_args, nonspatial_args
    
    def init_env(self, map_name):
        agent_format = sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(
                                                   screen=(64, 64),
                                                   minimap=(64, 64),))

        env = sc2_env.SC2Env(map_name=map_name, 
                             step_mul=80, 
                             visualize=False, 
                             game_steps_per_episode=10000, 
                             agent_interface_format=[agent_format], 
                             players=[sc2_env.Agent(sc2_env.Race.terran)])

        return env
 
    def get_observation(self, state):
        obs_flat = state.observation['available_actions']
        obs_flat = self._onehot1d(obs_flat)
        last_actions = state.observation['last_actions']
        if len(last_actions) == 0:
            last_actions = np.array([0])
        obs = [state.observation['feature_minimap'], state.observation['feature_screen'], obs_flat, state.observation['player']]
        return obs
    
    def save(self, i, r):
        torch.save(self.gnet.state_dict(), "modelsA3C2/sc2_actorcritic_{}_{}_{:.4f}.pth".format(self.map_name, int(i), r))

    def load(self, i, r):
        self.gnet.load_state_dict(torch.load("modelsA3C2/sc2_actorcritic_{}_{}_{:.4f}.pth".format(self.map_name, int(i), r)))

    def learn(self, next_minimap, next_screen, next_player, next_laction):

        self.learning_count = 0
        self.update_count += 1

        batch = self.memory
        state = []
        action = []
        reward = []
        last_action = []

        spatial_args = []
        nonspatial_args = []

        for x in batch:
            state.append(x[0])
            action.append(x[1])
            reward.append(x[2])
            last_action.append(x[3])
            spatial_args.append(x[4])
            nonspatial_args.append(x[5])

        minimap = torch.FloatTensor(np.stack([x[0] for x in state], axis=0)).to(device)
        screen = torch.FloatTensor(np.stack([x[1] for x in state], axis=0)).to(device)
        valid_action = torch.FloatTensor(np.stack([x[2] for x in state], axis=0)).to(device)
        player = torch.FloatTensor(np.log(1 + np.stack([x[3] for x in state], axis=0))).to(device)
        last_action = torch.LongTensor(last_action).squeeze(-1).to(device)

        base_actions = torch.LongTensor(action).to(device)
        spatial_args = torch.LongTensor(np.stack(spatial_args, axis=1)).to(device)
        nonspatial_args = torch.LongTensor(np.stack(nonspatial_args, axis=1)).to(device)

        loss = self.lnet.loss_fn(minimap, screen, player, 
                                last_action, valid_action, 
                                base_actions, spatial_args, nonspatial_args, reward,
                                next_minimap, next_screen, next_player, next_laction)

        self.memory = []

        self.gopt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lnet.parameters(), 100.0)

        for (gpara, lpara) in zip(self.gnet.parameters(), self.lnet.parameters()):
            if gpara._grad is None:
                gpara._grad = lpara.grad
        
        self.gopt.step()
        self.lnet.load_state_dict(self.gnet.state_dict())
    
    def test(self, i):
        r_all = 0
        for j in range(10):
            state = self.env.reset()[0]
            last_action = [0]
            while True:
                obs = self.get_observation(state)

                func_act, base_act, spatial_args, nonspatial_args = self.select_action(obs, valid_actions=obs[2], last_action=last_action, israndom=True)
                next_state = self.env.step(actions=[func_act])[0]
                next_obs = self.get_observation(next_state)

                state = deepcopy(next_state)
                last_action = [base_act]
                if state.last():
                    r_all += state.observation["score_cumulative"][0]
                    break
        r_all /= 10

        self.save(i, r_all)
        print('Hello from {}, Episode: {}, Reward for tests: {}'.format(self.name, i, r_all))

    def run(self):

        total_step = 1
        update_step = 1
        while self.global_nep.value < self.max_step:
            state = self.env.reset()[0]
            last_action = [0]

            for t in range(20000):
                obs = self.get_observation(state)

                func_act, base_act, spatial_args, nonspatial_args = self.select_action(obs, valid_actions=obs[2], last_action=last_action)
                next_state = self.env.step(actions=[func_act])[0]
                next_obs = self.get_observation(next_state)

                total_step += 1
                update_step += 1

                if update_step % self.POLICY_UPDATE == 0 or next_state.last():
                    n_mini = torch.FloatTensor(next_obs[0]).unsqueeze(0).to(device)
                    n_screen = torch.FloatTensor(next_obs[1]).unsqueeze(0).to(device)
                    n_player = torch.FloatTensor(np.log(1 + next_obs[3])).unsqueeze(0).to(device)
                    n_laction = torch.LongTensor([base_act]).to(device)
                    self.learn(n_mini, n_screen, n_player, n_laction)
                    update_step = 0
            
                self.memory.append((obs, base_act, next_state.reward, last_action, spatial_args, nonspatial_args))
            
                last_action = [base_act]
                state = deepcopy(next_state)

                if state.last():
                    reward = state.observation["score_cumulative"][0]
                    '''
                    if self.global_epr.value % self.TEST_FREQUENCY == 0:
                        self.test(self.global_nep.value)
                    '''
                    with self.global_nep.get_lock():
                        if int(self.global_nep.value) % self.TEST_FREQUENCY == 0:
                            self.test(self.global_nep.value)
                        self.global_nep.value += 1
                    with self.global_epr.get_lock():
                        if self.global_epr.value == 0:
                            self.global_epr.value = reward
                        else:
                            self.global_epr.value = self.global_epr.value * 0.99 + reward * 0.01

                    self.res_queue.put(self.global_epr.value)
                    print(self.name,
                    "Epoch: {}".format(self.global_nep.value),
                    "Reward: {}".format(self.global_epr.value))
                    break
        
        self.res_queue.put(None)