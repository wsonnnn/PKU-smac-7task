import torch
import numpy as np
from absl import flags

import copy
from pysc2.env import sc2_env
from collections import deque
import random
import math
import sys
from copy import deepcopy
from torch.distributions import Categorical
from pysc2.lib import actions as sc2_actions
import torch.nn.functional as F

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

sys.path.append("..")
from Preprocess import minimap_channel, screen_channel
from utils import mask_redundant_actions
from FullyConvA3C import ActorCritic

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

# ****************prarm
agent_format = sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(
    screen=(64, 64),
    minimap=(64, 64),
))

map_names = ["MoveToBeacon", "CollectMineralShards", "FindAndDefeatZerglings", "DefeatRoaches", "DefeatZerglingsAndBanelings", "CollectMineralsAndGas", "BuildMarines"]
max_episode = 10000
max_step = 20000
map_name = map_names[1]

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
env = sc2_env.SC2Env(map_name=map_name, step_mul=32, visualize=False, game_steps_per_episode=10000, agent_interface_format=[agent_format], players=[sc2_env.Agent(sc2_env.Race.terran)])

action_spec = env.action_spec()
obs_spec = env.observation_spec()
n_actions = len(actions.FUNCTIONS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 80
LR = 1e-4
BETAS = (0.9, 0.999)
START_EPSILON = 1.0
FINAL_EPSILON = 0.03
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
MEMORY_SIZE = 500000
MEMORY_THRESHOLD = 1000
TEST_FREQUENCY = 30
POLICY_UPDATE = 80
target_update_rate = 0.005


minimap_c = minimap_channel()
screen_c  = screen_channel()

debug = False

class Agent(object):
    def __init__(self):
        self.agent = ActorCritic(minimap_c, screen_c, 11, n_actions).to(device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=LR, eps=1e-8, betas=BETAS)

        self.memory = []
        self.learning_count = 1

    def _onehot1d(self, x):
        y = np.zeros((n_actions, ), dtype='float32')
        y[x] = 1.
        return y

    def mask_unavailable(self, policy, valid_actions):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        masked_policy_vb = policy * valid_actions
        masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb

    def _test_valid_action(self, function_id, valid_actions):
        if valid_actions[0][function_id] == 1:
            return True
        else:
            return False

    #(minimap, screen, valid_action, player, last)
    def select_action(self, obs, valid_actions, last_action, israndom=True):

        minimap = obs[0].astype('float32')
        screen = obs[1].astype('float32')

        minimap = torch.FloatTensor(np.expand_dims(minimap, axis=0)).to(device)
        screen = torch.FloatTensor(np.expand_dims(screen, axis=0)).to(device)
        valid_actions = torch.from_numpy(valid_actions).unsqueeze(0).to(device)
        player = torch.FloatTensor(obs[3]).unsqueeze(0).to(device)
        last_action = torch.LongTensor(last_action).to(device)

        # value，base_action_probability和args_probability
        act_func, base_action, spatial_args, nonspatial_args = self.agent.choose_action(minimap, 
                                                                                        screen, 
                                                                                        player, 
                                                                                        last_action, 
                                                                                        valid_actions,
                                                                                        israndom=israndom)

        return act_func, base_action, spatial_args, nonspatial_args
 
    def get_observation(self, state):
        obs_flat = state.observation['available_actions']

        obs_flat = self._onehot1d(obs_flat)

        last_actions = state.observation['last_actions']
        if len(last_actions) == 0:
            last_actions = np.array([0])
        obs = [state.observation['feature_minimap'], state.observation['feature_screen'], obs_flat, state.observation['player']]
        return obs

    def postprocess_action(self, act):
        act_categorical = np.zeros(shape=(n_actions, ), dtype='float32')
        act_categorical[act.function] = 1.

        act_screens = [np.zeros(shape=(1, 64, 64), dtype='float32')] * 2
        i = 0
        for arg in act.arguments:
            if arg != [0]:
                act_screens[i][0, int(arg[0]), int(arg[1])] = 1.
                i += 1

        act = {'categorical': act_categorical, 'screen1': act_screens[0], 'screen2': act_screens[1]}
        return act

    def save(self, i, r):
        torch.save(self.agent.state_dict(), "modelsPPO/sc2_actorcritic_{}_{}_{:.4f}.pth".format(map_name, i, r))

    def load(self, i, r):
        self.agent.load_state_dict(torch.load("modelsPPO/sc2_actorcritic_{}_{}_{:.4f}.pth".format(map_name, i, r), map_location=device))

    def learn(self, next_minimap, next_screen, next_player, next_laction):

        self.learning_count = 0

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

        loss = self.agent.loss_fn(minimap, screen, player, 
                                  last_action, valid_action, 
                                  base_actions, spatial_args, nonspatial_args, reward,
                                  next_minimap, next_screen, next_player, next_laction)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 100)
        self.optimizer.step()
        
        self.memory = []
