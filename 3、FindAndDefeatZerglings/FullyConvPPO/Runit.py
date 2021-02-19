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
from PPO import PPO

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

# ****************prarm
'''
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
'''

ppo_agent = PPO()
ppo_agent.load(89, 24.500)
'''
for _ in range(100):
    r_all = 0
    state = env.reset()[0]
    last_action = [0]
    while True:
        obs = agent.get_observation(state)

        func_act, base_act, spatial_args, nonspatial_args = agent.select_action(obs, valid_actions=obs[2], last_action=last_action)
        next_state = env.step(actions=[func_act])[0]
        next_obs = agent.get_observation(next_state)
 
        state = deepcopy(next_state)
        if state.last():
            r_all = state.observation["score_cumulative"][0]
            total_r += state.observation["score_cumulative"][0]
            break
        print('episode: {} , total_reward: {}'.format(_, r_all))

print("average reward {}".format(total_r/100))
'''
ppo_agent.test(test_episode=30)