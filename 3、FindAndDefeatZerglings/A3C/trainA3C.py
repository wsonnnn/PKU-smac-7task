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

import torch.multiprocessing as mp
from torch.distributions import Categorical
from pysc2.lib import actions as sc2_actions
import torch.nn.functional as F


from Preprocess import minimap_channel, screen_channel
from utils import mask_redundant_actions
from RelationalA3C import Worker, ActorCritic
from sharedAdam import SharedAdam

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import matplotlib.pyplot as plt

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
max_episode = 15000
max_step = 20000
map_name = map_names[2]

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
# env = sc2_env.SC2Env(map_name=map_name, step_mul=80, visualize=False, game_steps_per_episode=10000, agent_interface_format=[agent_format], players=[sc2_env.Agent(sc2_env.Race.terran)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 80
LR = 1e-5
BETAS = (0.9, 0.999)
START_EPSILON = 1.0
FINAL_EPSILON = 0.03
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
MEMORY_SIZE = 500000
MEMORY_THRESHOLD = 1000
TEST_FREQUENCY = 50
POLICY_UPDATE = 80
target_update_rate = 0.005


minimap_c = minimap_channel()
screen_c  = screen_channel()
nonspatial_dim = 11
n_actions = len(actions.FUNCTIONS)

n_processes = 4

def myfunc(process_id, gnet, goptimizer, map_name, max_episode, 
           POLICY_UPDATE, TEST_FREQUENCY, minimap_c, screen_c,
           nonspatial_dim, n_actions, global_nep, global_epr, res_queue):
    worker = Worker(gnet, 
                       goptimizer,
                       process_id,
                       map_name,
                       max_episode,
                       POLICY_UPDATE,
                       TEST_FREQUENCY,
                       minimap_c,
                       screen_c,
                       nonspatial_dim,
                       n_actions,
                       global_nep,
                       global_epr,
                       res_queue)
    worker.run()


if __name__ == "__main__":

    mp.set_start_method("spawn")
    
    gnet = ActorCritic(minimap_c, screen_c, nonspatial_dim, n_actions).to(device)
    '''
    gnet.load_state_dict(torch.load("models2/sc2_actorcritic_{}_{}_{:.4f}.pth".format(map_name, int(i), r)))
    '''
    gnet.share_memory()

    goptimizer = SharedAdam(gnet.parameters(), lr=LR, betas=BETAS)

    global_nep, global_epr, res_queue = mp.Value('i', 0), mp.Value('d', 0), mp.Queue()
    '''
    workers = [Worker( gnet, 
                       goptimizer,
                       i,
                       map_name,
                       max_episode,
                       POLICY_UPDATE,
                       TEST_FREQUENCY,
                       minimap_c,
                       screen_c,
                       nonspatial_dim,
                       n_actions,
                       global_nep,
                       global_epr,
                       res_queue) for i in range(n_threads)]

    [worker.start() for worker in workers]
    '''

    processes = []
    for rank in range(n_processes):
        p = mp.Process(target=myfunc, args=(rank,
                                            gnet, 
                                            goptimizer,
                                            map_name,
                                            max_episode,
                                            POLICY_UPDATE,
                                            TEST_FREQUENCY,
                                            minimap_c,
                                            screen_c,
                                            nonspatial_dim,
                                            n_actions,
                                            global_nep,
                                            global_epr,
                                            res_queue))
        p.start()
        processes.append(p)

    for p in processes: p.join()


