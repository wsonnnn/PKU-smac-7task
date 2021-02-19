import torch
import numpy as np
from absl import flags

import copy
from pysc2.env import sc2_env
from collections import deque
import random
import sys
from copy import deepcopy
from torch.distributions import Categorical
from pysc2.lib import actions as sc2_actions
import torch.nn.functional as F
import arglist
from AC_FullyConv_q import Actor_FullyConv, Critic_FullyConv

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

# ****************prarm
agent_format = sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(
    screen=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
    minimap=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
))
map_names = ["MoveToBeacon", "CollectMineralShards", "FindAndDefeatZerglings", "DefeatRoaches", "DefeatZerglingsAndBanelings", "CollectMineralsAndGas", "BuildMarines"]
max_episode = 15000
max_step = 20000
map_name = map_names[0]
FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
env = sc2_env.SC2Env(map_name=map_name, step_mul=2, visualize=False, game_steps_per_episode=10000, agent_interface_format=[agent_format], players=[sc2_env.Agent(sc2_env.Race.terran)])

action_spec = env.action_spec()
obs_spec = env.observation_spec()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 0.001
START_EPSILON = 1.0
FINAL_EPSILON = 0.03
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
MEMORY_SIZE = 2000
MEMORY_THRESHOLD = 500
UPDATE_TIME = 500
TEST_FREQUENCY = 20
screen_channel = 10
POLICY_UPDATE = 2
minimap_channel = 10
target_update_rate = 0.005


# ****************prarm
def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class Agent(object):
    def __init__(self):

        self.actor = Actor_FullyConv().to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic_FullyConv().to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.memory = deque()
        self.learning_count = 0

    def _onehot1d(self, x):
        y = np.zeros((arglist.NUM_ACTIONS, ), dtype='float32')
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

    def select_action_mtb(self, obs):
        if FUNCTIONS.Move_screen.id in obs[3]["available_actions"]:
            player_relative = obs[3]["feature_screen"]
            player_relative = player_relative["player_relative"]
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = np.mean(beacon, axis=0).round()

            return FUNCTIONS.Move_screen("now", beacon_center)
        else:
            return FUNCTIONS.select_army("select")

    def select_action(self, obs, valid_actions, train=False, target=False):

        minimap = obs[0].astype('float32')
        screen = obs[1].astype('float32')

        minimap = torch.from_numpy(minimap).unsqueeze(0).to(device)
        screen = torch.from_numpy(screen).unsqueeze(0).to(device)
        valid_actions = torch.from_numpy(valid_actions).unsqueeze(0).to(device)
        # value，base_action_probability和args_probability
        if target is False:
            action_total = self.actor.forward(minimap, screen, valid_actions)
        else:
            action_total = self.actor_target.forward(minimap, screen, valid_actions)
        action_prob = torch.nn.functional.softmax(action_total['categorical'], dim=1).detach()
        scr1_prob = torch.nn.functional.softmax(action_total['screen1'].view(1, -1), dim=-1).detach()
        scr2_prob = torch.nn.functional.softmax(action_total['screen2'].view(1, -1), dim=-1).detach()
        action_prob = self.mask_unavailable(action_prob, valid_actions)
        dicision = Categorical(action_prob)
        action_id = dicision.sample().item()

        scr1_prob = Categorical(scr1_prob).sample()[0].item()
        scr2_prob = Categorical(scr2_prob).sample()[0].item()

        action_pos = [[int(scr1_prob % arglist.FEAT2DSIZE), int(scr1_prob // arglist.FEAT2DSIZE)], [int(scr2_prob % arglist.FEAT2DSIZE), int(scr2_prob // arglist.FEAT2DSIZE)]]  # (x, y)

        args = []
        cnt = 0
        for arg in sc2_actions.FUNCTIONS[action_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(action_pos[cnt])
                cnt += 1
            else:
                args.append([0])

        action = sc2_actions.FunctionCall(action_id, args)
        return action

    def get_observation(self, state):
        obs_flat = state.observation['available_actions']
        obs_flat = self._onehot1d(obs_flat)
        obs = [state.observation['feature_minimap'], state.observation['feature_screen'], obs_flat]
        return obs

    def postprocess_action(self, act):
        act_categorical = np.zeros(shape=(arglist.NUM_ACTIONS, ), dtype='float32')
        act_categorical[act.function] = 1.

        act_screens = [np.zeros(shape=(1, arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), dtype='float32')] * 2
        i = 0
        for arg in act.arguments:
            if arg != [0]:
                act_screens[i][0, arg[0], arg[1]] = 1.
                i += 1

        act = {'categorical': act_categorical, 'screen1': act_screens[0], 'screen2': act_screens[1]}
        return act

    def save(self, i):
        torch.save(self.critic.state_dict(), "models/sc2_critic_{}_{}.pth".format(map_name, i))
        torch.save(self.actor.state_dict(), "models/sc2_actor_{}_{}.pth".format(map_name, i))

    def load(self, i):
        self.critic.load_state_dict(torch.load("models/sc2_critic_{}_{}.pth".format(map_name, i)))
        self.actor.load_state_dict(torch.load("models/sc2_actor_{}_{}.pth".format(map_name, i)))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

    def learn(self):
        if len(self.memory) < MEMORY_THRESHOLD:
            return
        self.learning_count += 1

        batch = random.sample(self.memory, BATCH_SIZE)
        state = []
        action = []
        reward = []
        next_state = []
        for x in batch:

            state.append(x[0])
            action.append(x[1])
            reward.append(x[2])
            next_state.append(x[3])

        # state = torch.FloatTensor(state)
        minimap = torch.FloatTensor(np.stack([x[0] for x in state], axis=0)).to(device)
        screen = torch.FloatTensor(np.stack([x[1] for x in state], axis=0)).to(device)
        valid_action = torch.FloatTensor(np.stack([x[2] for x in state], axis=0)).to(device)
        #action = torch.LongTensor(action).unsqueeze(1)
        batch_action = {'categorical': [], 'screen1': [], 'screen2': []}
        for act_dict in action:
            for key, act in act_dict.items():
                batch_action[key].append(act)
        for key, act in batch_action.items():
            temp = np.stack(act, axis=0)
            batch_action[key] = torch.FloatTensor(temp).to(device)
        action = batch_action
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        # next_state = torch.FloatTensor(next_state)
        n_minimap = torch.FloatTensor(np.stack([x[0] for x in next_state], axis=0)).to(device)
        n_screen = torch.FloatTensor(np.stack([x[1] for x in next_state], axis=0)).to(device)
        n_valid = torch.FloatTensor(np.stack([x[2] for x in next_state], axis=0)).to(device)

        with torch.no_grad():
            # n_minimap = next_state[:][0]
            # n_screen = next_state[:][1]
            # n_valid = self._onehot1d(next_state[:][2])
            # n_minimap = torch.from_numpy(n_minimap).to(device)
            # n_screen = torch.from_numpy(n_screen).to(device)
            # n_valid = torch.from_numpy(n_valid).to(device)

            next_action = self.actor_target.forward(n_minimap, n_screen, n_valid)
            next_action["categorical"] = torch.nn.functional.softmax(next_action['categorical'], dim=-1)
            masked_base_prob = torch.mul(next_action["categorical"], valid_action)
            next_action["categorical"] = masked_base_prob / torch.sum(masked_base_prob, dim=1).view(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target.forward(n_minimap, n_screen, n_valid, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            y_target_Q = reward + GAMMA * target_Q

        q1, q2 = self.critic(minimap, screen, valid_action, action)

        critic_loss = F.mse_loss(q1, y_target_Q) + F.mse_loss(q2, y_target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_count % POLICY_UPDATE == 0:
            action_c = self.actor.forward(n_minimap, n_screen, n_valid)
            action_c["categorical"] = torch.nn.functional.softmax(action_c['categorical'], dim=-1)
            masked_base_prob = torch.mul(action_c["categorical"], valid_action)
            action_c["categorical"] = masked_base_prob / torch.sum(masked_base_prob, dim=1).view(-1, 1)
            actor_loss = -self.critic.Q_value(minimap, screen, valid_action, action_c).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)


agent = Agent()
agent.load(20)
test_game = True
if test_game is True:
    sums = 0
    for _ in range(100):
        r_all = 0
        state = env.reset()[0]
        while True:
            obs = agent.get_observation(state)
            if map_name == "MoveToBeacon":
                if obs[2][331] == 0:
                    for x in range(573):
                        obs[2][x] = 0
                    obs[2][7] = 1
                else:
                    for x in range(573):
                        obs[2][x] = 0
                    obs[2][331] = 1
                
            actions = agent.select_action(obs, valid_actions=obs[2], train=False)
            next_state = env.step(actions=[actions])[0]
            next_obs = agent.get_observation(next_state)
            state = deepcopy(next_state)
            if state.last():
                r_all = state.observation["score_cumulative"][0]
                sums += r_all
                break
        print(r_all)
    print("sum=", sums/100)
else:
    for i in range(max_episode):
        if (i % 5 == 0):
            print(i)
        state = env.reset()[0]
        num = 0
        for t in range(max_step):
            obs = agent.get_observation(state)
            actions = agent.select_action(obs, valid_actions=obs[2], train=True)
            next_state = env.step(actions=[actions])[0]
            next_obs = agent.get_observation(next_state)
            actions = agent.postprocess_action(actions)

            agent.memory.append((obs, actions, state.reward, next_obs))
            if len(agent.memory) > MEMORY_SIZE:
                agent.memory.popleft()
            num += 1
            state = deepcopy(next_state)
            if state.last():
                break
        for _ in range(10):
            agent.learn()

        if (i % TEST_FREQUENCY == 0):
            r_all = 0
            state = env.reset()[0]
            while True:
                obs = agent.get_observation(state)
                actions = agent.select_action(obs, valid_actions=obs[2], train=False)
                next_state = env.step(actions=[actions])[0]
                next_obs = agent.get_observation(next_state)
                state = deepcopy(next_state)
                if state.last():
                    r_all = state.observation["score_cumulative"][0]
                    break
            agent.save(i)
            print('episode: {} , total_reward: {}'.format(i, r_all))
