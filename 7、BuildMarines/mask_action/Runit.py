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
import arglist
from RelationalAC import RelationalActor, RelationalCritic

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
map_name = map_names[6]

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
env = sc2_env.SC2Env(map_name=map_name, step_mul=4, visualize=False, game_steps_per_episode=10000, agent_interface_format=[agent_format], players=[sc2_env.Agent(sc2_env.Race.terran)])

action_spec = env.action_spec()
obs_spec = env.observation_spec()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 5e-5
BETAS = (0.9, 0.999)
START_EPSILON = 1.0
FINAL_EPSILON = 0.03
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
MEMORY_SIZE = 3000
MEMORY_THRESHOLD = 500
TEST_FREQUENCY = 10
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

        self.actor = RelationalActor().to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = RelationalCritic().to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR, betas=BETAS)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR, betas=BETAS)
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

    def gumbel_softmax_hard(self, x):
        shape = x.shape
        if len(shape) == 4:
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(shape[0], -1)
            y = torch.nn.functional.gumbel_softmax(x_reshape, hard=True, dim=-1)
            # We have to reshape Y
            y = y.contiguous().view(shape)
        else:
            y = torch.nn.functional.gumbel_softmax(x, hard=True, dim=-1)

        return y

    def select_action_cmd(self, obs, t):
        if FUNCTIONS.Harvest_Gather_screen.id in obs[3]["available_actions"]:
            player_relative = obs[3]["feature_screen"]["player_relative"]
            m = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            return FUNCTIONS.Harvest_Gather_screen("now", [10, 10])

        if FUNCTIONS.select_rect.id in obs[3]["available_actions"] and t < 1:
            player_relative = obs[3]["feature_screen"]["player_relative"]
            m = _xy_locs(player_relative == _PLAYER_SELF)

            return FUNCTIONS.select_rect("select", [0, 0], [63, 63])

    def _test_valid_action(self, function_id, valid_actions):
        if valid_actions[0][function_id] == 1:
            return True
        else:
            return False

    #(minimap, screen, valid_action, player, last)
    def select_action(self, obs, valid_actions, last_action, train=False, target=False):

        minimap = obs[0].astype('float32')
        screen = obs[1].astype('float32')

        minimap = torch.from_numpy(minimap).unsqueeze(0).to(device)
        screen = torch.from_numpy(screen).unsqueeze(0).to(device)
        valid_actions = torch.from_numpy(valid_actions).unsqueeze(0).to(device)
        player = torch.FloatTensor(obs[3]).unsqueeze(0).to(device)
        last_action = torch.LongTensor(last_action).to(device)
        if last_action.shape[-1] > 1:
            r = 1
        # value，base_action_probability和args_probability
        if target is False:
            action_total = self.actor.forward(minimap, screen, valid_actions, player, last_action)
        else:
            action_total = self.actor_target.forward(minimap, screen, valid_actions)

        action_prob = torch.nn.functional.softmax(action_total['categorical'], dim=1).detach()
        scr1_prob = torch.nn.functional.softmax(action_total['screen1'].view(1, -1), dim=-1).detach()
        scr2_prob = torch.nn.functional.softmax(action_total['screen2'].view(1, -1), dim=-1).detach()
        action_prob = self.mask_unavailable(action_prob, valid_actions)
        '''
        action_id = int(torch.argmax(action_prob, dim=1)[0])
        scr1_prob = torch.argmax(scr1_prob, dim=1)
        scr2_prob = torch.argmax(scr2_prob, dim=1)
        
        '''
        for x in action_prob:
            x[91] /= 3
            x[2] *= 3
        dicision = Categorical(action_prob + 1e-5)
        action_id = dicision.sample()[0].item()
        is_valid_action = self._test_valid_action(action_id, valid_actions)
        while not is_valid_action:
            action_id = dicision.sample()[0].item()
            is_valid_action = self._test_valid_action(action_id, valid_actions)
        scr1_prob_1 = Categorical(scr1_prob + 1e-5).sample()[0].item()
        scr2_prob_2 = Categorical(scr2_prob + 1e-5).sample()[0].item()
        action_pos = [[int(scr1_prob_1 % arglist.FEAT2DSIZE), int(scr1_prob_1 // arglist.FEAT2DSIZE)], [int(scr2_prob_2 % arglist.FEAT2DSIZE), int(scr2_prob_2 // arglist.FEAT2DSIZE)]]  # (x, y)
        while action_pos[0][0] > 65 or action_pos[1][0] > 65:
            scr1_prob_1 = Categorical(scr1_prob).sample()[0].item()
            scr2_prob_2 = Categorical(scr2_prob).sample()[0].item()
            action_pos = [[int(scr1_prob_1 % arglist.FEAT2DSIZE), int(scr1_prob_1 // arglist.FEAT2DSIZE)], [int(scr2_prob_2 % arglist.FEAT2DSIZE), int(scr2_prob_2 // arglist.FEAT2DSIZE)]]  # (x, y)
        if action_id == 3:
            action_pos = [[0, 0], [63, 63]]
        args = []
        cnt = 0
        for arg in sc2_actions.FUNCTIONS[action_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(action_pos[cnt])
                cnt += 1
            else:
                args.append([0])

        action = sc2_actions.FunctionCall(action_id, args)
        '''
        if action_id == 2:
            player_relative = obs[1]["player_relative"]
            m = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            action = sc2_actions.FunctionCall(action_id, [[2], [25, 25]])
        '''
        return action

    def get_observation(self, state):
        obs_flat = state.observation['available_actions']
        obs_flat = self._onehot1d(obs_flat)
        last_actions = state.observation['last_actions']
        if len(last_actions) == 0:
            last_actions = np.array([0])
        obs = [state.observation['feature_minimap'], state.observation['feature_screen'], obs_flat, state.observation['player']]
        return obs

    def postprocess_action(self, act):
        act_categorical = np.zeros(shape=(arglist.NUM_ACTIONS, ), dtype='float32')
        act_categorical[act.function] = 1.

        act_screens = [np.zeros(shape=(1, arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), dtype='float32')] * 2
        i = 0
        for arg in act.arguments:
            if arg != [0] and arg != [2]:
                act_screens[i][0, int(arg[0]), int(arg[1])] = 1.
                i += 1

        act = {'categorical': act_categorical, 'screen1': act_screens[0], 'screen2': act_screens[1]}
        return act

    def save(self, i, r):
        torch.save(self.critic.state_dict(), "models/sc2_critic_{}_{}_{}.pth".format(map_name, i, r))
        torch.save(self.actor.state_dict(), "models/sc2_actor_{}_{}_{}.pth".format(map_name, i, r))

    def load(self, i, r):
        self.critic.load_state_dict(torch.load("models/sc2_critic_{}_{}_{}.pth".format(map_name, i, r)))
        self.actor.load_state_dict(torch.load("models/sc2_actor_{}_{}_{}.pth".format(map_name, i, r)))
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
        last_action = []
        for x in batch:

            state.append(x[0])
            action.append(x[1])
            reward.append(x[2])
            next_state.append(x[3])
            last_action.append(x[4])

        minimap = torch.FloatTensor(np.stack([x[0] for x in state], axis=0)).to(device)
        screen = torch.FloatTensor(np.stack([x[1] for x in state], axis=0)).to(device)
        valid_action = torch.FloatTensor(np.stack([x[2] for x in state], axis=0)).to(device)
        player = torch.FloatTensor(np.log(1 + np.stack([x[3] for x in state], axis=0))).to(device)
        last_action = torch.LongTensor(last_action).squeeze(-1).to(device)

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
        # n_player = torch.FloatTensor(np.stack([x[3] for x in next_state], axis = 0)).to(device)
        n_player = torch.FloatTensor(np.log(1 + np.stack([x[3] for x in next_state], axis=0))).to(device)
        n_last_action = []
        for x in action['categorical']:
            n_last_action.append(x.argmax())
        n_last_action = torch.LongTensor(n_last_action).to(device)

        with torch.no_grad():
            # n_minimap = next_state[:][0]
            # n_screen = next_state[:][1]
            # n_valid = self._onehot1d(next_state[:][2])
            # n_minimap = torch.from_numpy(n_minimap).to(device)
            # n_screen = torch.from_numpy(n_screen).to(device)
            # n_valid = torch.from_numpy(n_valid).to(device)

            next_action = self.actor_target.forward(n_minimap, n_screen, n_valid, n_player, n_last_action)
            next_action["categorical"] = torch.nn.functional.softmax(next_action['categorical'], dim=-1)
            masked_base_prob = torch.mul(next_action["categorical"], valid_action)
            next_action["categorical"] = masked_base_prob / torch.sum(masked_base_prob, dim=1).view(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target.forward(n_minimap, n_screen, n_valid, n_player, n_last_action, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            y_target_Q = reward + GAMMA * target_Q

        q1, q2 = self.critic(minimap, screen, valid_action, player, last_action, action)

        critic_loss = 1 * (F.mse_loss(q1, y_target_Q) + F.mse_loss(q2, y_target_Q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.critic_optimizer.step()

        #update actor & critic with frequency as POLICY_UPDATE
        if self.learning_count % POLICY_UPDATE == 0:
            action_c = self.actor.forward(n_minimap, n_screen, n_valid, n_player, n_last_action)
            action_c["categorical"] = torch.nn.functional.softmax(action_c['categorical'], dim=-1)  #softmax porb
            masked_base_prob = torch.mul(action_c["categorical"], valid_action)
            action_c["categorical"] = masked_base_prob / torch.sum(masked_base_prob, dim=1).view(-1, 1)  #valid action prob
            actor_loss = -self.critic.Q_value(minimap, screen, valid_action, player, last_action, action_c).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)


agent = Agent()
test_game = True
agent.load(0, 35)
if test_game is True:
    sums = 0
    T = 30
    for _ in range(T):
        r_all = 0
        state = env.reset()[0]
        last_action = [0]
        while True:
            obs = agent.get_observation(state)
            obss = deepcopy(obs)
            if map_name == 'BuildMarines':
                if obs[2][477] == 0:
                    if obs[2][42] == 0:
                        for x in range(8, 20):
                            obs[2][x] = 0
                        for x in range(269, 573):
                            obs[2][x] = 0
                        obs[2][3] = 1
                        obs[2][0] = 0
                        obs[2][2] = obss[2][2]
                        obs[2][269] = obss[2][269]
                        obs[2][264] = obss[2][264]
                        obs[2][477] = obss[2][477]
                        obs[2][490] = obss[2][490]
                    else:
                        for x in range(573):
                            obs[2][x] = 0
                        obs[2][42] = 1
                        obs[2][2] = obss[2][2]
                        obs[2][490] = obss[2][490]
                else:
                    for x in range(573):
                        obs[2][x] = 0
                    obs[2][477] = obss[2][477]
            actions = agent.select_action(obs, valid_actions=obs[2], train=True, last_action=last_action)
            next_state = env.step(actions=[actions])[0]
            next_obs = agent.get_observation(next_state)
            actions = agent.postprocess_action(actions)
            state = deepcopy(next_state)
            last_action = [actions["categorical"].argmax()]
            if state.last():
                r_all = state.observation["score_cumulative"][0]
                sums += r_all
                break
        #print('episode: {} , total_reward: {}'.format(_, r_all))
        print(r_all)
    print("sums=", sums / T)
else:
    for i in range(0, max_episode):
        if (i % 5 == 0):
            print(i)
        state = env.reset()[0]
        num = 0
        last_action = [0]
        for t in range(max_step):
            obs = agent.get_observation(state)
            obss = deepcopy(obs)
            if map_name == 'BuildMarines':
                if obs[2][477] == 0:
                    if obs[2][42] == 0:
                        for x in range(8, 20):
                            obs[2][x] = 0
                        for x in range(269, 573):
                            obs[2][x] = 0
                        obs[2][3] = 1
                        obs[2][0] = 0
                        obs[2][2] = obss[2][2]
                        obs[2][269] = obss[2][269]
                        obs[2][264] = obss[2][264]
                        obs[2][477] = obss[2][477]
                        obs[2][490] = obss[2][490]
                    else:
                        for x in range(573):
                            obs[2][x] = 0
                        obs[2][42] = 1
                        obs[2][2] = obss[2][2]
                        obs[2][490] = obss[2][490]
                else:
                    for x in range(573):
                        obs[2][x] = 0
                    obs[2][477] = obss[2][477]
            #actions = agent.select_action_cmd(state, t)
            actions = agent.select_action(obs, valid_actions=obs[2], train=True, last_action=last_action)
            next_state = env.step(actions=[actions])[0]
            next_obs = agent.get_observation(next_state)
            actions = agent.postprocess_action(actions)
            agent.memory.append((obs, actions, next_state.reward, next_obs, last_action))

            last_action = [actions["categorical"].argmax()]
            # if last_action != [331]:
            #print(last_action)

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
            last_action = [0]
            while True:
                obs = agent.get_observation(state)
                obss = deepcopy(obs)
                if map_name == 'BuildMarines':
                    if obs[2][477] == 0:
                        if obs[2][42] == 0:
                            for x in range(8, 20):
                                obs[2][x] = 0
                            for x in range(269, 573):
                                obs[2][x] = 0
                            obs[2][3] = 1
                            obs[2][0] = 0
                            obs[2][2] = obss[2][2]
                            obs[2][269] = obss[2][269]
                            obs[2][264] = obss[2][264]
                            obs[2][477] = obss[2][477]
                            obs[2][490] = obss[2][490]
                        else:
                            for x in range(573):
                                obs[2][x] = 0
                            obs[2][42] = 1
                            obs[2][2] = obss[2][2]
                            obs[2][490] = obss[2][490]
                            obs[2][477] = obss[2][477]
                    else:
                        for x in range(573):
                            obs[2][x] = 0
                        obs[2][477] = obss[2][477]
                actions = agent.select_action(obs, valid_actions=obs[2], train=True, last_action=last_action)
                next_state = env.step(actions=[actions])[0]
                next_obs = agent.get_observation(next_state)
                actions = agent.postprocess_action(actions)
                state = deepcopy(next_state)
                last_action = [actions["categorical"].argmax()]
                if state.last():
                    r_all = state.observation["score_cumulative"][0]
                    break
            if r_all >= 15:
                agent.save(i, r_all)
            print('episode: {} , copy_cmg_total_reward_no: {}'.format(i, r_all))
