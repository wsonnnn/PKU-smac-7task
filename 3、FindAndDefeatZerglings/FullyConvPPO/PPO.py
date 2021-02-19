from FullyConvNet import FullyConvNet
from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from copy import deepcopy

from Preprocess import get_feature_embed
from utils import Flatten

from pysc2.env import sc2_env
from pysc2.lib import actions, features

import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spatial_features = ['screen', 'minimap', 'screen2']
nonspatial_features = [arg.name for arg in actions.TYPES if arg.name not in spatial_features]
total_args = [arg.name for arg in actions.TYPES]

minimap_channel = 101
screen_channel = 171
EPS = 1e-6
nonspatial = 11
n_actions = len(actions.FUNCTIONS)

BASELINE_SCALE = 0.5
ENTROPY_SCALE = 1e-3
save_std = 30

class ActorCritic(nn.Module):
    def __init__(self, minimap_channel, screen_channel, nonspatial, n_actions):

        super(ActorCritic, self).__init__()

        self.gamma = 0.99
        self.LAMBDA= 0.95

        self.latent_net = FullyConvNet(minimap_channel, screen_channel, nonspatial, n_actions)

        self.base_policy_logit = nn.Sequential(nn.Linear(256, 256),
                                               nn.ReLU(),
                                               nn.Linear(256, n_actions),
                                               nn.Softmax(dim=-1))


        self.mini_embed = get_feature_embed(features.MINIMAP_FEATURES)
        self.screen_embed = get_feature_embed(features.SCREEN_FEATURES)

        spatial_args = {"screen": 0, "minimap": 0, "screen2": 0}
        
        # calculate spatial_args dict
        for (key, value) in spatial_args.items():
            spatial_args[key] = nn.Sequential(nn.ConvTranspose2d(96, 1, 1, 1),
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
                nonspatial[arg.name] = nn.Sequential(nn.Linear(256, 256),
                                                     nn.ReLU(),
                                                     nn.Linear(256, size),
                                                     nn.Softmax(dim=-1))
                mask = torch.FloatTensor(np.eye(size))
                temp = torch.cat([torch.zeros(size).view(1, -1), mask], dim=0).to(device)
                self.nonspatial_mask[arg.name] = temp
                
        self.nonspatial_args = nn.ModuleDict(nonspatial)

        self.value_net = nn.Sequential(nn.Linear(256, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 1))

    def encode_screen(self, screen):
        layers = []
        for i in range(len(features.SCREEN_FEATURES)):
            if features.SCREEN_FEATURES[i].type == features.FeatureType.CATEGORICAL:
                name = features.SCREEN_FEATURES[i].name
                layer = screen[:, i].type(torch.int64).to(device)
                layer = self.screen_embed[name](layer).permute(0, 3, 1, 2)
            else:
                layer = torch.log(1.0 + screen[:,i:i + 1]).to(device)
                # layer = screen[:, i:i+1] / features.SCREEN_FEATURES[i].scale
                # layer = screen[:, i:i + 1]
            layers.append(layer)
        return torch.cat(layers, dim=1).to(device)

    def encode_minimap(self, minimap):
        layers = []
        for i in range(len(features.MINIMAP_FEATURES)):
            if features.MINIMAP_FEATURES[i].type == features.FeatureType.CATEGORICAL:
                name = features.MINIMAP_FEATURES[i].name
                layer = minimap[:, i].type(torch.int64).to(device)
                if debug:
                    print("categorical shape", layer.shape)
                layer = self.mini_embed[name](layer).permute(0, 3, 1, 2)
            else:
                if debug:
                    print("Minimap shape", minimap.shape)
                    print("non categorical shape", minimap[:, i:i + 1].shape)
                layer = torch.log(1.0 + minimap[:,i:i+1]).to(device)
                # layer = minimap[:,i:i+1] / features.MINIMAP_FEATURES[i].scale
                # layer = minimap[:, i:i + 1]
            layers.append(layer.to(device))
        return torch.cat(layers, dim=1).to(device)

    def forward(self, minimaps, screens, players, last_actions):
        """
        return the base_action_prob, spatial_prob (as dict), nonspatial_prob (as dict)
        """
        minimaps = self.encode_minimap(minimaps)
        screens = self.encode_screen(screens)
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

        base_prob, spatial_prob, nonspatial_prob, values = self.forward(minimap, screen, player, last_action)

        base_prob = torch.clamp(base_prob, 1e-6, 1)
        masked_base_prob = base_prob * valid_action
        sum_prob = masked_base_prob.sum(-1).view(-1, 1)
        masked_base_prob = masked_base_prob / sum_prob

        arg_list, spatial_args, nonspatial_args = [], [], []

        dice = torch.distributions.Categorical

        if israndom:
            base_action = dice(masked_base_prob.detach()).sample().item()
        else:
            base_action = torch.argmax(masked_base_prob, dim=1).item()

        required_args = [arg.name for arg in actions.FUNCTIONS[base_action].args]

        picked_up = {}
        for name in required_args:
            if name in spatial_features:
                if israndom:
                    pos = dice(spatial_prob[name]).sample().item()
                else:
                    pos = torch.argmax(spatial_prob[name], dim=1).item()
                arg_list.append([pos // 64, pos % 64])
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

        actor_logits = torch.log(torch.clamp(masked_base_prob, EPS, 1.0))[0][base_action].view(-1)
        for i in range(len(spatial_features)):
            name = spatial_features[i]
            actor_logits = actor_logits + (torch.log(torch.clamp(spatial_prob[name], EPS, 1.0)) * self.spatial_mask[spatial_args[i] + 1]).sum(-1)
        
        for i in range(len(nonspatial_features)):
            name = nonspatial_features[i]
            actor_logits = actor_logits + (torch.log(torch.clamp(nonspatial_prob[name], EPS, 1.0)) * self.nonspatial_mask[name][nonspatial_args[i] + 1]).sum(-1)
        
        func_action = actions.FunctionCall(base_action, arg_list)

        return func_action, base_action, spatial_args, nonspatial_args, actor_logits

    def loss_fn(self, minimaps, screens, players, last_actions, valid_actions, base_action, spatial_args, nonspatial_args, rewards, n_minimap, n_screen, n_player, n_laction):

        # a batch is [trace1, trace2, trace3, ... trace_m], m is batch_size, we let m = 8
        batch_size = len(base_action)

        base_prob, spatial_prob, nonspatial_prob, values = self.forward(minimaps, screens, players, last_actions)

        with torch.no_grad():
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
                             torch.log(torch.clamp(spatial_prob[feat], EPS, 1.0))).sum(-1)
        
        for feat in nonspatial_features:
            entropy_loss = entropy_loss + (nonspatial_prob[feat] * \
                             torch.log(torch.clamp(nonspatial_prob[feat], EPS, 1.0))).sum(-1)
        
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


map_names = ["MoveToBeacon", "CollectMineralShards", "FindAndDefeatZerglings", "DefeatRoaches", "DefeatZerglingsAndBanelings", "CollectMineralsAndGas", "BuildMarines"]
map_name = map_names[2]
minimap_channel = 101
screen_channel = 171
nonspatial_dim = 11
n_actions = len(actions.FUNCTIONS)
lr = 1e-3
MAX_GRAD_NORM = 100


class PPO(object):
    def __init__(self):
        self.map_name = map_name
        self.net = ActorCritic(minimap_channel, screen_channel, nonspatial_dim, n_actions).to(device)
        self.env = self.init_env(map_name)
        self.update_timestep = 200
        self.max_episode = 15000
        self.max_step = 20000
        self.memory = []
        self.optimizer = optim.Adam(self.net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-8)
        self.buffer_size = 8000
        self.batch_size = 40
        self.max_grad_norm = MAX_GRAD_NORM
        self.ppo_epoch = 10
        self.gamma = 0.99
        self.clip_param = 0.2

    def init_env(self, map_name):
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        flags.DEFINE_bool("render", False, "Whether to render with pygame.")
        agent_format = sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(
            screen=(64, 64),
            minimap=(64, 64),
        ))

        env = sc2_env.SC2Env(map_name=map_name, step_mul=32, visualize=False, game_steps_per_episode=10000, agent_interface_format=[agent_format], players=[sc2_env.Agent(sc2_env.Race.terran)])

        return env

    def _onehot1d(self, x):
        y = np.zeros((n_actions, ), dtype='float32')
        y[x] = 1.
        return y

    #def normal_logprob(x, mean, logstd, std = None):
    #    if std is None:
    #        std = torch.exp(logstd)
    #    std_sq = std.pow(2)
    #    log_prob = -0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
    #    return log_prob.sum(1)
    def select_action(self, obs, valid_actions, last_action, israndom=True):

        minimap = obs[0].astype('float32')
        screen = obs[1].astype('float32')

        minimap = torch.FloatTensor(np.expand_dims(minimap, axis=0)).to(device)
        screen = torch.FloatTensor(np.expand_dims(screen, axis=0)).to(device)
        valid_actions = torch.from_numpy(valid_actions).unsqueeze(0).to(device)
        player = torch.FloatTensor(obs[3]).unsqueeze(0).to(device)
        last_action = torch.LongTensor(last_action).to(device)

        act_func, base_action, spatial_args, nonspatial_args, actor_logits = self.net.choose_action(minimap, screen, player, last_action, valid_actions, israndom=israndom)

        return act_func, base_action, spatial_args, nonspatial_args, actor_logits

    def get_observation(self, state):
        obs_flat = state.observation['available_actions']
        obs_flat = self._onehot1d(obs_flat)
        last_actions = state.observation['last_actions']
        if len(last_actions) == 0:
            last_actions = np.array([0])
        obs = [state.observation['feature_minimap'], state.observation['feature_screen'], obs_flat, state.observation['player']]
        return obs

    def save(self, i, r):
        torch.save(self.net.state_dict(), "modelsPPO/sc2_actorcritic_{}_{}_{:.4f}.pth".format(self.map_name, int(i), r))

    def load(self, i, r):
        self.net.load_state_dict(torch.load("modelsPPO/sc2_actorcritic_{}_{}_{:.4f}.pth".format(self.map_name, int(i), r), map_location=device))

    def learn(self, update=True):

        state = []
        action = []
        reward = []
        last_action = []
        spatial_args = []
        nonspatial_args = []
        actor_logits = []
        terminal = []
        n_state = []
        batch = self.memory

        for x in batch:
            state.append(x[0])
            action.append(x[1])
            reward.append(x[2])
            last_action.append(x[3])
            spatial_args.append(x[4])
            nonspatial_args.append(x[5])
            terminal.append(x[6])
            actor_logits.append(x[7])
            n_state.append(x[8])

        minimap = torch.FloatTensor(np.stack([x[0] for x in state], axis=0)).to(device)
        screen = torch.FloatTensor(np.stack([x[1] for x in state], axis=0)).to(device)
        valid_action = torch.FloatTensor(np.stack([x[2] for x in state], axis=0)).to(device)
        player = torch.FloatTensor(np.log(1 + np.stack([x[3] for x in state], axis=0))).to(device)

        n_minimap = torch.FloatTensor(np.stack([x[0] for x in n_state], axis=0)).to(device)
        n_screen = torch.FloatTensor(np.stack([x[1] for x in n_state], axis=0)).to(device)
        n_valid_action = torch.FloatTensor(np.stack([x[2] for x in n_state], axis=0)).to(device)
        n_player = torch.FloatTensor(np.log(1 + np.stack([x[3] for x in n_state], axis=0))).to(device)

        last_action = torch.LongTensor(last_action).squeeze(-1).to(device)
        base_actions = torch.LongTensor(action).to(device)
        spatial_args = torch.LongTensor(np.stack(spatial_args, axis=1)).to(device)
        nonspatial_args = torch.LongTensor(np.stack(nonspatial_args, axis=1)).to(device)
        old_actor_logits = torch.cat(actor_logits, dim=0).to(device)

        with torch.no_grad():
            lminimap = minimap[-1:]
            lscreen = screen[-1:]
            lplayer = player[-1:]
            laction = base_actions[-1:]
            base_, spatial_, nonspatial_, boostrap = self.net.forward(lminimap, lscreen, lplayer, laction)


        R = boostrap[0].item()

        dis_R = []
        for r in reward[::-1]:
            R = r + R * self.gamma
            dis_R.insert(0, R)
        
        dis_R = torch.FloatTensor(dis_R).to(device)

        #with torch.no_grad()
        #TODO能否直接传base_action,保存prob时对梯度计算的影响？
        #   _, _, _, n_values = self.net(n_minimap, n_screen, n_player, base_actions)
        #    target_values = reward + GAMMA * n_values

        sample_size = min(minimap.shape[0], self.buffer_size)
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(sample_size)), self.batch_size, False):
                
                dis_index = dis_R[index].view(-1, 1)

                #计算一下sample中的action分布
                base_prob, spatial_prob, nonspatial_prob, value_index = self.net.forward(minimap[index], screen[index], player[index], last_action[index])
                delta = dis_index - value_index.view(-1)

                critic_loss = delta.pow(2)
                advantage = delta.detach()

                masked_base_prob = base_prob * valid_action[index]

                sum_prob = masked_base_prob.sum(-1).view(-1, 1)
                masked_base_prob = masked_base_prob / sum_prob

                actor_logits = torch.log(torch.clamp(masked_base_prob, EPS, 1.0))[np.arange(len(index)), base_actions[index]]

                #TODO 这里不是很确定。。。。。 在进行log prob计算时，去掉了很多clamp，不知道应不应该去掉。。。
                for i in range(len(spatial_features)):
                    name = spatial_features[i]
                    actor_logits = actor_logits + (torch.log(torch.clamp(spatial_prob[name], EPS, 1.0)) * \
                                                   self.net.spatial_mask[spatial_args[i][index] + 1]).sum(-1)
                
                for i in range(len(nonspatial_features)):
                    name = nonspatial_features[i]
                    actor_logits = actor_logits + (torch.log(torch.clamp(nonspatial_prob[name], EPS, 1.0)) * \
                                                    self.net.nonspatial_mask[name][nonspatial_args[i][index] + 1]).sum(-1)
                

                ratio = torch.exp(actor_logits - old_actor_logits[index])

                ratio2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)

                action_loss = -1 * torch.min(ratio, ratio2) * advantage

                entropy_loss = torch.sum(masked_base_prob * torch.log(torch.clamp(masked_base_prob, EPS, 1.0)), dim=-1)

                for feat in spatial_features:
                    entropy_loss = entropy_loss + (spatial_prob[feat] * \
                             torch.log(torch.clamp(spatial_prob[feat], EPS, 1.0))).sum(-1)
                
                for feat in nonspatial_features:
                    entropy_loss = entropy_loss + (nonspatial_prob[feat] * \
                             torch.log(torch.clamp(nonspatial_prob[feat], EPS, 1.0))).sum(-1)
                
                loss = (action_loss + BASELINE_SCALE * critic_loss + ENTROPY_SCALE * entropy_loss).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def run(self):
        time_step = 0
        seed = 1234
        # self.env.seed(seed)
        torch.manual_seed(seed)

        self.memory = []
        for i_episode in range(self.max_episode):
            state = self.env.reset()[0]
            last_action = [0]
            total_reward = 0.0
    
            zerg_count, dead_mar = 0, 0
            for t in range(self.max_step):
                time_step += 1
                obs = self.get_observation(state)
                func_act, base_act, spatial_args, nonspatial_args, actor_logits = self.select_action(obs, valid_actions=obs[2], last_action=last_action)

                next_state = self.env.step(actions=[func_act])[0]

                zerg_count += (next_state.reward > 0)
                dead_mar += (next_state.reward < 0)

                next_obs = self.get_observation(next_state)

                reward_shaping = 0
                if map_name == "FindAndDefeatZerglings":
                    reward_shaping = (next_state.reward > 0) * zerg_count/30
                    reward_shaping = (next_state.reward < 0) * dead_mar/6

                self.memory.append((obs, base_act, next_state.reward + reward_shaping, last_action, spatial_args, nonspatial_args, state.last(), actor_logits.detach(), next_obs))
                if time_step % self.update_timestep == 0 or state.last():
                    #print("episode: ",i_episode,"  reward: ",total_reward)
                    self.learn(update=True)
                    self.memory = []
                    time_step = 0
                if state.last():
                    total_reward = state.observation["score_cumulative"][0]
                    zerg_count = 0
                    dead_mar = 0
                    print("episode: ", i_episode, "  reward: ", total_reward)
                    '''
                    if total_reward >= save_std:
                        self.save(i_episode, total_reward)
                    '''
                    break
                else:
                    last_action = [base_act]
                    state = deepcopy(next_state)

            if (i_episode + 1) % 30 == 0:
                r_all = 0
                for j in range(6):
                    state = self.env.reset()[0]
                    last_action = [0]
                    while True:
                        obs = self.get_observation(state)

                        func_act, base_act, spatial_args, nonspatial_args, _ = self.select_action(obs, valid_actions=obs[2], last_action=last_action, israndom=True)
                        next_state = self.env.step(actions=[func_act])[0]
                        next_obs = self.get_observation(next_state)

                        state = deepcopy(next_state)
                        last_action = [base_act]
                        if state.last():
                            r_all += state.observation["score_cumulative"][0]
                            break
                r_all /= 6

                self.save(i_episode, r_all)
                print('Hello from test, Episode: {}, Reward for tests: {}'.format(i_episode, r_all))

        self.env.close()
        
    def test(self, test_episode=100):
        total_r = 0
        for _ in range(test_episode):
            r_all = 0
            state = self.env.reset()[0]
            last_action = [0]
            while True:
                obs = self.get_observation(state)

                func_act, base_act, spatial_args, nonspatial_args, aha = self.select_action(obs, valid_actions=obs[2], last_action=last_action)
                next_state = self.env.step(actions=[func_act])[0]
                next_obs = self.get_observation(next_state)
 
                state = deepcopy(next_state)
                if state.last():
                    r_all = state.observation["score_cumulative"][0]
                    total_r += state.observation["score_cumulative"][0]
                    
                    print(r_all)
                    break

        print("average reward {}".format(total_r/test_episode))

if __name__ == "__main__":
    ppo = PPO()
    ppo.run()