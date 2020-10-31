from itertools import count
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pickle
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Enviornment.Env3DQNFixCount as env
import Enviornment.Env3DQNFixStorage as env2
from Model import PR_Buffer as BufferX
from Model import ReplyBuffer as Buffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
directory = './exp' + script_name + "mview" + '/'


class NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            # nn.Sigmoid()
        )

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        actions = self.layers(state)
        return actions


class DNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        # self.l3 = nn.Linear(256, 128)
        self.adv1 = nn.Linear(256, 256)
        self.adv2 = nn.Linear(256, action_dim)
        self.val1 = nn.Linear(256, 64)
        self.val2 = nn.Linear(64, 1)

    def _init_weights(self):
        self.l1.weight.data.normal_(0.0, 1e-2)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l2.weight.data.normal_(0.0, 1e-2)
        self.l2.weight.data.uniform_(-0.1, 0.1)
        # self.l3.weight.data.normal_(0.0, 1e-2)
        # self.l3.weight.data.uniform_(-0.1, 0.1)
        self.adv1.weight.data.normal_(0.0, 1e-2)
        self.adv1.weight.data.uniform_(-0.1, 0.1)
        self.adv2.weight.data.normal_(0.0, 1e-2)
        self.adv2.weight.data.uniform_(-0.1, 0.1)
        self.val1.weight.data.normal_(0.0, 1e-2)
        self.val1.weight.data.uniform_(-0.1, 0.1)
        self.val2.weight.data.normal_(0.0, 1e-2)
        self.val2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        # actions = self.layers(state)
        x = self.relu(self.l1(state))
        x = self.relu(self.l2(x))
        # x = self.relu(self.l3(x))
        adv = self.relu(self.adv1(x))
        val = self.relu(self.val1(x))
        adv = self.relu(self.adv2(adv))
        val = self.relu(self.val2(val))
        qvals = val + (adv-adv.mean())
        return qvals


class DQN:
    def __init__(self, workload, action, index_mode, conf, is_dnn, is_ps, is_double, a):
        self.conf = conf
        self.workload = workload
        self.action = action
        self.index_mode = index_mode

        self.state_dim = len(workload) + len(action)
        # we do not need another flag to indicate 'deletion/creation'
        self.action_dim = len(action)
        self.is_ps = is_ps
        self.is_double = is_double
        if is_dnn:
            self.actor = DNN(self.state_dim, self.action_dim).to(device)
            self.actor_target = DNN(self.state_dim, self.action_dim).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        else:
            self.actor = NN(self.state_dim, self.action_dim).to(device)
            self.actor_target = NN(self.state_dim, self.action_dim).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), conf['LR']) #optim.SGD(self.actor.parameters(), lr=self.conf['LR'], momentum=0.9)#

        self.replay_buffer = None
        # some monitor information
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.index_mode = index_mode
        self.actor_loss_trace = list()

        # environment
        self.envx = env.Env(self.workload, self.action, self.index_mode,a)

        # store the parameters
        self.writer = SummaryWriter(directory)

        self.learn_step_counter = 0

    def select_action(self, t, state):
        if not self.replay_buffer.can_update():
            action = np.random.randint(0, len(self.action))
            action = [action]
            return action
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= self.conf['EPISILO']: # *(1 - math.pow(0.5, t/50)):  #*(t/MAX_STEP):  # greedy policy
            action_value = self.actor.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            return action
        else:  # random policy
            action = np.random.randint(0, len(self.action))
            action = [action]
            return action

    def _sample(self):
        batch, idx = self.replay_buffer.sample(self.conf['BATCH_SIZE'])
        # state, next_state, action, reward, np.float(done))
        # batch = self.replay_memory.sample(self.batch_size)
        x, y, u, r, d = [], [], [], [], []
        for _b in batch:
            x.append(np.array(_b[0], copy=False))
            y.append(np.array(_b[1], copy=False))
            u.append(np.array(_b[2], copy=False))
            r.append(np.array(_b[3], copy=False))
            d.append(np.array(_b[4], copy=False))
        return idx, np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.conf['LR'] * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, ep):
        if self.learn_step_counter % self.conf['Q_ITERATION'] == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1
        # self.adjust_learning_rate(self.actor_optimizer, ep)
        for it in range(self.conf['U_ITERATION']):
            idxs = None
            if self.is_ps:
                idxs, x, y, u, r, d = self._sample()
            else:
                x, y, u, r, d = self.replay_buffer.sample(self.conf['BATCH_SIZE'])
            state = torch.FloatTensor(x).to(device)
            action = torch.LongTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            q_eval = self.actor(state).gather(1, action)

            if self.is_double:
                next_batch = self.actor(next_state)
                nx = next_batch.max(1)[1][:,None]
                # max_act4next = np.argmax(q_eval_next, axis=1)
                q_next = self.actor_target(next_state)
                qx = q_next.gather(1,nx)
                # q_target = reward + (1 - done) * self.conf['GAMMA'] * qx.max(1)[0].view(self.conf['BATCH_SIZE'], 1)
                q_target = reward + (1 - done) * self.conf['GAMMA'] * qx
            else:
                q_next = self.actor_target(next_state).detach()
                q_target = reward + (1-done)*self.conf['GAMMA'] * q_next.max(1)[0].view(self.conf['BATCH_SIZE'], 1)
            actor_loss = F.mse_loss(q_eval, q_target)
            error = torch.abs(q_eval - q_target).data.numpy()
            if self.is_ps:
                for i in range(self.conf['BATCH_SIZE']):
                    idx = idxs[i]
                    self.replay_buffer.update(idx, error[i][0])

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.actor_loss_trace.append(actor_loss.data.item())
            # for item in self.actor.named_parameters():
                # h = item[1].register_hook(lambda grad: print(grad))

    def save(self):
        torch.save(self.actor_target.state_dict(), directory + 'dqn.pth')
        print('====== Model Saved ======')

    def load(self):
        print('====== Model Loaded ======')
        self.actor.load_state_dict(torch.load(directory + 'dqn.pth'))

    def train(self, load, __x):
        if load:
            self.load()
        is_first = True
        # check whether have an index will 90% improvement
        self.envx.max_count = __x
        pre_create = self.envx.checkout()
        if not (pre_create is None):
            print(pre_create)
            if len(pre_create) >= __x:
                return pre_create[:__x]
        if self.is_ps:
            self.replay_buffer = BufferX.PrioritizedReplayMemory(self.conf['MEMORY_CAPACITY'], min(self.conf['LEARNING_START'],200*self.envx.max_count))
        else:
            self.replay_buffer = Buffer.ReplayBuffer(self.conf['MEMORY_CAPACITY'], min(self.conf['LEARNING_START'],200*self.envx.max_count))
        current_best_reward = 0
        current_best_index = None
        rewards = []
        __how_m = self.envx.max_count
        for ep in range(self.conf['EPISODES']):
            print("======"+str(ep)+"=====")
            state = self.envx.reset()

            t_r = 0
            _state = []
            _next_state = []
            _action = []
            _reward = []
            _done = []
            for t in count():
                action = self.select_action(ep, state)
                # print(action)
                next_state, reward, done = self.envx.step(action)
                # print(reward)
                t_r += reward
                '''_state.append(state)
                _next_state.append(next_state)
                _action.append(action)
                _reward.append(reward)
                _done.append(np.float(done))'''
                if self.is_ps:
                    self.replay_buffer.add(1.0, (state, next_state, action, reward, np.float(done)))
                else:
                    self.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                # if self.replay_buffer.can_update():
                #    self.update()
                if done:
                    '''for i in range(len(_state)):
                        if self.isPS:
                            self.replay_buffer.add(1.0, (_state[i], _next_state[i], _action[i], _reward[i]+t_r/__how_m, _done[i]))
                        else:
                            self.replay_buffer.push((_state[i], _next_state[i], _action[i], _reward[i]+t_r/__how_m, _done[i]))'''
                    if ep > (self.conf['EPISODES']-100) and t_r > current_best_reward:
                        current_best_reward = t_r
                        current_best_index = self.envx.index_trace_overall[-1]
                        # print(current_best_index)
                    # self.replay_buffer.add(1.0, (state, next_state, action, reward, np.float(done)))
                    if self.replay_buffer.can_update() and ep % 5 == 0:
                        self.update(ep)
                    break
                state = next_state
            rewards.append(t_r)
        self.save()
        plt.figure(__x)
        x = range(len(self.envx.cost_trace_overall))
        y2 = [math.log(a, 10) for a in self.envx.cost_trace_overall]
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "freq.png", dpi=120)
        plt.clf()
        plt.close()
        plt.figure(__x+1)
        x = range(len(rewards))
        y2 = rewards
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "rewardfreq.png", dpi=120)
        plt.clf()
        plt.close()
        # return self.envx.index_trace_overall[-1]
        with open('{}.pickles'.format(self.conf['NAME']), 'wb') as f:
            pickle.dump(self.envx.cost_trace_overall, f, protocol=0)
        print(current_best_reward)
        return current_best_index


