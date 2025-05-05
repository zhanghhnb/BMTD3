'''

                                        基础对比论文
                “GRU-Attention based TD3 Network for Mobile Robot Navigation“
                “           基于GRU - Attention的TD3网络移动机器人导航          ”
                
                            https:// youtu.be/HkqUZSsT5a0
                            https://github.com/Barry2333/ DRL_Navigation.git
    

'''

import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from ..common.settings import POLICY_NOISE, POLICY_NOISE_CLIP, POLICY_UPDATE_FREQUENCY
from ..common.ounoise import OUNoise

from .off_policy_agent import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# https://github.com/djbyrne/TD3

import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size, num_heads = 2):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        self.state_dim = state_size
        self.action_dim = action_size
        self.hidden_size = hidden_size
        self.h = 256
        # self.hist_size = hist_size



        self.CFC1 = nn.ModuleList()
        self.CFC2 = nn.ModuleList()
        self.CFC3 = nn.ModuleList()

        # self.LSTM += [nn.GRU(256, 256, batch_first=True)]
        self.FL1 = nn.ModuleList([nn.Linear(self.state_dim, self.hidden_size), nn.ReLU()])
        self.GRU = nn.ModuleList([nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)])

        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.FL2 = nn.ModuleList([nn.Linear(self.hidden_size, self.h), nn.ReLU(), nn.Dropout()])

        self.CFC1 += [nn.Linear(self.state_dim, self.hidden_size), nn.ReLU(), nn.Dropout()]
        self.CFC2 += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Dropout()]
        self.CFC3 += [nn.Linear(3 * self.h, self.action_dim), nn.ReLU(), nn.Tanh()]


        self.apply(super().init_weights)

    def forward(self, states, his_state, his_len, visualize=False):
        # --- define forward pass here ---
        his_input = his_state
        for layer in self.FL1: his_input = layer(his_input)
        for layer in self.GRU: his_input, _ = layer(his_input)
        # 开始attention
        x = his_input
        u = torch.tanh(torch.matmul(x, self.w_omega))                               # [1, 10, 256]
        att = torch.matmul(u, self.u_omega)                                         # [1, 10, 1]
        att_score = F.softmax(att, dim=1)                                           # [1, 10, 1]
        scored_x = x * att_score                                                    # [1, 10, 256]
        feat = torch.sum(scored_x, dim=1)                                           # [1, 256]

        for layer in self.FL2: feat = layer(feat)
        his_output = feat


        cur_states = states

        for layer in self.CFC1: cur_states = layer(cur_states)
        x1 = cur_states
        # for layer in self.GRU: cur_states, _ = layer(cur_states)
        for layer in self.CFC2: cur_states = layer(cur_states)
        x2 =cur_states
        cur_output = cur_states
        final_input = torch.cat([his_output, cur_output], dim = -1) 
        # print(f"final_input shape: {final_input.shape}")
        for layer in self.CFC3:final_input = layer(final_input)
        action = final_input

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size, num_heads = 2):
        super(Critic, self).__init__(name)
        self.state_dim = state_size
        self.action_dim = action_size
        # hidden_size = 512
        self.hidden_size = hidden_size
        # self.hist_size = hist_size    
        self.h = 256

        # Q1
        # 历史数据
        self.p1 = nn.ModuleList([nn.Linear(self.state_dim, self.hidden_size), nn.ReLU()])
        self.GRU_1 = nn.ModuleList([nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)])

        self.w_omega_1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.u_omega_1 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        nn.init.uniform_(self.w_omega_1, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_1, -0.1, 0.1)

        self.p2 = nn.ModuleList([nn.Linear(self.hidden_size, self.h), nn.ReLU(), nn.Dropout()])

        # 当前数据
        self.h1 = nn.ModuleList()
        self.h2 = nn.ModuleList()
        self.h3 = nn.ModuleList()

        self.h1 += [nn.Linear(self.state_dim, self.h), nn.ReLU()]
        self.h2 += [nn.Linear(self.action_dim, self.h), nn.ReLU()]
        # self.h3 += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1)]
        self.h3 += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()]
        # 合并
        self.h4 = nn.ModuleList([nn.Linear(3 * self.h, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1)])

        # Q2
        # 历史数据
        self.p_1 = nn.ModuleList([nn.Linear(self.state_dim, self.hidden_size), nn.ReLU()])
        self.GRU_2 = nn.ModuleList([nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)])

        self.w_omega_2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.u_omega_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        nn.init.uniform_(self.w_omega_2, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_2, -0.1, 0.1)

        self.p_2 = nn.ModuleList([nn.Linear(self.hidden_size, self.h), nn.ReLU(), nn.Dropout()])

        # 当前数据
        self.h_1 = nn.ModuleList()
        self.h_2 = nn.ModuleList()
        self.h_3 = nn.ModuleList()

        self.h_1 += [nn.Linear(self.state_dim, self.h), nn.ReLU()]
        self.h_2 += [nn.Linear(self.action_dim, self.h), nn.ReLU()]
        # self.h3 += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1)]
        self.h_3 += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()]
        # 合并
        self.h_4 = nn.ModuleList([nn.Linear(3 * self.h, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1)])


        self.apply(super().init_weights)    

    def forward(self, states, actions, his_state, his_action, his_len):
        # Q1
        # 历史数据
        hs_1 = his_state
        hs_2 = his_state

        for layer in self.p1: hs_1 = layer(hs_1)
        for layer in self.p_1: hs_2 = layer(hs_2)
        for layer in self.GRU_1: hs_1, _ = layer(hs_1)
        for layer in self.GRU_2: hs_2, _ = layer(hs_2)

        # 开始attention
        x_1 = hs_1
        x_2 = hs_2
        u_1 = torch.tanh(torch.matmul(x_1, self.w_omega_1))                               # [1, 10, 256]
        u_2 = torch.tanh(torch.matmul(x_2, self.w_omega_2))
        att_1 = torch.matmul(u_1, self.u_omega_1)                                         # [1, 10, 1]
        att_2 = torch.matmul(u_2, self.u_omega_2)
        att_score_1 = F.softmax(att_1, dim=1)                                           # [1, 10, 1]
        att_score_2 = F.softmax(att_2, dim=1)
        scored_x_1 = x_1 * att_score_1                                                    # [1, 10, 256]
        scored_x_2 = x_2 * att_score_2
        feat_1 = torch.sum(scored_x_1, dim=1)
        feat_2 = torch.sum(scored_x_2, dim=1)
        for layer in self.p2: feat_1 = layer(feat_1)
        for layer in self.p_2: feat_2 = layer(feat_2)
        his_output_1 = feat_1
        his_output_2 = feat_2

        # 当前数据
        cur_states1 = states
        cur_states2 = states
        cur_actions1 = actions
        cur_actions2 = actions

        for layer in self.h1: cur_states1 = layer(cur_states1)
        for layer in self.h_1: cur_states2 = layer(cur_states2)
        for layer in self.h2: cur_actions1 = layer(cur_actions1)
        for layer in self.h_2: cur_actions2 = layer(cur_actions2)
        sa1 = torch.cat([cur_states1, cur_actions1], dim = 1)
        sa2 = torch.cat([cur_states2, cur_actions2], dim = 1)
        for layer in self.h3: sa1 = layer(sa1)
        for layer in self.h_3: sa2 = layer(sa2)
        # 合并
        final1 = torch.cat([sa1, his_output_1], dim = 1)
        final2 = torch.cat([sa2, his_output_2], dim = 1)
        for layer in self.h4: final1 = layer(final1)
        for layer in self.h_4: final2 = layer(final2)
        
        x1 = final1
        x2 = final2



        return x1, x2


class TD3(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # DRL parameters
        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        # TD3 parameters
        self.policy_noise   = POLICY_NOISE
        self.noise_clip     = POLICY_NOISE_CLIP
        self.policy_freq    = POLICY_UPDATE_FREQUENCY

        self.last_actor_loss = 0

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=50, gamma=0.9)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=50, gamma=0.9)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, his_state, his_len, is_training, step, visualize=False):


        his_state = torch.Tensor(his_state).view(1, his_state.shape[0], his_state.shape[1]).float().to(self.device)  
        his_len = torch.Tensor([his_len]).float().to(self.device)  # 转换为张量并移到设备

        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        action = self.actor(state, his_state, his_len, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        
        # print(f"state shape: {state.shape}, his_state shape: {his_state.shape}")
        # print(state)
        # print(action)
        # print(his_state)
        # print(his_len)

        return action.detach().cpu().data.numpy().tolist()


    def get_action_random(self):
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size

    def train(self, state, action, reward, state_next, done, h_state, h_action, h_next_state, h_next_action, 
          h_state_length, h_next_state_length):
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        action_next = (self.actor_target(state_next, h_next_state, h_next_state_length) + noise).clamp(-1.0, 1.0)
        Q1_next, Q2_next = self.critic_target(state_next, action_next, h_next_state, h_next_action,
                                          h_next_state_length)
        
        # print(f"Q1_next shape: {Q1_next.shape}, Q2_next shape: {Q2_next.shape}")
        
        Q_next = torch.min(Q1_next, Q2_next)

        # print(f"Q_next shape: {Q_next.shape}")

        # print(f"done shape: {done.shape}, reward shape: {reward.shape}")
        # print(done)
        # print(state)
        # print(reward)
        # print(action)



        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q1, Q2 = self.critic(state, action, h_state, h_action, h_state_length)

        # print(f"Q1 shape: {Q1.shape}, Q2 shape: {Q2.shape}")
        # print(f"Q_target shape: {Q_target.shape}")

        # loss_critic = self.loss_function(Q1, Q_target) + self.loss_function(Q2, Q_target)
        loss_critic = self.loss_function(Q1, Q_target) + self.loss_function(Q2, Q_target)





        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()
        
        self.critic_scheduler.step()

        if self.iteration % self.policy_freq == 0:
            # optimize actor
            
            loss_actor, _ = self.critic(state, self.actor(state, h_state, h_state_length), h_state, h_action, h_state_length)
            loss_actor = -1 * loss_actor.mean()
            
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()
            self.actor_scheduler.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss]



