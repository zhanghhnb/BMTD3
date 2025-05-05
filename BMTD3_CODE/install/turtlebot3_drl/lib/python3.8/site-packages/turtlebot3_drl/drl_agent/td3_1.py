'''---------------------------------------------
                  对比
                原始TD3
             没有改进的算法

---------------------------------------------------'''

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
        self.fa1 = nn.Linear(state_size, hidden_size)

        self.fa2 = nn.Linear(hidden_size, hidden_size)

        self.fa3 = nn.Linear(hidden_size, action_size)


        self.apply(super().init_weights)

    def forward(self, states, his_state, his_len, visualize=False):
        # --- define forward pass here ---
        x1 = torch.relu(self.fa1(states))
        x2 = torch.relu(self.fa2(x1))
        action = torch.tanh(self.fa3(x2))

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size, num_heads = 2):
        super(Critic, self).__init__(name)
        # Q1
        # --- define layers here ---
        self.l1 = nn.Linear(state_size, int(hidden_size / 2))

        self.l2 = nn.Linear(action_size, int(hidden_size / 2))

        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        # Q2
        # --- define layers here ---
        self.l5 = nn.Linear(state_size, int(hidden_size / 2))

        self.l6 = nn.Linear(action_size, int(hidden_size / 2))

        self.l7 = nn.Linear(hidden_size, hidden_size)
        self.l8 = nn.Linear(hidden_size, 1)


        self.apply(super().init_weights)    

    def forward(self, states, actions, his_state, his_action, his_len):


        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        x1 = self.l4(x)

        xs = torch.relu(self.l5(states))
        xa = torch.relu(self.l6(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l7(x))
        x2 = self.l8(x)

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



