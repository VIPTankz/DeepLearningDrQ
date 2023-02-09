import itertools
import os
import torch as T
import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent():
    def __init__(self, alpha=0.01, beta=0.01, input_dims=[8], env=None,
                 gamma=0.99, n_actions=1, max_size=1000000, tau=0.01,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=1, entr_scal=0.2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.zeta = entr_scal

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, name='actor',
                                  max_action=env.action_space.high)

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.targ_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='targ_critic_1')
        self.targ_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='targ_critic_2')

        for p in self.targ_critic_1.parameters():
            p.requires_grad = False

        for p in self.targ_critic_2.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.q_params = itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=beta)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)

        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        q1_params = self.critic_1.named_parameters()
        q2_params = self.critic_2.named_parameters()
        q1_targ_params = self.targ_critic_1.named_parameters()
        q2_targ_params = self.targ_critic_2.named_parameters()

        q1_params_dict = dict(q1_params)
        q2_params_dict = dict(q2_params)
        q1_targ_params_dict = dict(q1_targ_params)
        q2_targ_params_dict = dict(q2_targ_params)

        for name in q1_params_dict:
            q1_params_dict[name] = tau * q1_params_dict[name].clone() + \
                                   (1 - tau) * q1_targ_params_dict[name].clone()
        self.targ_critic_1.load_state_dict(q1_params_dict)

        for name in q2_params_dict:
            q2_params_dict[name] = tau * q2_params_dict[name].clone() + \
                                   (1 - tau) * q2_targ_params_dict[name].clone()

        self.targ_critic_2.load_state_dict(q1_params_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.targ_critic_1.save_checkpoint()
        self.targ_critic_2.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.targ_critic_1.load_checkpoint()
        self.targ_critic_2.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def compute_critic_loss(self, state, action, reward, new_state, done):
        r = T.tensor(reward, dtype=T.float).to(self.actor.device)
        d = T.tensor(done, dtype=T.float).to(self.actor.device)
        o2 = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        o = T.tensor(state, dtype=T.float).to(self.actor.device)
        a = T.tensor(action, dtype=T.float).to(self.actor.device)

        q1 = self.critic_1(o, a)
        q2 = self.critic_2(o, a)

        with torch.no_grad():
            a2, logp_a2 = self.actor.sample_normal(o2)

            q1_pi_targ = self.targ_critic_1(o2, a2)
            q2_pi_targ = self.targ_critic_2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.zeta * logp_a2)


        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_policy_loss(self, state):
        o = T.tensor(state, dtype=T.float).to(self.actor.device)

        a, log_a = self.actor.sample_normal(o)
        q1_pi = self.critic_1(o, a)
        q2_pi = self.critic_2(o, a)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.zeta * log_a - q_pi).mean()

        return loss_pi

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        self.q_optimizer.zero_grad()
        loss_q = self.compute_critic_loss(state, action, reward, new_state, done)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self.compute_policy_loss(state)
        loss_pi.backward()
        self.actor_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        #print("critic loss: ", loss_q)
        #print("actor loss: ", loss_pi)

        self.update_network_parameters()
