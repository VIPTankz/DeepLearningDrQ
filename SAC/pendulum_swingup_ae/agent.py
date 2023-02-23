from torch import optim
from torchvision import transforms
from buffer import ReplayBuffer
from networks import PolicyNetwork, QNetwork
import torch.nn.functional as F
import torch
import numpy as np
import yaml

class SAC_Agent:
    def __init__(self, cfg):
        self.obs_shape = (cfg["frame_stack"], cfg["image_size"], cfg["image_size"])
        self.action_shape = (cfg["action_size"],)
        self.chkpt_dir = cfg["agent"]["checkpoint_dir"]
        self.feature_dim = (cfg["agent"]["params"]["feature_dim"],)
        self.lr_pi = cfg["agent"]["params"]["policy_lr"]
        self.lr_q = cfg["agent"]["params"]["critic_lr"]
        self.discount = cfg["agent"]["params"]["discount"]
        self.buffer_limit = cfg["agent"]["params"]["buffer_limit"]
        self.batch_size = cfg["batch_size"]
        self.init_alpha = cfg["agent"]["params"]["init_temperature"]
        self.lr_alpha = cfg["agent"]["params"]["lr_alpha"]
        self.target_entropy = -self.action_shape[0]
        self.device = cfg["agent"]["params"]["device"]
        self.critic_update_freq = cfg["agent"]["params"]["critic_target_update_frequency"]
        self.actor_update_freq = cfg["agent"]["params"]["actor_update_frequency"]
        self.soft_target_update = cfg["agent"]["params"]["critic_soft_update_rate"]

        self.log_std_min = cfg["agent"]["params"]["log_std_min"]
        self.log_std_max = cfg["agent"]["params"]["log_std_max"]
        self.max_action = cfg["agent"]["params"]["max_action"]
        self.min_action = cfg["agent"]["params"]["min_action"]


        self.train_counter = 0
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.memory = ReplayBuffer(self.buffer_limit, self.obs_shape, self.action_shape, self.device)

        self.PI = PolicyNetwork(self.obs_shape, self.feature_dim, self.action_shape, self.lr_pi,
                                self.log_std_min, self.log_std_max, self.max_action, self.min_action).to(self.device)
        self.Q1 = QNetwork(self.obs_shape, self.feature_dim, self.action_shape, self.lr_q).to(self.device)
        self.Q2 = QNetwork(self.obs_shape, self.feature_dim, self.action_shape, self.lr_q).to(self.device)
        self.Q1_target = QNetwork(self.obs_shape, self.feature_dim, self.action_shape, self.lr_q).to(self.device)
        self.Q2_target = QNetwork(self.obs_shape, self.feature_dim, self.action_shape, self.lr_q).to(self.device)

        #self.PI.encoder.copy_weights_from(self.Q1.encoder)
        #self.Q2.encoder.copy_weights_from(self.Q1.encoder)
        #self.Q1_target.encoder.copy_weights_from(self.Q1.encoder)
        #self.Q2_target.encoder.copy_weights_from(self.Q1.encoder)

        self.Q1_opt = optim.Adam(self.Q1.parameters(), lr=self.lr_q)
        self.Q2_opt = optim.Adam(self.Q2.parameters(), lr=self.lr_q)
        self.PI_opt = optim.Adam(self.PI.parameters(), lr=self.lr_pi)


    def choose_action(self, obs):
        with torch.no_grad():
            action, log_prob = self.PI.sample(obs.to(self.device))
        return action, log_prob

    def calc_target(self, mini_batch):
        o, a, r, o_, d = mini_batch
        with torch.no_grad():
            a_, lp_ = self.PI.sample(o)
            entropy = - self.log_alpha.exp() * lp_
            q1_t, q2_t = self.Q1_target(o_, a_), self.Q2_target(o_, a_)
            q_t = torch.min(q1_t, q2_t)
            targ = r + self.discount * d * (q_t + entropy)
        return targ

    def learn(self):
        self.train_counter += 1
        o, a, r, o_, d = self.memory.sample(self.batch_size)
        td_target = self.calc_target((o, a, r, o_, d))

        ## Q1 train ##
        self.Q1_opt.zero_grad()
        out = self.Q1(o, a)
        q1_loss = F.mse_loss(self.Q1(o,a), td_target)
        q1_loss.mean().backward()
        self.Q1_opt.step()

        ## Q2 train ##
        self.Q2_opt.zero_grad()
        q2_loss = F.smooth_l1_loss(self.Q2(o, a), td_target)
        q2_loss.mean().backward()
        self.Q2_opt.step()

        ## policy train ##
        if self.train_counter % self.actor_update_freq == 0:
            a, lp = self.PI.sample(o)
            entropy = -self.log_alpha.exp() * lp
            q1, q2 = self.Q1(o, a), self.Q2(o, a)
            q = torch.min(q1, q2)
            pi_loss = -(q + entropy)
            self.PI_opt.zero_grad()
            pi_loss.mean().backward()
            self.PI_opt.step()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (lp + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        if self.train_counter % self.critic_update_freq == 0:
            for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
                param_target.data.copy_(param_target.data * (1.0 - self.soft_target_update) + param.data * self.soft_target_update)
            for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
                param_target.data.copy_(param_target.data * (1.0 - self.soft_target_update) + param.data * self.soft_target_update)
    def save_models(self):
        torch.save(self.PI.state_dict(), self.chkpt_dir + '/saved_models/sac/policy')
        torch.save(self.Q1.state_dict(), self.chkpt_dir + '/saved_models/sac/Q1')
        torch.save(self.Q2.state_dict(), self.chkpt_dir + '/saved_models/sac/Q2')
        torch.save(self.Q1_target.state_dict(), self.chkpt_dir + '/saved_models/sac/Q1_target')
        torch.save(self.Q2.state_dict(), self.chkpt_dir + '/saved_models/sac/Q2_target')
        self.memory.save(self.chkpt_dir + '/saved_replay_buffer/')

    def load_models(self):
        self.PI.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/policy'))
        self.Q1.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q1'))
        self.Q2.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q2'))
        self.Q1_target.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q1_target'))
        self.Q2_target.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q2_target'))
        self.memory.load(self.chkpt_dir + '/saved_replay_buffer/')


