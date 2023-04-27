import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from ExperienceReplay import ExperienceReplay
import numpy as np
from collections import deque
import kornia.augmentation as aug
from torchvision.utils import save_image

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.div(observation,255)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1,64 * 7 * 7)
        observation = F.relu(self.fc1(observation))
        V = self.V(observation)
        A = self.A(observation)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class EpsilonGreedy():
    def __init__(self):
        self.eps = 1.0
        self.steps = 5000
        self.eps_final = 0.1

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps,self.eps_final)

class Agent():
    def __init__(self, discount, lr, input_dims,batch_size,n_actions,
                 max_mem_size = 1000000, replace=1):

        self.epsilon = EpsilonGreedy()
        self.lr = lr
        self.num_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.num_actions)]
        self.learn_step_counter = 0
        self.chkpt_dir = 'tmp/dueling_ddqn'
        self.min_sampling_size = 1600
        self.n = 10
        self.gamma = discount
        self.eval_mode = False

        self.memory = ExperienceReplay(input_dims,max_mem_size,self.batch_size)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.num_actions,
                                   input_dims=self.input_dims,
                                   name='lunar_lander_dueling_ddqn_q_eval',
                                   chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.num_actions,
                                   input_dims=self.input_dims,
                                   name='lunar_lander_dueling_ddqn_q_next',
                                   chkpt_dir=self.chkpt_dir)

        self.n_states = deque([],self.n)
        self.n_rewards = deque([],self.n)
        self.n_actions = deque([],self.n)

        #self.crop = torchvision.transforms.RandomCrop((84,84),padding=4)
        #transform = torchvision.transforms.RandomCrop((84,84),padding=4)
        #self.crop = torchvision.transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))
        self.random_shift = nn.Sequential(nn.ReplicationPad2d(4),aug.RandomCrop((84, 84)))



        self.K = 2
        self.M = 2

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.eps or self.eval_mode:
            state = T.tensor(np.array([observation]),dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.n_step(state, action, reward, state_, done)
        #self.memory.store_transition(state, action, reward, state_, done)

    def n_step(self, state, action, reward, state_, done):
        self.n_states.append(state)
        self.n_rewards.append(reward)
        self.n_actions.append(action)

        if len(self.n_states) == self.n:
            fin_reward = 0
            for i in range(self.n):
                fin_reward += self.n_rewards.index(i) * (self.gamma ** i)
            self.memory.store_transition(self.n_states.index(0),self.n_actions.index(0),fin_reward, \
                                         state_,done)

        if done:
            self.n_states = deque([], self.n)
            self.n_rewards = deque([], self.n)
            self.n_actions = deque([], self.n)

        """if done:
            self.n_states.popleft()
            self.n_rewards.popleft()
            self.n_actions.popleft()

            while len(self.n_states) != 0:

                fin_reward = 0
                for i in range(len(self.n_states)):
                    fin_reward += self.n_rewards.index(i) * (self.gamma ** i)
                self.memory.store_transition(self.n_states.index(0), self.n_actions.index(0), fin_reward, \
                                             state_, done)

                self.n_states.popleft()
                self.n_rewards.popleft()
                self.n_actions.popleft()"""

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def augment(self,images):
        #[K,stack_frames,X,Y] ie [2,4,84,84]
        x = torch.stack([self.crop(i) for i in images],dim=1)

        #save_image(x[0][0][0] / 255,"test0.png")
        #save_image(x[0][0][1] / 255,"test1.png")
        return x

    def learn(self):
        if self.memory.mem_cntr < self.min_sampling_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states,actions,rewards,new_states,dones = self.memory.sample_memory()

        states = T.tensor(states).to(self.q_eval.device)
        rewards = T.tensor(rewards).to(self.q_eval.device)
        dones = T.tensor(dones).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        states_ = T.tensor(new_states).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        states_aug = self.random_shift(states)
        states_aug_ = self.random_shift(states_)

        # K = 2
        V_s, A_s = (self.q_eval.forward(states) + self.q_eval.forward(states_aug)) / 2
        V_s_, A_s_ = (self.q_eval.forward(states_) + self.q_eval.forward(states_aug_)) / 2

        """
        statesM = states.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
        V_s, A_s = self.q_eval.forward(statesM)
        A_s = A_s.reshape((self.M,self.batch_size,self.num_actions)).mean(dim=0)
        V_s = V_s.reshape((self.M,self.batch_size,1)).mean(dim=0)

        states_K = states_.unsqueeze(1).repeat(1, self.K, 1, 1, 1)
        V_s_, A_s_ = self.q_next.forward(states_K)
        A_s_ = A_s_.reshape((self.K,self.batch_size,self.num_actions)).mean(dim=0)
        V_s_ = V_s_.reshape((self.K,self.batch_size,1)).mean(dim=0)
        """

        #ideal = [32,2,4,84,84]
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + (self.gamma ** self.n)*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.epsilon.update_eps()


        
