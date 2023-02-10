from collections import UserDict
import torch
import os


import gymnasium as gym

import numpy as np
from agent import SAC_Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':



    env = gym.make('Pendulum-v1')
    agent = SAC_Agent()

    EPISODE = 500
    steps_count = 0
    print_once = True
    score_list = []

    for EP in range(EPISODE):
        state, _ = env.reset()
        score, done = 0.0, False

        while not done:
            action, log_prob = agent.choose_action(torch.FloatTensor(state))
            action = action.detach().cpu().numpy()

            state_prime, reward, done, trun, _ = env.step(action)
            done = done or trun

            agent.memory.put((state, action, reward, state_prime, done))

            score += reward

            state = state_prime

            if agent.memory.size() > agent.batch_size:
                agent.learn()

            steps_count += 1

        print("EP:{}, Avg_Score:{:.1f}".format(EP, score))
        score_list.append(score)

        if EP % 10 == 0:
            agent.save_models()

    np.savetxt(log_save_dir + '/pendulum_score.txt', score_list)