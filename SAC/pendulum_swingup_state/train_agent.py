from collections import UserDict
import torch
import os


from dm_control import suite
import pickle
import numpy as np
from agent import SAC_Agent
import queue


STEPS_LIMIT = 1000000
LOGGING_DIR = 'training_logs/'
RETRAIN = False
ENV_NAME = "cartpole"
TASK_NAME = "swingup"


if __name__ == '__main__':
    pickle.dump(ENV_NAME, open(LOGGING_DIR + 'train_data/env_name.txt', 'wb'))
    env = suite.load(domain_name=ENV_NAME, task_name=TASK_NAME)
    agent = SAC_Agent(chkpt_dir=LOGGING_DIR)
    step_count = 0
    episode_count = 0
    score_list = []
    step_list = []

    if not RETRAIN:
        agent.load_models()
        episode_count = pickle.load(open(LOGGING_DIR + 'train_data/episode_count.txt', 'rb'))
        step_count = pickle.load(open(LOGGING_DIR + 'train_data/step_count.txt', 'rb'))
        score_list = list(np.load(LOGGING_DIR + 'train_data/scores.npy'))
        step_list = list(np.load(LOGGING_DIR + 'train_data/step_list.npy'))


while step_count < STEPS_LIMIT:
    timestep = env.reset()
    state = np.concatenate((timestep.observation['position'], timestep.observation['velocity']))
    score, done = 0.0, False
    episode_count += 1
    while not timestep.last():
        action, log_prob = agent.choose_action(torch.FloatTensor(state))
        action = action.detach().cpu().numpy()

        timestep = env.step(action)
        state_prime = np.concatenate((timestep.observation['position'], timestep.observation['velocity']))

        agent.memory.put((state, action, timestep.reward, state_prime, timestep.last()))

        score += timestep.reward

        state = state_prime

        if agent.memory.size() > agent.batch_size:
            agent.learn()

        step_count += 1

    print("Episode:{}, Step_Count:{} Avg_Score:{:.1f}".format(episode_count, step_count, score))
    if score == False:
        print("should not be false")
    score_list.append(score)
    step_list.append(step_count)

    if episode_count % 10 == 0:
        agent.save_models()
        pickle.dump(episode_count, open(LOGGING_DIR + 'train_data/episode_count.txt', 'wb'))
        pickle.dump(step_count, open(LOGGING_DIR + 'train_data/step_count.txt', 'wb'))
        np.save(LOGGING_DIR + 'train_data/scores.npy', np.array(score_list))
        np.save(LOGGING_DIR + 'train_data/step_list.npy', np.array(step_list))



