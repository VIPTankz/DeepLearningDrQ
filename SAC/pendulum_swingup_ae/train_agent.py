from collections import UserDict
import torch
import os


from dm_control import suite
from dm_control.suite.wrappers import pixels
import pickle
import numpy as np
from agent import SAC_Agent
from environment import Environment



STEPS_LIMIT = 1000000
LOGGING_DIR = 'training_logs/'
RETRAIN = True
ENV_NAME = "cartpole"
TASK_NAME = "swingup"


if __name__ == '__main__':
    pickle.dump(ENV_NAME, open(LOGGING_DIR + 'train_data/env_name.txt', 'wb'))
    env = Environment()
    timestep = env.reset()
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
    score, done = 0.0, False
    episode_count += 1
    while not timestep.last():
        action, log_prob = agent.choose_action(timestep.observation['pixels'].to(agent.DEVICE))
        print(timestep.observation['pixels'].shape)
        print(action)
        action = action.detach().cpu().numpy()

        next_timestep = env.step(action)
        agent.memory.put((timestep.observation['pixels'],
                          action,
                          next_timestep.reward,
                          next_timestep.observation['pixels'],
                          timestep.last()))

        score += next_timestep.reward

        timestep = next_timestep

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



