
import pickle

import matplotlib.pyplot as plt
import numpy as np

DIV_LINE_WIDTH = 50




scores = np.load('training_logs/train_data/scores.npy')
steps = np.load('training_logs/train_data/step_list.npy')
env_name = pickle.load(open('training_logs/train_data/env_name.txt', 'rb'))


def smooth_data(scores, window=3):
    all_means = []
    all_maxs = []
    all_mins = []
    for i, score in enumerate(scores):
        mean = False
        max = False
        min = False
        if i >= window and i <= len(scores)-window:
            max = np.max(scores[i-window:i+window])
            min = np.min(scores[i - window:i + window])
            mean = np.mean(scores[i-window:i+window])
        if i < window:
            max = np.max(scores[i:i+window])
            min = np.min(scores[i:i+window])
            mean = np.mean(scores[i:i+window])
        if i >= len(scores)-window:
            max = np.max(scores[i-window:i])
            min = np.min(scores[i-window:i])
            mean = np.mean(scores[i-window:i])

        all_means.append(mean)
        all_maxs.append(max)
        all_mins.append(min)
    return all_means, all_maxs, all_mins

all_means, all_maxs, all_mins = smooth_data(scores)

plt.plot(steps, all_means)
plt.fill_between(steps, y1=all_mins, y2=all_maxs, alpha=0.2, color="blue")
plt.xlabel("Environment Steps", fontsize=12)
plt.ylabel("Rewards", fontsize=12)
plt.title("SAC agent on " + env_name + " Environment", fontsize=14)
plt.savefig('plots/reward_plot_' + env_name + '.png')
plt.show()





