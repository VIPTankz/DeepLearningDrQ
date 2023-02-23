import pickle
import numpy as np
from agent import SAC_Agent
from environment import Environment
import yaml

with open('config.yml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

if __name__ == '__main__':
    env = Environment()
    obs, _ = env.reset()
    agent = SAC_Agent(cfg)
    step_count = 0
    episode_count = 0
    score_list = []
    step_list = []

    # Seed the replay buffer with initial experiences
    if cfg["retrain"]:
        step = 0
        while step < cfg["num_seed_steps"]:
            obs, _ = env.reset()
            not_done = True
            while not_done:
                print("Seeding: ", step/cfg["num_seed_steps"])
                action, lp = agent.choose_action(obs[None, :].to(agent.device))
                action = action.detach().cpu().numpy()[0][0]
                obs_, reward, done, _ = env.step(action)
                agent.memory.put((obs, action, reward, obs_, done))
                obs = obs_
                step += 1
                if step > cfg["num_seed_steps"]:
                    break

    else:
        agent.load_models()
        episode_count = pickle.load(open(cfg["logging_dir"] + '/train_data/episode_count.txt', 'rb'))
        step_count = pickle.load(open(cfg["logging_dir"]  + '/train_data/step_count.txt', 'rb'))
        score_list = list(np.load(cfg["logging_dir"]  + '/train_data/scores.npy'))
        step_list = list(np.load(cfg["logging_dir"] + '/train_data/step_list.npy'))

    while step_count < cfg["num_train_steps"]:
        obs, _ = env.reset()
        episode_score, done = 0.0, False
        episode_count += 1
        while not done:
            action, log_prob = agent.choose_action(obs[None, :].to(agent.device))
            action = action.detach().cpu().numpy()[0][0]
            obs_, reward, done, _ = env.step(action)
            agent.memory.put((obs, action, reward, obs_, done))
            episode_score += reward
            obs = obs_
            if agent.memory.size() > agent.batch_size:
                agent.learn()

            step_count += 1

        print("Episode: {}, Step_Counter:{}, Avg_Score:{:.1f}".format(episode_count, step_count, episode_score))
        score_list.append(episode_score)
        step_list.append(step_count)

        if episode_count % cfg["save_training_state_freq"] == 0:
            agent.save_models()
            pickle.dump(episode_count, open(cfg["logging_dir"] + 'train_data/episode_count.txt', 'wb'))
            pickle.dump(step_count, open(cfg["logging_dir"] + 'train_data/step_count.txt', 'wb'))
            np.save(cfg["logging_dir"] + 'train_data/scores.npy', np.array(score_list))
            np.save(cfg["logging_dir"] + 'train_data/step_list.npy', np.array(step_list))








