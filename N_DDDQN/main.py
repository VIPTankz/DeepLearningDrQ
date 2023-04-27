import gym
import ale_py
import numpy as np
from gym.wrappers import AtariPreprocessing
import time
from copy import deepcopy

#Main.py && Main.py -double True && Main.py -per True && Main.py -double True -per True

if __name__ == '__main__':

    #from NDDDQN_Agent import Agent
    from DrQ_NDDDQN_Agent import Agent
    for runs in range(5):
        env = gym.make('ALE/Breakout-v5')
        env = AtariPreprocessing(env, frame_skip=1)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(runs)
        print(env.observation_space)
        print(env.action_space)

        agent = Agent(discount=0.99, batch_size=32, n_actions=env.action_space.n,
                       input_dims=[4,84,84], lr=0.0001,
                      max_mem_size=100000)

        scores = []
        scores_temp = []
        n_steps = 100000
        steps = 0
        episodes = 0
        start = time.time()
        while steps < n_steps:

            score = 0
            episodes += 1
            done = False
            trun = False
            observation, info = env.reset()
            while not done and not trun:
                steps += 1
                action = agent.choose_action(observation)
                observation_, reward, done, trun, info = env.step(action)
                reward = np.clip(reward, -1., 1.)

                score += reward
                agent.memory.store_transition(observation, action, reward,
                                              observation_, done)
                agent.learn()
                observation = deepcopy(observation_)
            scores.append([score,steps])
            scores_temp.append(score)

            avg_score = np.mean(scores_temp[-50:])

            if steps % 10 == 0:
                print('DDQN {} episode {} score {:.0f} avg score {:.0f} eps {:.2f} total_steps {:.0f}'
                      .format(True, episodes, score, avg_score, agent.epsilon.eps, steps))
                print("FPS: " + str(time.time() - start))
                start = time.time()

        fname = 'Experiment' + str(runs) + '.npy'
        np.save(fname, np.array(scores))
        agent.eval_mode = True
        evals = []
        steps = 0
        while steps < 125000:
            done = False
            trun = False
            observation, info = env.reset()
            score = 0
            while not done and not trun:
                steps += 1
                action = agent.choose_action(observation)
                observation_, reward, done, trun, info = env.step(action)
                reward = np.clip(reward, -1., 1.)
                score += reward
                observation = observation_

            evals.append(score)
            print("Evaluation Score: " + str(score))

        fname = 'Evaluation' + str(runs) + '.npy'
        np.save(fname, np.array(evals))