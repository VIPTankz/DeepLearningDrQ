import gym
import numpy as np

#Main.py && Main.py -double True && Main.py -per True && Main.py -double True -per True

if __name__ == '__main__':

    from NDDDQN_Agent import Agent
    for runs in range(15):
        env = gym.make('LunarLander-v2')

        agent = Agent(gamma=0.99, epsilon=0.1, batch_size=32, n_actions=4,
                      eps_end=0.1, input_dims=[8], lr=0.001,
                      max_mem_size=1000000)

        scores = []
        n_steps = 1000000
        steps = 0
        episodes = 0
        while steps < n_steps:
            score = 0
            episodes += 1
            done = False
            observation = env.reset()
            while not done:
                steps += 1
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                env.render()
                score += reward
                agent.memory.store_transition(observation, action, reward,
                                              observation_, done)
                agent.learn()
                observation = observation_
            scores.append(score)

            avg_score = np.mean(scores[-100:])

            if steps % 10 == 0:
                print('DDQN {} episode {} score {:.0f} avg score {:.0f} eps {:.2f}'
                      .format(True, episodes, score, avg_score, agent.epsilon))

        fname = 'Experiment' + str(runs) + '.npy'

        np.save(fname, np.array(scores))