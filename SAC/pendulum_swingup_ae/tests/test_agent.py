import sys
import os
import torch
import pytest
import yaml
import itertools

get_dir = os.getcwd()
sys.path.append(os.getcwd().replace('/tests', ''))

from agent import SAC_Agent


def test1():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

def test2():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(100):
        obs = torch.rand(3, 84, 84)
        obs_ = torch.rand(3, 84, 84)
        r = torch.rand(1).item()
        a = torch.rand(1).item()
        d = 1 if torch.rand(1).item() > 0.05 else 0
        agent.memory.put((obs, a, r, obs_, d))

def test3():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(100):
        obs = torch.rand(3, 84, 84)
        obs_ = torch.rand(3, 84, 84)
        r = torch.rand(1).item()
        a = torch.rand(1).item()
        d = 1 if torch.rand(1).item() > 0.05 else 0
        agent.memory.put((obs, a, r, obs_, d))
    mini_batch = agent.memory.sample(20)
    targ = agent.calc_target(mini_batch)

def test4():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(10):
        obs = torch.rand(100, 3, 84, 84)
        action, _ = agent.choose_action(obs)
        assert action.max() <= cfg["agent"]["params"]["max_action"]

def test5():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(10):
        obs = torch.rand(100, 3, 84, 84)
        action, _ = agent.choose_action(obs)
        assert action.min() >= cfg["agent"]["params"]["min_action"]

def test6():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(300):
        obs = torch.rand(3, 84, 84)
        obs_ = torch.rand(3, 84, 84)
        r = torch.rand(1).item()
        a = torch.rand(1).item()
        d = 1 if torch.rand(1).item() > 0.05 else 0
        agent.memory.put((obs, a, r, obs_, d))

    for i in range(cfg["agent"]["params"]["critic_target_update_frequency"] + 1):
        agent.batch_size = 20
        agent.learn()


def test7():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(300):
        obs = torch.rand(3, 84, 84)
        obs_ = torch.rand(3, 84, 84)
        r = torch.rand(1).item()
        a = torch.rand(1).item()
        d = 1 if torch.rand(1).item() > 0.05 else 0
        agent.memory.put((obs, a, r, obs_, d))

    for i in range(cfg["agent"]["params"]["critic_target_update_frequency"] + 1):
        agent.batch_size = 20
        old_encoders = [
            agent.Q1.encoder,
            agent.Q2.encoder
        ]
        agent.learn()
        new_encoders = [
            agent.Q1.encoder,
            agent.Q2.encoder
        ]
        for i in range(4):
            assert (old_encoders[0].convs[i].weight == new_encoders[0].convs[i].weight).all()
            assert (old_encoders[1].convs[i].weight == new_encoders[1].convs[i].weight).all()
            assert (old_encoders[0].convs[i].bias == new_encoders[0].convs[i].bias).all()
            assert (old_encoders[1].convs[i].bias == new_encoders[1].convs[i].bias).all()



def test8():
    with open('../config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    os.chdir('../')
    agent = SAC_Agent(cfg)
    agent.save_models()


def test9():
    with open('config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)
    agent.load_models()




def test10():
    with open('config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(300):
        obs = torch.rand(3, 84, 84)
        obs_ = torch.rand(3, 84, 84)
        r = torch.rand(1).item()
        a = torch.rand(1).item()
        d = 1 if torch.rand(1).item() > 0.05 else 0
        agent.memory.put((obs, a, r, obs_, d))
    old_params = []
    for i in range(len(list(agent.Q1.parameters()))):
        old_params.append(list(agent.Q1.parameters())[i].clone().detach().cpu().numpy())

    for i in range(cfg["agent"]["params"]["critic_target_update_frequency"] + 5):
        agent.batch_size = 50
        agent.learn()

    new_params = []
    for i in range(len(list(agent.Q1.parameters()))):
        new_params.append(list(agent.Q1.parameters())[i].clone().detach().cpu().numpy())

    for i in range(len(new_params)):
        assert (new_params[i] != old_params[i]).any()


def test11():
    with open('config.yml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    agent = SAC_Agent(cfg)

    for i in range(300):
        obs = torch.rand(3, 84, 84)
        obs_ = torch.rand(3, 84, 84)
        r = torch.rand(1).item()
        a = torch.rand(1).item()
        d = 1 if torch.rand(1).item() > 0.05 else 0
        agent.memory.put((obs, a, r, obs_, d))

    for i in range(cfg["agent"]["params"]["critic_target_update_frequency"] + 5):
        agent.batch_size = 50
        agent.learn()

    for i in range(len(list(agent.Q1.parameters()))):
        print(list(agent.Q1.parameters())[i].clone().detach().cpu().numpy() == list(agent.Q2.parameters())[i].clone().detach().cpu().numpy())



