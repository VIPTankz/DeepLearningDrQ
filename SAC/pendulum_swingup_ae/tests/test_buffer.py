import sys
import os
import torch
import pytest

get_dir = os.getcwd()
sys.path.append(os.getcwd().replace('/tests', ''))

import networks
import buffer
import environment

def test1():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cpu')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))
    assert buf.size() == 100


def test2():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cpu')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))

    o, a, r, o_, d = buf.sample(5)
    assert o.shape == (5, 3, 84, 84)
    assert o_.shape == (5, 3, 84, 84)
    assert a.shape == (5, 1)
    assert r.shape == (5, 1)
    assert d.shape == (5, 1)

def test3():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cpu')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))

    o, a, r, o_, d = buf.sample(5)
    assert torch.is_floating_point(o)
    assert torch.is_floating_point(a)
    assert torch.is_floating_point(r)
    assert torch.is_floating_point(o_)
    assert torch.is_floating_point(d)

def test3():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cpu')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))

    o, a, r, o_, d = buf.sample(5)
    assert torch.is_floating_point(o)
    assert torch.is_floating_point(a)
    assert torch.is_floating_point(r)
    assert torch.is_floating_point(o_)
    assert torch.is_floating_point(d)


def test4():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cuda')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))

    o, a, r, o_, d = buf.sample(5)
    assert o.is_cuda
    assert a.is_cuda
    assert r.is_cuda
    assert o_.is_cuda
    assert d.is_cuda

def test5():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cpu')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))

    o, a, r, o_, d = buf.sample(5)
    assert not o.is_cuda
    assert not a.is_cuda
    assert not r.is_cuda
    assert not o_.is_cuda
    assert not d.is_cuda

def test5():
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cpu')
    obs = torch.rand(3, 84, 84)
    obs_ = torch.rand(3, 84, 84)
    action = 0.9812
    reward = 1.0
    done = True
    for i in range(120):
        buf.put((obs, action, reward, obs_, done))

    o, a, r, o_, d = buf.sample(5)
    assert not o.is_cuda
    assert not a.is_cuda
    assert not r.is_cuda
    assert not o_.is_cuda
    assert not d.is_cuda


def test6():
    env = environment.Environment()
    o, _ = env.reset()
    buf = buffer.ReplayBuffer(100, (3, 84, 84), (1,), 'cuda')
    step = 0
    while step < 50:
        d = False
        o, _ = env.reset()
        while not d:
            step += 1
            a = torch.rand(1).item()
            o_, r, d, _ = env.step(a)
            buf.put((o, a, r, o_, d))
            o = o_
            if step < 50:
                break

    mini_batch = buf.sample(20)
    o, a, r, o_, d = mini_batch
    assert o.max() <= 1
    assert o.min() >= 0
    assert o_.max() <= 1
    assert o.min() >= 0








