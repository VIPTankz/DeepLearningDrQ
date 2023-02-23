import sys
import os
import torch
import pytest

get_dir = os.getcwd()
sys.path.append(os.getcwd().replace('/tests', ''))

import networks

# ============================================= Test Encoder ===================================================

def test1():
    enc = networks.Encoder((3, 84, 84), (50,))
    input = torch.rand(5, 3, 84, 84)
    out = enc(input)
    assert out.shape == (5, 50)


def test2():
    enc1 = networks.Encoder((3, 84, 84), (50,))
    enc2 = networks.Encoder((3, 84, 84), (50,))
    for i in range(4):
        assert (enc1.convs[i].weight != enc2.convs[i].weight).any()
        assert (enc1.convs[i].bias != enc2.convs[i].bias).any()


def test3():
    enc1 = networks.Encoder((3, 84, 84), (50,))
    enc2 = networks.Encoder((3, 84, 84), (50,))
    enc1.copy_weights_from(enc2)
    for i in range(4):
        assert (enc1.convs[i].weight == enc2.convs[i].weight).all()
        assert (enc1.convs[i].bias == enc2.convs[i].bias).all()


# ================================================== Test Policy ====================================================

def test4():
    pi = networks.PolicyNetwork((3, 84, 84), (50,), (1,), 0.001,
                                -20, 2, 2.0, 2.0)
    obs = torch.rand(5, 3, 84, 84)
    mu, log_std = pi(obs)
    assert mu.shape == (5,1)
    assert log_std.shape == (5,1)


def test5():
    pi = networks.PolicyNetwork((3, 84, 84), (50,), (1,), 0.001,
                                -20, 2, 2.0, 2.0)
    obs = torch.rand(5, 3, 84, 84)
    action, log_prob = pi.sample(obs)
    assert action.shape == (5, 1)
    assert log_prob.shape == (5, 1)

def test6():
    pi = networks.PolicyNetwork((3, 84, 84), (50,), (1,), 0.001,
                                -20, 2, 2.0, 2.0)
    obs = torch.rand(512, 3, 84, 84)
    action, log_prob = pi.sample(obs)
    assert action.max() <= 2
    assert action.min() >= -2


# =============================================== Test Critic ===============================================
def test7():
    critic = networks.QNetwork((3,84,84), (50,), (1,), 0.001)
    obs = torch.rand((5, 3, 84, 84))
    action = torch.rand(5, 1)
    q = critic(obs, action)
    assert q.shape == (5, 1)


