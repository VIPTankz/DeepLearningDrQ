# DeepLearningDrQ
Looking to replicate the results of the paper "IMAGE AUGMENTATION IS ALL YOU NEED: REGULARIZING DEEP REINFORCEMENT LEARNING FROM PIXELS"

This project however only focuses on the DQN-variant of the paper, benchmarked on Atari-100k.

Current Results (MINE):

Alien: 692.4
Amidar: 89.005
Assault: 432.48
Asterix: 326.5
BankHeist: 62.6
BattleZone: 6295.0
Boxing: -5.19
Breakout: 10.815
ChopperCommand: 1221.0
CrazyClimber: 17844.5
DemonAttack: 231.225
Freeway: 24.22
Frostbite: 1056.55
Gopher: 320.8
Hero: 11578.225
Jamesbond: 128.5
Kangaroo: 846.0
Krull: 2548.75
KungFuMaster: 1507.5
MsPacman: 1058.3
Pong: -17.09
PrivateEye: 403.995
Qbert: 1042.875
RoadRunner: 2488.5
Seaquest: 239.7
UpNDown: 2170.2

Human-Normalised Median:
0.108

This is far exceeded however by the the author's results, claiming a human-normliased median of 0.270

The base of my code was originally based on the code from "Machine Learning with Phil"'s Dueling Double DQN, but has been very heavily modified and udpated since, including the additions of the DrQ paper.

If you wish to to run this code, the environment.yml file is provided to replicate environment (this was done on windows with conda).

To run the agent, replicate the environment using the environment.yml file and run the follwing command:
main.py 0 0 0

This however can take a very long time to run, especially on a regular desktop pc.
