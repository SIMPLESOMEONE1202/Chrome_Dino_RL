🦖 Chrome Dino AI – Vision-Based Reinforcement Learning Agent

A vision-based Reinforcement Learning agent trained to autonomously play the Chrome Dino game using raw pixel input, real-time screen capture, and Deep Q-Networks (DQN).
The project focuses on environment design, reward shaping, and stability analysis under CPU-only constraints.

📌 Project Overview

This project implements a custom Gymnasium-compatible environment for the Chrome Dino game and trains a Deep Q-Network (DQN) agent using Stable-Baselines3.
Unlike simulator-based RL tasks, the agent interacts with the actual game running in a browser, making this a realistic and challenging control problem.

Key Characteristics

Real-time screen capture (no game API)

Vision-only observation space (grayscale frames)

Discrete action space (Jump, Duck, No-op)

CPU-only training

Robust GAME OVER detection

🎯 Objectives

Learn obstacle avoidance directly from pixel observations

Design a stable RL environment for a real-time game

Analyze DQN behavior and limitations under partial observability

Achieve consistent autonomous gameplay without simulator access

🧠 Methodology
Environment Design

Custom Gymnasium environment wrapping the Chrome Dino game

Screen capture using mss

Input preprocessing:

Grayscale conversion

Resizing

Frame stacking (4 frames)

Action execution via pydirectinput
Observation Space
Box(0, 255, shape=(4, H, W), dtype=uint8)

Action Space
Action	Description
0	Jump
1	Duck
2	No-op
Reward Structure

+1 per timestep survived

-15 on game termination

🤖 Reinforcement Learning Algorithm

Algorithm: Deep Q-Network (DQN)

Framework: Stable-Baselines3

Enhancements:

Double DQN (enabled by default in SB3)

Frame stacking for temporal awareness

Careful exploration scheduling

Key Hyperparameters
Learning Rate        : 7e-5
Replay Buffer Size   : 60,000
Batch Size           : 32
Gamma                : 0.99
Exploration Fraction : 0.7
Final Epsilon        : 0.08
Target Update        : 1000 steps

📊 Results
Evaluation (10 Episodes, Deterministic Policy)
MAX STEPS  : ~140
MEAN STEPS : ~60


The agent learned effective obstacle avoidance

Two behaviors emerged:

A stable policy surviving ~140 steps

A degenerate short-horizon policy (~6 steps)

This oscillation highlights known limitations of vanilla DQN in:

partially observable environments

real-time control with input latency

⚠️ Limitations & Insights

Vanilla DQN struggles with policy stability in real-time vision tasks

Q-value overestimation leads to occasional policy collapse

Performance plateaus without algorithmic upgrades (e.g., PPO, LSTM)

These behaviors are expected and well-documented in RL literature.

🔮 Future Work

Upgrade to PPO or A2C for better policy stability

Introduce recurrent (LSTM) policies

Replace screen capture with a simulator clone

Train on GPU for faster convergence

Add curriculum learning (speed-based difficulty)

🛠 Tech Stack

Python

Stable-Baselines3

Gymnasium

OpenCV

MSS

PyDirectInput

NumPy

TensorBoard