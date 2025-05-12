# Reinforcement Learning for Robotic Task Recognition

This project demonstrates a reinforcement learning (RL) system for robotic task recognition in event‐driven environments. It integrates two types of agents:

- **Agent 1:** A deep Q-learning–based decision maker that selects a target (a cube) to open in the “Open the Chests” environment. It supports several environment versions:
  - **v0:** Simple one-to-one mapping using a Multi-Layer Perceptron (MLP).
  - **v1:** Uses a Long Short-Term Memory (LSTM) network to process short sequences.
  - **v2:** Employs an advanced LSTM + Transformer architecture for handling long-term dependencies.

- **Agent 2:** A Proximal Policy Optimization (PPO)–based controller for a simulated Kuka IIWA robotic arm within the `KukaMultiCubeReachEnv`. Its objective is to move the arm’s end-effector to a target cube (as chosen by Agent 1) while avoiding collisions and unnecessary movements.

The system further integrates both agents into a **Combined Agent System** that:
- Uses Agent 1 to choose the target cube.
- Commands Agent 2 to execute the reaching task—either from a full reset or continuously (partial reset) depending on configuration.

## Workspace Structure

- **agent_1.ipynb**  
  Contains the notebook for Agent 1. It details network architectures, training procedures, and analysis for decision making in various environment configurations.

- **agent_1.py**  
  Implements the deep Q-learning models and network architectures used by Agent 1.

- **agent_2.py**  
  Implements the PPO-based agent that controls the Kuka IIWA arm in the multi-cube environment.

- **Reinforcement_Learning_Project.ipynb**  
  Serves as the main notebook. It integrates environment registration, training loops, evaluation routines, and visualization tools. It explains how both agents are trained, evaluated, and combined.

- **register_envs.py**  
  Registers the custom Gym (Gymnasium) environments used by both agents.

- **requirements.txt**  
  Lists all dependencies required for this project.

- **logs/**  
  Stores training logs, model checkpoints, and vector normalization statistics.

- **imgs/** and **plots/**  
  Contain generated images like performance plots and video snapshots during training and evaluation.

- **weights/**  
  Contains saved model weights for both Agent 1 and Agent 2.

## Environment and Setup

- **Frameworks & Libraries:**  
  Python 3.11, PyTorch, OpenAI Gym (Gymnasium), PyBullet, Stable Baselines3, and several utility libraries (NumPy, Matplotlib, etc.).
  
- **Simulation:**  
  Custom environments simulate robotic tasks using PyBullet. Visual debugging is available via PyBullet’s GUI when required.

### Installation

Install all necessary dependencies using pip. For example:

```sh
pip install pybullet openthechests gym matplotlib imageio-ffmpeg torch torchvision torchaudio torchmetrics wandb opencv-python