# Flight Control Reinforcement Learning repository for course AE4350

This repository contains a reinforcement learning solution for a game inspired by the mobile Flight Control game, developed for the AE4350 Bio-Inspired Intelligence and Learning for Aerospace Applications course assignment. A picture of this game is shown below:
![alt text](https://github.com/DidierVDVoort/AE4350_Flight_Control_RL/blob/main/figures/flight_control.jpg?raw=true)

## Overview

The project implements two RL approaches to solve the Flight Control game:
- **Deep Q-Network (DQN)**: A deep learning approach using neural networks
- **Q-Learning**: A classic Q-table method with state discretization. It is multi-agent, meaning that one agent is assigned to one plane in the environment.

However, note that the main focus was put on the DQN, hence that is more developed and also used to obtain most results.
Both methods train agents to navigate multiple aircraft to landing zones while avoiding collisions.

## Key Files

- `sim.py`: Flight control simulation environment
- `dqn_agent.py` & `dqn_train.py`: DQN implementation and training
- `q_learning_agent.py` & `q_learning_train.py`: Q-learning implementation and training
- `dqn_visualisation.py` & `q_learning_visualisation.py`: Visualization tools to show the learned policies
- `dqn_sensitivity.py`: Hyperparameter and environment sensitivity analysis
- `method_comparison.py`: Performance comparison between both methods

## Usage

1. Train a DQN agent: `python dqn_train.py`
2. Train a Q-learning agent: `python q_learning_train.py`
3. Visualize trained policies using the respective visualization files: `python dqn_visualisation.py` or `python q_learning_visualisation.py`
4. Run sensitivity analysis for the DQN approach: `python dqn_sensitivity.py`
5. Compare DQN with classic Q-learning: `python method_comparison.py`

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- tqdm

## Results

The results that were discussed in the report have also been saved already as .pth or .csv files. They are stored in the following folders:
- `dqn_results`: Results of the DQN approach for the comparison of methods (i.e. used in `method_comparison.py`) --> section 3.1 in the report.
- `dqn_solution_analysis`: Results of the DQN approach for the analysis of found solutions (custom environment) --> section 3.2 in the report.
- `figures`: Includes all figures shown in the report.
- `qlearning_results`: Results of the Q-Learning approach for the comparison of methods (i.e. used in `method_comparison.py`) --> section 3.1 in the report.
- `sensitivity_results`: Sensitivity analysis results stored as .csv files. Can be used to plot sensitivity analysis resutls --> section 3.3 in the report.

### Happy coding!
