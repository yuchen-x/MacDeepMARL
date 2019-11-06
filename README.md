# Macro-Action-Based Deep Multi-Agent Reinforcement Learning

This is the code for implementing the macro-action-based decentralized learning and centralized learning frameworks presented in the paper [Macro-Action-Based Deep Multi-Agent Reinforcement Learning](https://drive.google.com/file/d/1R5bh7Hqs_Dhzz7FMmPP8TmMmk_IppcWL/view).

## Installation

- To install the anaconda virtual env with all the dependencies:
  ```
  cd Anaconda_Env/
  conda env create -f 367_corl2019_env.yml
  ```
- To install the python module:
  ```
  cd CoRL2019/
  pip install -e .
  ```
## Decentralized Learning for Decentralized Execution

The first framework presented in this paper extends [Dec-HDRQN](https://arxiv.org/pdf/1703.06182.pdf) with Double Q-learning to learn decentralized macro-action-value function by proposing a **Macro-Action Concurrent Experience Replay Trajectories (Mac-CERTs)** to maintain macro-action-observation transitions for training.

Example:

![](https://github.com/yuchen-x/CoRL2019/blob/master/images/dec_buffer.png)

A mini-batch of squeezed experience is then used for optimizing each agent's decentralized Q-net.

## Centralized Learning for Centralized Execution

## Paper Citation
If you used this code for your reasearch or found it helpful, consider citing the following paper:
```
@InProceedings{xiao_corl_2019,
    author = "Xiao, Yuchen and Hoffman, Joshua and Amato, Christopher",
    title = "Macro-Action-Based Deep Multi-Agent Reinforcement Learning",
    booktitle = "3rd Annual Conference on Robot Learning",
    year = "2019"
}
```
