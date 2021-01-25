import numpy as np
import IPython
import matplotlib.pyplot as plt
import torch
import pickle
import math
import copy
import random

from torch.optim import Adam, RMSprop
from itertools import chain

from .model import DDRQN
from .utils.Agent import Agent
from .utils.utils import Linear_Decay, save_check_point
from .envs_runner import EnvsRunner

from IPython.core.debugger import set_trace

# parameters for e-greedy policy
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 2000
EPS_DECAY_LINEAR_RATE = 0.9999

OPTIMIZERS = {'Adam': Adam,
              'RMSprop': RMSprop}

class Team:

    """Base class of a team of agents"""
    
    def __init__(self, 
                 env, 
                 memory, 
                 n_agent, 
                 h_stable_at, 
                 dynamic_h=False, 
                 hysteretic=None, 
                 discount=0.99,
                 epsilon_linear_decay=False, 
                 epsilon_linear_decay_steps=0):
        """
        Parameters
        ----------
        env : gym.env
            A domain environment.
        memory : ReplayBuffer
            A instance of the ReplayBuffer class.
        n_agent : int
            The number of agent.
        h_stable_at : int
            The number of dacaying episodes/stpes for hysteretic learning rate.
        dynamic_h : bool
            Whether apply hysteratic learning rate decay.
        hysteretic : tuple
            A tuple of initialzed and ending hysteritic learning rates.
        discount : float
            Discount factor for learning.
        epsilon_linear_decay : bool
            Whether apply epsilon decay for explorating policy
        epsilon_linear_decay_steps : int
            The number of episodes/steps for epsilon decay
        """

        self.env = env
        self.n_agent = n_agent
        self.memory = memory

        self.step_count = 0.0
        self.episode_count = 0.0
        self.episode_rewards = 0.0
 
        # hysteretic settings
        self.dynamic_h = dynamic_h
        (self.init_hysteretic, self.end_hysteretic) = hysteretic
        self.hysteretic = self.init_hysteretic
        self.discount = discount

        # epsilon for e-greedy
        self.epsilon = EPS_START
        self.epsilon_linear_decay = epsilon_linear_decay
        self.eps_l_d = Linear_Decay(epsilon_linear_decay_steps, EPS_START, EPS_END)
        
        self.HYSTERESIS_STABLE_AT = h_stable_at

    def create_agents(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get_next_actions(self):
        raise NotImplementedError

    def update_target_net(self):
        for agent in self.agents:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    def train(self):
        raise NotImplementedError

    def update_epsilon(self, step):
        # update epsilon:
        if self.epsilon_linear_decay:
            self.epsilon = self.eps_l_d._get_value(step)
        else:
            self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (step//8)  / EPS_DECAY)
    
    def update_hysteretic(self, step):
        if self.dynamic_h:
            self.hysteretic = min(self.end_hysteretic,
                                  ((self.end_hysteretic - self.init_hysteretic) / self.HYSTERESIS_STABLE_AT) * step + self.init_hysteretic)
        else:
            self.hysteretic = 1 - self.epsilon
    
    def get_init_inputs(self):
        raise NotImplementedError

    def sep_joint_exps(self, joint_exps):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def load_check_point(self, idx_run):
        for idx, agent in enumerate(self.agents):
            PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_agent_" + str(idx) + "1.tar"
            ckpt = torch.load(PATH)
            agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
            agent.target_net.load_state_dict(ckpt['target_net_state_dict'])
            agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.hysteretic = ckpt['cur_hysteretic']
        self.epsilon = ckpt['cur_eps']
        self.episode_count = ckpt['n_epi']
        self.step_count = ckpt['cur_step']
        self.TEST_PERFORM = ckpt['TEST_PERFORM']
        self.memory.buf = ckpt['mem_buf']
        random.setstate(ckpt['random_state'])
        np.random.set_state(ckpt['np_random_state'])
        torch.set_rng_state(ckpt['torch_random_state'])

class Team_RNN(Team):

    """A instance of Team class with RNN agent"""

    def __init__(self, 
                 env, 
                 n_env, 
                 memory, 
                 n_agent, 
                 training_method, 
                 h_stable_at, 
                 discount=0.99, 
                 centralized_training=False, 
                 sample_epi=False, 
                 dynamic_h=False, 
                 hysteretic=None, 
                 h_explore=False, 
                 epsilon_linear_decay=False, 
                 epsilon_linear_decay_steps=0, 
                 epsilon_exp_decay=False, 
                 optimizer='Adam', 
                 learning_rate=0.001, 
                 device='cpu', 
                 save_dir=None, 
                 nn_model_params={}, 
                 **hyper_params):

        """
        Parameters
        ----------
        env : gym.env
            A domain environment.
        n_env : int
            The number of envs running in parallel.
        memory : ReplayBuffer
            A instance of the ReplayBuffer class.
        n_agent : int
            The number of agent.
        training_method : python function
            A algorithm for calculating loss and performing optimization.
        h_stable_at : int
            The number of dacaying episodes/stpes for hysteretic learning rate.
        discount : float
            Discount factor for learning.
        centralized_training : bool
            Whether performs centralized training or not.
        sample_epi : bool
            Whether simples entire episode in mini-batch learning.
        dynamic_h : bool
            Whether apply hysteratic learning rate decay.
        hysteretic : tuple
            A tuple of initialzed and ending hysteritic learning rates.
        h_explore : bool
            Whether uses history-based exploring policy.
        epsilon_linear_decay : bool
            Whether apply epsilon decay for explorating policy
        epsilon_linear_decay_steps : int
            The number of episodes/steps for epsilon decay
        epsilon_exp_decay : bool
            Whether apply exponentially decay for epsilon.
        optimizer : str
            Name of an optimizer.
        learning_rate : float
            Learning rate.
        device : str
            CPU/GPU for training.
        save_dir : str
            Name of a directory to save results/ckpt.
        nn_model_params : dict[..]
            A dictionary of network parameters.
        hyper_params : dict[..] 
            A dictionary of some rest hyper-parameters.
        """

        super(Team_RNN, self).__init__(env, memory, n_agent, h_stable_at, 
                                       dynamic_h, hysteretic, discount,
                                       epsilon_linear_decay, epsilon_linear_decay_steps)

        # create multiprocessor for multiple envs running parallel
        self.envs_runner = EnvsRunner(self.env, self.memory, n_env, h_explore, self.get_next_actions)
        self.envs_runner.reset()
        self.n_env = n_env

        # sample the whole episode for training
        self.sample_epi = sample_epi
        
        # training method
        self.cen_train = centralized_training
        self.training_method = training_method
        self.nn_model_params = nn_model_params
        self.hyper_params = hyper_params
        self.optimizer = optimizer
        self.lr = learning_rate
        
        # save model
        self.save_dir = save_dir
        self.device = device

        # statistic of training and testing
        self.TRAIN_PERFORM = []
        self.TEST_PERFORM = []

        # create agents
        self.create_agents()

    def create_agents(self):
        self.agents=[]
        for i in range(self.n_agent):
            agent = Agent()
            agent.idx = i
            agent.policy_net = DDRQN(self.env.obs_size[i], self.env.n_action[i], **self.nn_model_params).to(self.device)
            agent.policy_net.is_policy_net = True
            agent.target_net = DDRQN(self.env.obs_size[i], self.env.n_action[i], **self.nn_model_params).to(self.device)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.is_policy_net = False
            agent.optimizer = OPTIMIZERS[self.optimizer](agent.policy_net.parameters(), lr=self.lr)
            self.agents.append(agent)

    def step(self, idx_run):
        if self.step_count == 0:
            self.evaluate()
            with open("./performance/" + self.save_dir + "/test/test_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                pickle.dump(self.TEST_PERFORM, handle)

        self.step_count += 1.0

        n_episode_done = self.envs_runner.step()
        self.episode_count += n_episode_done

        if n_episode_done > 0 and not self.episode_count % 10:
            self.evaluate()

            with open("./performance/" + self.save_dir + "/test/test_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                pickle.dump(self.TEST_PERFORM, handle)

            for agent in self.agents:
                 PATH = "./policy_nns/" + self.save_dir + "/agent_" + str(agent.idx) + ".pt"
                 torch.save(agent.policy_net, PATH)

    def get_next_actions(self, 
                         joint_obs, 
                         joint_h_states, 
                         last_valid, 
                         eval=False):

        """
        Parameters
        ----------
        joint_obs : ndarry | List[..]
            A list of each agent's observation.
        joint_h_states : ndarry | List[..]
            A list of hidden state of each agent's rnn-net
        last_valid : int | List[..]
            A list of integer indicates whether each agent has finished the previous macro-action.
        eval : bool
            Whether use evaluation mode or not.

        Returns
        -------
        actions : int | List[..]
            A list of the index of macro-action for each agent.
        h_states : ndarry | List[..]
            A list of hidden state of each agent's rnn-net.
        """

        with torch.no_grad():
            actions = []
            h_states = []
            for agent in self.agents:
                if last_valid[agent.idx]:
                    Q, h = agent.policy_net(joint_obs[agent.idx].view(1,1,self.env.obs_size[agent.idx]), joint_h_states[agent.idx])
                    a = Q.squeeze(1).max(1)[1].item()
                    actions.append(a)
                    h_states.append(h)
                else:
                    actions.append(-1)
                    h_states.append(joint_h_states[agent.idx])

        if not eval:
            actions = [a if np.random.random() > self.epsilon else self.env.action_space_sample(i) for i, a in enumerate(actions)]
        return actions, h_states

    def train(self):
        if self.sample_epi:
            batch, trace_len = self.memory.sample()
            batch = self.sep_joint_exps(batch)
            self.hyper_params['trace_len'] = trace_len
        else:
            batch = self.sep_joint_exps(self.memory.sample())
        self.training_method(self.agents, batch, self.hysteretic, self.discount, **self.hyper_params)

    def get_init_inputs(self):
        return [torch.from_numpy(i).float() for i in self.env.reset()], [None]*self.n_agent

    def sep_joint_exps(self, joint_exps):

        """
        Parameters
        ----------
        joint_exps : List[List[tuple(..)]]
            A sampled batch of episodes/sequences, whose size equals to the number of episodes..

        Return
        ------
        exps : List[List[tuple(..)]]
            A separeted batch of episdoes/sequences for each agent, whose size equals to the number of agents. 
        """
        # seperate the joint experience for individual agents
        exps = [[] for _ in range(self.n_agent)]
        for o, a, r, o_n, t, v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([o[i], a[i], r[i], o_n[i], t, v[i]])
        return exps

    def evaluate(self, n_episode=10):

        R, L = 0, 0

        for _ in range(n_episode):
            t = 0
            step = 0
            last_obs, h_states = self.get_init_inputs()
            last_valid = [1] * self.n_agent
            while not t:
                a, h_states = self.get_next_actions(last_obs, h_states, last_valid, eval=True)
                a, last_obs, r, t, v = self.env.step(a)
                last_obs = [torch.from_numpy(o).float() for o in last_obs]
                last_valid = v
                R += self.discount**step * r
                step += 1

        self.TEST_PERFORM.append(R/n_episode)
