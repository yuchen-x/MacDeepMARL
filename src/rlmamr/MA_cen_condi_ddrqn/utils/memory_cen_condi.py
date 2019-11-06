import numpy as np
import torch
import IPython

from IPython.core.debugger import set_trace
from collections import deque

class ReplayMemory:

    def __init__(self, n_agent, obs_size, batch_size, size):
        # obs_size is a list, coresponde to n_agents
        assert len(obs_size) == n_agent

        self.batch_size, self.obs_size = batch_size, obs_size

        self.ZERO_JOINT_OBS = [torch.zeros(s) for s in obs_size]
        self.ZERO_JOINT_ACT = [torch.tensor(0).view(1,-1)] * n_agent
        self.ZERO_ID_REWARD = [torch.tensor(0.0).view(1,-1)] * n_agent
        self.ZERO_JOINT_REWARD = torch.tensor(0.0).view(1,-1)
        self.ZERO_ID_VALID = [torch.tensor(0, dtype=torch.uint8).view(1,-1)] * n_agent
        self.ZERO_JOINT_VALID = torch.tensor(0, dtype=torch.uint8).view(1,-1)

        self.ONE_ID_VALID = [torch.tensor(1, dtype=torch.uint8).view(1,-1)] * n_agent
        self.ONE_JOINT_VALID = torch.tensor(1, dtype=torch.uint8).view(1,-1)

        self.ZERO_PADDING = [(self.ZERO_JOINT_OBS, 
                              self.ZERO_JOINT_ACT, 
                              self.ZERO_ID_REWARD,
                              self.ZERO_JOINT_REWARD, 
                              self.ZERO_JOINT_OBS, 
                              torch.tensor(0).float().view(1,-1), 
                              self.ZERO_ID_VALID, 
                              self.ZERO_JOINT_VALID)]

        self.ZEROS_ONE_PADDING = [(self.ZERO_JOINT_OBS,
                                   self.ZERO_JOINT_ACT, 
                                   self.ZERO_ID_REWARD,
                                   self.ZERO_JOINT_REWARD, 
                                   self.ZERO_JOINT_OBS, 
                                   torch.tensor(0).float().view(1,-1), 
                                   self.ONE_ID_VALID,
                                   self.ONE_JOINT_VALID)]

        self.buf = deque(maxlen=size)

    def append(self, transition):
        self.scenario_cache.append(transition)

    def scenario_cache_reset(self):
        raise NotImplementedError

    def flush_scenario_cache(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError

class ReplayMemory_rand(ReplayMemory):

    def __init__(self, n_agent, obs_size, trace_len, batch_size, size=100000):
        super(ReplayMemory_rand, self).__init__(n_agent, obs_size, batch_size, size)
        self.trace_len = trace_len
        self.scenario_cache_reset()

    def flush_scenario_cache(self):
        for i in range(len(self.scenario_cache)):
            trace = self.scenario_cache[i:i+self.trace_len]
            # end-of-episode padding
            trace = trace + self.ZERO_PADDING * (self.trace_len - len(trace)) 
            self.buf.append(trace)
        self.scenario_cache_reset()

    def scenario_cache_reset(self):
        self.scenario_cache = self.ZERO_PADDING * (self.trace_len - 1)

    def sample(self):
        indices = np.random.choice(len(self.buf), self.batch_size)
        return [self.buf[i] for i in indices]

class ReplayMemory_epi(ReplayMemory):
    
    def __init__(self, n_agent, obs_size, batch_size, size=100000):
        super(ReplayMemory_epi, self).__init__(n_agent, obs_size, batch_size, size)
        self.scenario_cache_reset()

    def flush_scenario_cache(self):
        self.buf.append(self.scenario_cache)
        self.scenario_cache_reset()

    def scenario_cache_reset(self):
        self.scenario_cache = []

    def sample(self):
        indices = np.random.choice(len(self.buf), self.batch_size)
        batch = [self.buf[i] for i in indices]
        return self.padding_batches(batch)
    
    def padding_batches(self, batch):
        max_len = max([len(epi) for epi in batch])
        batch = [epi + self.ZERO_PADDING * (max_len - len(epi)) for epi in batch]
        return batch, max_len
