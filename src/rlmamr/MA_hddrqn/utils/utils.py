import torch
import os
import numpy as np
import random
import pickle

from IPython.core.debugger import set_trace

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def _get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def save_check_point(Agents, cur_step, n_epi, cur_hysteretic, cur_eps, save_dir, mem, run_id, test_perform, max_save=3):

    for idx, agent in enumerate(Agents):
        #PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_agent_" + str(agent.idx) + ".{}tar"
        PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_agent_" + str(idx) + "{}.tar"

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save({
                    'cur_step': cur_step,
                    'n_epi': n_epi,
                    'policy_net_state_dict':agent.policy_net.state_dict(),
                    'target_net_state_dict':agent.target_net.state_dict(),
                    'optimizer_state_dict':agent.optimizer.state_dict(),
                    'cur_hysteretic':cur_hysteretic,
                    'cur_eps':cur_eps,
                    'TEST_PERFORM': test_perform,
                    'mem_buf':mem.buf,
                    'random_state':random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_random_state': torch.random.get_rng_state()
                    }, PATH)

