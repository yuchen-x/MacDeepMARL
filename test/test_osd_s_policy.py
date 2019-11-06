import argparse
import numpy as np
import torch
import os
import sys
import IPython
import logging

sys.path.append("..")
import time
import IPython

from rlmamr.my_env.osd_ma_single_room import ObjSearchDelivery_v4 as OSD_S_4
from rlmamr.MA_cen_condi_ddrqn.utils.utils import Linear_Decay, get_conditional_argmax, get_conditional_action
from rlmamr.MA_cen_condi_ddrqn.utils.Cen_ctrl import Cen_Controller

from IPython.core.debugger import set_trace

ENVIRONMENTS = {
        'OSD_S_4':OSD_S_4}

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def get_actions_and_h_states(env, agent, joint_obs, h_state, last_action, last_valid):
    with torch.no_grad():
        if max(last_valid) == 1.0:
            Q, h = agent.policy_net(torch.cat(joint_obs).view(1,1,np.sum(env.obs_size)), h_state)
            a = get_conditional_argmax(Q, get_conditional_action(torch.cat(last_action).view(1,-1), torch.cat(last_valid).view(1,-1)), env.n_action).item()
            actions = np.unravel_index(a, env.n_action)
            new_h_state = h
        else:                
            actions = [-1] * 3
            new_h_state = h_state

    return actions, new_h_state

def get_init_inputs(env,n_agent):
    return [torch.from_numpy(i).float() for i in env.reset(True)], None

def test(env_name, env_terminate_step,n_agent, n_episode, p_id):
    ENV = ENVIRONMENTS[env_name]
    env = ENV()

    agent = Cen_Controller()
    agent.idx = 0
    agent.policy_net = torch.load("./policy_nns/" + str(p_id) + "_cen_controller.pt")
    agent.policy_net.eval()

    R = 0

    for e in range(n_episode):
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)
        if e==0:
            set_trace()
        last_valid = [torch.tensor([[1]]).byte()] * n_agent
        last_action = [torch.tensor([[-1]])] * n_agent
        step = 0
        while not t:
            a, h_states = get_actions_and_h_states(env, agent, last_obs, h_states, last_action, last_valid)
            time.sleep(0.4)
            a, last_obs, r, t, v = env.step(a,True)
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_action = [torch.tensor(a_idx).view(1,1) for a_idx in a]
            last_valid = [torch.tensor(_v, dtype=torch.uint8).view(1,-1) for _v in v]
            R += r
            step += 1

        set_trace()
        time.sleep(0.2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', action='store', type=str, default='OSD_S_4')
    parser.add_argument('--env_terminate_step', action='store', type=int, default=150)
    parser.add_argument('--n_agent', action='store', type=int, default=3)
    parser.add_argument('--n_episode', action='store', type=int, default=1)
    parser.add_argument('--p_id', action='store', type=int, default=2)

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()


