import numpy as np
import torch
import IPython

from multiprocessing import Process, Pipe
from IPython.core.debugger import set_trace

def worker(child, env):
    """
    Worker function which interacts with the environment over remote
    """
    try:
        while True:
            # wait cmd sent by parent
            cmd, data = child.recv()
            if cmd == 'step':
                actions, obs, reward, terminate, valid = env.step(data)

                for idx, v in enumerate(valid):
                    accu_rewards_per_step[idx] = accu_rewards_per_step[idx] + reward if not last_valid[idx] else reward
                last_valid = valid

                # sent experience back
                child.send((last_obs, actions, accu_rewards_per_step, obs, terminate, valid))

                last_obs = obs
                R += reward

            elif cmd == 'reset':
                last_obs =  env.reset()
                last_h = [None] * env.n_agent
                last_valid = [1.0] * env.n_agent
                accu_rewards_per_step = [0.0] * env.n_agent
                R = 0.0

                child.send((last_obs, last_h, last_valid))
            elif cmd == 'close':
                child.close()
                break
            else:
                raise NotImplementerError
 
    except KeyboardInterrupt:
        print('EnvRunner worker: caught keyboard interrupt')
    except Exception as e:
        print('EnvRunner worker: uncaught worker exception')
        raise

class EnvsRunner(object):
    """
    Environment runner which runs multiple environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, env, memory, n_env, h_explore, get_actions):
        
        # func for getting next action via current policy nn
        self.get_actions = get_actions
        # create connections via Pipe
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_env)])]
        # create multip processor with multiple envs
        self.envs = [Process(target=worker, args=(child, env)) for child in self.children]
        # replay buffer
        self.memory = memory

        self.hidden_states = [None] * env.n_agent
        self.h_explore = h_explore
        self.episodes = [[]] * n_env

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def step(self):

        n_episode_done = 0

        for idx, parent in enumerate(self.parents):
            # get next action
            if self.h_explore:
                actions, self.h_states[idx] = self.get_actions(self.last_obses[idx], self.h_states[idx], self.last_valids[idx])
            else:
                actions, self.hidden_states[idx] = self.get_actions(self.last_obses[idx], self.h_states[idx], self.last_valids[idx])

            # send cmd to trigger env step
            parent.send(("step", actions))

        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            # env_return is (last_obs, a, acc_r, obs, t, v)
            env_return = parent.recv()
            env_return = self.exp_to_tensor(env_return)
            self.episodes[idx].append(env_return)

            self.last_obses[idx] = env_return[3]
            self.last_valids[idx] = env_return[5]

            # if episode is done, add it to memory buffer
            if env_return[-2]:
                n_episode_done += 1
                self.memory.scenario_cache += self.episodes[idx]
                self.memory.flush_scenario_cache()

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.h_states[idx], self.last_valids[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                self.episodes[idx] = []
                continue

        return n_episode_done

    def reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.last_valids = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses]

    def close(self):
        [parent.send(('close', None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def obs_to_tensor(self, obs):
        return [torch.from_numpy(o).float() for o in obs]

    def exp_to_tensor(self, exp):
        last_obs = [torch.from_numpy(o).float() for o in exp[0]]
        a = [torch.tensor(a).view(1,-1) for a in exp[1]]
        r = [torch.tensor(r).float().view(1,-1) for r in exp[2]]
        obs = [torch.from_numpy(o).float() for o in exp[3]]
        t = torch.tensor(exp[4]).float().view(1,-1)
        v = [torch.tensor(v, dtype=torch.uint8).view(1,-1) for v in exp[5]]
        return (last_obs, a, r, obs, t, v)

