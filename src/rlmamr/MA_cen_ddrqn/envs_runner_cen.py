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
                    accu_id_rewards[idx] = accu_id_rewards[idx] + reward if not last_id_valid[idx] else reward
                last_id_valid = valid

                accu_joint_rewards = accu_joint_rewards + reward if not last_joint_valid else reward
                last_joint_valid = max(valid)

                # sent experience back
                child.send((last_obs, 
                            actions, 
                            accu_id_rewards, 
                            accu_joint_rewards,
                            obs, 
                            terminate, 
                            valid,
                            max(valid)))

                last_obs = obs
                R += reward

            elif cmd == 'reset':
                last_obs =  env.reset()
                last_h = None  # single network for cen control
                last_id_valid = [1.0] * env.n_agent
                last_joint_valid = 1.0
                accu_id_rewards = [0.0] * env.n_agent
                accu_joint_rewards = 0.0
                R = 0.0

                child.send((last_obs, last_h, last_id_valid))
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

    def __init__(self, 
                 env, 
                 memory, 
                 n_env, 
                 h_explore, 
                 get_actions):

        """
        Parameters
        ----------
        env : gym.env
            A macro-action-based gym envrionment.
        memory : ReplayBuffer
            An instance of RepalyBuffer class.
        n_env : int
            The number of envs running in parallel.
        h_explore : bool
            Whether use history-based policy for rollout.
        get_actions : python function
            A function for getting macro-actions 
        """
        
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
            # env_return is (last_obs, a, acc_id_r, acc_j_r, obs, t, v)
            env_return = parent.recv()
            env_return = self.exp_to_tensor(env_return)
            self.episodes[idx].append(env_return)

            self.last_obses[idx] = env_return[4]
            self.last_valids[idx] = env_return[6]

            # if episode is done, add it to memory buffer
            if env_return[-3]:
                n_episode_done += 1
                self.memory.scenario_cache += self.episodes[idx]
                self.memory.flush_scenario_cache()

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.h_states[idx], self.last_valids[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                self.episodes[idx] = []

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
        acc_id_r = [torch.tensor(r).float().view(1,-1) for r in exp[2]]
        acc_joint_r = torch.tensor(exp[3]).float().view(1,-1)
        obs = [torch.from_numpy(o).float() for o in exp[4]]
        t = torch.tensor(exp[5]).float().view(1,-1)
        id_v = [torch.tensor(v, dtype=torch.uint8).view(1,-1) for v in exp[6]]
        joint_v = torch.tensor(exp[7], dtype=torch.uint8).view(1,-1)
        return (last_obs, a, acc_id_r, acc_joint_r, obs, t, id_v, joint_v)

