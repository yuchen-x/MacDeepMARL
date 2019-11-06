#!/usr/bin/python

import numpy as np
import IPython
import gym

from gym import spaces
from numpy.random import randint
from .capture_target_MA_core import Agent_v1
from IPython.core.debugger import set_trace

NORTH = np.array([0, 1])
SOUTH = np.array([0, -1])
WEST = np.array([-1, 0])
EAST = np.array([1, 0])
STAY = np.array([0, 0])

TRANSLATION_TABLE = [
    # [left, intended_direction, right]
    [WEST,  NORTH, EAST],
    [EAST,  SOUTH, WEST],
    [SOUTH, WEST,  NORTH],
    [NORTH, EAST,  SOUTH],
    [STAY,  STAY,  STAY]
]

DIRECTION = np.array([[0.0, 1.0],
                      [0.0, -1.0],
                      [-1.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.0]])

ACTIONS = ["M_T_T", "STAY"]

class CaptureTarget_MA(gym.Env):

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, n_target, n_agent, grid_dim, terminate_step=60,
                 intermediate_r=False, target_flick_prob=0.3, obs_one_hot=False,
                 tgt_avoid_agent=False, tgt_trans_noise=0.0, agent_trans_noise=0.1):

        # env generic settings
        self.n_target = n_target 
        self.n_agent = n_agent
        self.multi_task = self.n_target != 1
        self.intermediate_reward = intermediate_r
        self.terminate_step=terminate_step

        # dimensions 
        self.grid_dim = grid_dim
        self.x_len, self.y_len = self.grid_dim
        self.x_mean, self.y_mean = np.mean(np.arange(self.x_len)), np.mean(np.arange(self.y_len))

        # probabilities
        self.target_flick_prob = target_flick_prob
        self.tgt_avoid_agent = tgt_avoid_agent
        self.tgt_trans_noise = tgt_trans_noise       # is not implemented here
        self.agt_trans_noise = agent_trans_noise   # is not implemented here

        self.action_space = spaces.Discrete(len(ACTIONS))
        if obs_one_hot:
            self.observation_space = spaces.MultiBinary((self.x_len*self.y_len)*2)
        else:
            self.observation_space = spaces.Discrete(len(grid_dim)*2)

        self.obs_one_hot = obs_one_hot
        self.viewer = None

    @property
    def obs_size(self):
        return [self.observation_space.n]*2

    @property
    def n_action(self):
        return [self.action_space.n]*2

    def action_space_sample(self, idx):
        return np.random.randint(self.n_action[idx])

    def createAgents(self):
        raise NotImplementedError

    def createTargets(self):
        raise NotImplementedError

    def reset(self, debug=False):
        self.step_n = 0
        self.createAgents()
        # collect agents' positions
        self.createTargets()

        assert self.target_positions.shape == (self.n_target, 2) 

        self.primitive_obs = self.get_obs()
        self.macro_obs = [obs for obs in self.primitive_obs]

        if debug:
            self.render()

        return self.macro_obs

    def step(self, actions, debug=False):
        raise NotImplementedError

    def get_obs(self):

        if self.obs_one_hot:
            agt_pos_obs = self.one_hot_positions(self.agent_positions)
            tgt_pos_obs = self.one_hot_positions(self.target_positions)
        else:
            agt_pos_obs = self.normalize_positions(self.agent_positions)
            tgt_pos_obs = self.normalize_positions(self.target_positions)

            if self.n_agent > 1 and self.n_target == 1:
                tgt_pos_obs = np.tile(tgt_pos_obs, (self.n_agent, 1))

        tgt_pos_obs = self.flick(tgt_pos_obs, prob=self.target_flick_prob)

        obs = np.concatenate([agt_pos_obs, tgt_pos_obs], axis=1)

        return obs

    ###################################################################################################
    # Helper functions

    def get_tgt_moves(self, single=False):
        assert self.target_positions.shape[0] == 1
        moves = self.wrap_target_positions(DIRECTION + self.target_positions)
        if single:
            cl_agt_idx = np.linalg.norm(self.agent_positions-self.target_positions, axis=1).argmin()
            h = np.linalg.norm(self.agent_positions[cl_agt_idx]-moves, axis=1)
        else:
            h_0 = np.linalg.norm(self.agent_positions[0]-moves, axis=1)
            h_1 = np.linalg.norm(self.agent_positions[1]-moves, axis=1)
            h = h_0 + h_1
        return np.random.choice(np.where(h == h.max())[0], size=1)

    def move(self, positions, directions, noise=0):
        translations = np.stack([self.translation(d, noise=noise) for d in directions])
        positions += translations
        return self.wrap_target_positions(positions)

    def translation(self, direction, noise=0.0):
        return TRANSLATION_TABLE[direction][np.random.choice(3, p=[noise/2, 1-noise, noise/2])]

    def wrap_agent_positions(self):
        return np.stack([agent.position for agent in self.agents])

    def wrap_target_positions(self, positions):
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([X % self.x_len, Y % self.y_len], axis=1)

    def target_captured(self):
        return all(np.array_equal(a, t) for a, t in zip(self.agent_positions, self.respective_target_positions))

    def decode_position(self, agent, obs):
        a_p = agent._get_position_from_one_hot(obs[0:self.x_len*self.y_len])
        t_p = agent._get_position_from_one_hot(obs[self.x_len*self.y_len:])
        return np.concatenate([a_p, t_p])

    @property
    def respective_target_positions(self):
        if self.multi_task:
            return self.target_positions
        else:
            return (self.target_positions[0] for _ in range(self.n_agent))

    def flick(self, N, prob=0.3):
        mask = np.random.random(N.shape[0]).reshape(N.shape[0], -1) > prob
        if self.obs_one_hot:
            return N * mask
        else:
            flicker = np.stack([np.array([-1,-1]) for _ in range(N.shape[0])])
            return N * mask + flicker * np.logical_not(mask)
    
    def one_hot_positions(self, positions):
        one_hot_vector = np.zeros((self.n_agent, self.x_len*self.y_len))
        index = positions[:,1] * self.y_len + positions[:,0]
        one_hot_vector[np.arange(self.n_agent), index.astype(int)] = 1
        return one_hot_vector
    
    def normalize_positions(self, positions):
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([
                X / self.x_len,
                Y / self.y_len], axis=1)

    def render(self, mode='human'):

        screen_width = 8 * 100
        screen_height = 8 * 100

        scale = 8 / self.y_len

        agent_size = 40.0
        agent_in_size = 35.0
        agent_clrs = [((0.15,0.15,0.65), (0.0, 0.4,0.8)),((0.15,0.65,0.15), (0.0,0.8,0.4))]

        target_l = 80.0
        target_w = 26.0
        target_in_l = 70.0
        target_in_w = 16.0
        target_clrs = ((0.65,0.15,0.15), (1.0, 0.5,0.5))

        if self.viewer is None:
            from rlmamr.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #-------------------draw agents
            self.render_agents = []
            #agent_clrs = [(0.0,153.0/255.0,0.0), (0.0,0.0,153.0/255.0)]
            for i in range(self.n_agent):
                agent = rendering.make_circle(radius=agent_size*scale)
                agent.set_color(*agent_clrs[i][0])
                agent_trans = rendering.Transform(translation=((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale))
                agent.add_attr(agent_trans)
                self.render_agents.append(agent_trans)
                self.viewer.add_geom(agent)

            #-------------------draw agents contours
            for i in range(self.n_agent):
                agent = rendering.make_circle(radius=agent_in_size*scale)
                agent.set_color(*agent_clrs[i][1])
                agent_trans = rendering.Transform(translation=((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale))
                agent.add_attr(agent_trans)
                self.render_agents.append(agent_trans)
                self.viewer.add_geom(agent)

            #-------------------draw target
            tgt_l = rendering.FilledPolygon([(-target_w/2.0*scale,-target_l/2.0*scale), (-target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,-target_l/2.0*scale)])
            tgt_l.set_color(*target_clrs[0])
            self.tgt_l_trans = rendering.Transform(translation=tuple((self.target_positions[0]+0.5)*100*scale), rotation=np.pi/4)
            tgt_l.add_attr(self.tgt_l_trans)
            self.viewer.add_geom(tgt_l)

            tgt_r = rendering.FilledPolygon([(-target_w/2.0*scale,-target_l/2.0*scale), (-target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,-target_l/2.0*scale)])
            tgt_r.set_color(*target_clrs[0])
            self.tgt_r_trans = rendering.Transform(translation=tuple((self.target_positions[0]+0.5)*100*scale), rotation=-np.pi/4)
            tgt_r.add_attr(self.tgt_r_trans)
            self.viewer.add_geom(tgt_r)

            #-------------------draw target----contours
            tgt_l = rendering.FilledPolygon([(-target_in_w/2.0*scale,-target_in_l/2.0*scale), (-target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,-target_in_l/2.0*scale)])
            tgt_l.set_color(*target_clrs[1])
            self.tgt_lc_trans = rendering.Transform(translation=tuple((self.target_positions[0]+0.5)*100*scale), rotation=np.pi/4)
            tgt_l.add_attr(self.tgt_lc_trans)
            self.viewer.add_geom(tgt_l)

            tgt_r = rendering.FilledPolygon([(-target_in_w/2.0*scale,-target_in_l/2.0*scale), (-target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,-target_in_l/2.0*scale)])
            tgt_r.set_color(*target_clrs[1])
            self.tgt_rc_trans = rendering.Transform(translation=tuple((self.target_positions[0]+0.5)*100*scale), rotation=-np.pi/4)
            tgt_r.add_attr(self.tgt_rc_trans)
            self.viewer.add_geom(tgt_r)

            #-------------------draw line-----------------
            for l in range(1, self.y_len):
                line = rendering.Line((0.0, l*100*scale), (screen_width, l*100*scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.y_len):
                line = rendering.Line((l*100*scale, 0.0), (l*100*scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

        self.tgt_l_trans.set_translation((self.target_positions[0][0]+0.5)*100*scale, (self.target_positions[0][1]+0.5)*100*scale)
        self.tgt_r_trans.set_translation((self.target_positions[0][0]+0.5)*100*scale, (self.target_positions[0][1]+0.5)*100*scale)
        self.tgt_lc_trans.set_translation((self.target_positions[0][0]+0.5)*100*scale, (self.target_positions[0][1]+0.5)*100*scale)
        self.tgt_rc_trans.set_translation((self.target_positions[0][0]+0.5)*100*scale, (self.target_positions[0][1]+0.5)*100*scale)

        for i in range(self.n_agent):
            self.render_agents[i].set_translation((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale)
        for i in range(self.n_agent):
            self.render_agents[i+2].set_translation((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class CaptureTarget_MA_v1(CaptureTarget_MA):

    """M_T_T macro_action continues moving towards to the previous target's
       location if the taget is flicked."""

    def __init__(self, *args, **kwargs):
        super(CaptureTarget_MA_v1, self).__init__(*args, **kwargs)

    def createAgents(self):
        self.agents = [Agent_v1(int(i), self.grid_dim, agent_trans_noise=self.agt_trans_noise) for i in range(self.n_agent)]
        self.agent_positions = self.wrap_agent_positions()

    def createTargets(self):
        self.target_positions = np.stack([Agent_v1.rand_position(*self.grid_dim) for _ in range(self.n_target)])

    def step(self, actions, debug=False):
        self.step_n += 1
        assert len(actions) == self.n_agent
        cur_actions = []
        cur_actions_done = []

        for idx, agent in enumerate(self.agents):
            agent.step(actions[idx], self.primitive_obs[idx])

            cur_actions.append(agent.cur_action)
            cur_actions_done.append(int(agent.cur_action_done))

        # collect agents' positions
        self.agent_positions = self.wrap_agent_positions()

        if not self.target_captured():
            if not self.tgt_avoid_agent:
                target_directions = np.random.randint(len(TRANSLATION_TABLE), size=self.n_target)
            else:
                target_directions = self.get_tgt_moves()
            self.target_positions = self.move(self.target_positions, target_directions)

        # check if target is captured
        won = self.target_captured()
        r = float(won)

        # update obs
        self.primitive_obs = self.get_obs()
        obs = [self.primitive_obs[idx] if cur_actions_done[idx] else self.macro_obs[idx] for idx in range(self.n_agent)]
        self.macro_obs = obs

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            for i in range(self.n_agent):
                print(cur_actions[i])
                print("Agent_"+str(i)+" \t action \t\t{}".format(ACTIONS[cur_actions[i]]))
                print("        \t action_t_left \t\t{}".format(self.agents[i].cur_action_time_left))
                print("        \t action_done \t\t{}".format(self.agents[i].cur_action_done))
                print(" ")
            print("Macro-Observations list:")
            for i in range(self.n_agent):
                print("Agent_"+str(i)+" \t\t\t{}".format(obs[i] if not self.obs_one_hot else self.decode_position(self.agents[i], obs[i])))
            print("")
            print("Primitive-Observations list:")
            for i in range(self.n_agent):
                print("Agent_"+str(i)+" \t\t\t{}".format(self.primitive_obs[i] if not self.obs_one_hot else self.decode_position(self.agents[i], self.primitive_obs[i])))

        return cur_actions, obs, r, int(won or self.step_n >= self.terminate_step), cur_actions_done
