# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
import logging
import gym

from environment import environment

logger = logging.getLogger('StRADRL.mujoco_env')

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def worker(conn, render):
    env = gym.make('Humanoid-v1')
    conn.send(COMMAND_RESET)
    
    while True:
        command, arg = conn.recv()

        if command == COMMAND_RESET:
            obs = env.reset()
            if render:
                env.render()
            #logger.warn("episode was reset")
            #logger.debug("reset output:{}".format(obs))
            conn.send(obs)
        elif command == COMMAND_ACTION:
            #logger.debug("action argument:".format(arg))
            obs, reward, terminal, _ = env.step(arg)
            if render:
                env.render()
            conn.send([obs, reward, terminal])
        elif command == COMMAND_TERMINATE:
            break
        else:
            logger.warn("bad command: {}".format(command))
    env.close()
    conn.send(0)
    conn.close()

"""
{'_env_closer_id': 1, 
  '_max_episode_seconds': None, 
  'action_space': Box(17,), 
  '_spec': None, 
  '_closed': False, 
  '_elapsed_steps': 0, 
  '_episode_started_at': None, 
  'observation_space': Box(376,), 
  'reward_range': (-inf, inf), 
  'env': <gym.envs.mujoco.humanoid.HumanoidEnv object at 0x7f9741b25f10>, 
  '_max_episode_steps': 1000, 
  'metadata': {'video.frames_per_second': 67, 
               'render.modes': ['human', 'rgb_array']}
  }
"""

class MujocoEnvironment(environment.Environment):
    @staticmethod
    def get_action_size():
        env = gym.make('Humanoid-v1')
        action_size = np.asarray(env.action_space.high).shape[0]
        #logger.debug("acsize:{}".format(action_size))
        logger.debug("action_size:{}".format(action_size))
        env.close()
        return action_size
    
    def __init__(self):
        logger.warn("!! hardcoding set render to true here !!")
        render = True
    
        environment.Environment.__init__(self)
        self.conn, child_conn = Pipe()
        self.proc = Process(target=worker, args=(child_conn, render))
        self.proc.start()
        self.conn.recv()


    def reset(self):
        self.conn.send([COMMAND_RESET, 0])
        obs = self.conn.recv()
        #logger.debug("obs: {}".format(obs))
        
        self.last_state = obs
        
        #logger.debug("obs shape: {}".format(self.last_state.shape))
        self.last_action = 0
        self.last_reward = 0
        last_action_reward = np.zeros([self.action_size+1])
        
        return self.last_state, last_action_reward
        
    def stop(self):
        self.conn.send([COMMAND_TERMINATE, 0])
        ret = self.conn.recv()
        self.conn.close()
        self.proc.join()
        logger.info("lab environment stopped")
    
    def process(self, action):
        real_action = -0.4 + 0.8*action
        #logger.debug(real_action)
        self.conn.send([COMMAND_ACTION, real_action])
        obs, reward, terminal = self.conn.recv()
        if not terminal:
            state = obs
        else:
            state = self.last_state
        
        pixel_change = [] #self._calc_pixel_change(state, self.last_state)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        return state, reward, terminal, pixel_change
        
        
        
