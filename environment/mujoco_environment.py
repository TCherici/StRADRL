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
            logger.debug("reset output:{}".format(obs))
            conn.send(obs)
        elif command == COMMAND_ACTION:
            logger.debug("action argument:".format(arg))
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

class MujocoEnvironment(environment.Environment):
    @staticmethod
    def get_action_size(env_name):
        
        return len(env.action_space)
    
    def __init__(self):
        environment.Environment.__init__(self)
        self.conn, child_conn = Pipe()
        self.proc = Process(target=worker, args=(child_conn))
        self.proc.start()
        self.conn.recv()


    def reset(self):
        self.conn.send([COMMAND_RESET, 0])
        obs = self.conn.recv()
        #logger.debug("obs: {}".format(obs))
        
        self.last_state = obs
        
        logger.debug("obs shape: {}".format(self.last_state.shape))
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
        real_action = [action]
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
        
        
        
