# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
import cv2
import gym
import logging

from environment import environment

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2
COMMAND_RENDER    = 3

logger = logging.getLogger("StRADRL.gym_environment")

"""
def preprocess_frame(observation):
  # observation shape = (210, 160, 3)
  observation = observation.astype(np.float32)
  resized_observation = cv2.resize(observation, (84, 84))
  resized_observation = resized_observation / 255.0
  return resized_observation
"""
def worker(conn, env_name):
  env = gym.make(env_name)
  env.reset()
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      obs = env.reset()
      #state = preprocess_frame(obs)
      state = obs
      conn.send(state)
    elif command == COMMAND_ACTION:
      reward = 0
      for i in range(1):
        obs, r, terminal, _ = env.step(arg)
        reward += r
        if terminal:
          break
      #state = preprocess_frame(obs)
      state = obs
      conn.send([state, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    elif command == COMMAND_RENDER:
        env.render()
    else:
      print("bad command: {}".format(command))
  env.close()
  conn.send(0)
  conn.close()


class GymEnvironment(environment.Environment):
  @staticmethod
  def get_action_size(env_name):
    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.Box):
      dim = env.action_space.shape[0]
      discrete = False
    elif isintance(env.action_space, gym.spaces.n):
      dim = env.action_space.n
      discrete = True
    env.close()
    return dim, discrete
    
  @staticmethod  
  def get_obs_size(env_name):
    env = gym.make(env_name)
    obs_size = np.size(env.reset())
    env.close()
    return obs_size
  
  def __init__(self, env_name):
    environment.Environment.__init__(self)
    
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn, env_name))
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    self.last_state = self.conn.recv()
    return self.last_state

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    logger.warn("gym environment stopped")

  def process(self, action):
    self.conn.send([COMMAND_ACTION, action])
    #logger.debug("action:{}".format(action))
    try:
        state, reward, terminal = self.conn.recv()
    except TypeError:
        logger.warn("!! Received single int value, environment probably terminated !!")
        logger.warn("Closing now")
        self.conn.close()
        self.proc.join()
        logger.warn("gym environment stopped")
    
    #pixel_change = self._calc_pixel_change(state, self.last_state)
    pixel_change = []
    self.last_state = state
    return state, reward, terminal, pixel_change
    
  def render(self):
    self.conn.send([COMMAND_RENDER, 0])
