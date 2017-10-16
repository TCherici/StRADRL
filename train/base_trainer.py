# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.signal
import random
import time
import sys
import logging
import six.moves.queue as queue
from collections import namedtuple

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import Experience, ExperienceFrame

logger = logging.getLogger("StRADRL.base_trainer")

LOG_INTERVAL = 1000
PERFORMANCE_LOG_INTERVAL = 1000

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class BaseTrainer(object):
    def __init__(self,
               runner,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               env_type,
               env_name,
               entropy_beta,
               local_t_max,
               gamma,
               experience_history_size,
               max_global_time_step,
               device):
        self.runner = runner
        self.learning_rate_input = learning_rate_input
        self.env_type = env_type
        self.env_name = env_name
        self.local_t_max = local_t_max
        self.gamma = gamma
        self.experience_history_size = experience_history_size
        self.max_global_time_step = max_global_time_step
        self.action_size = Environment.get_action_size(env_type, env_name)
        self.local_network = UnrealModel(self.action_size,
                                         0,
                                         False,
                                         False,
                                         False,
                                         0,
                                         entropy_beta,
                                         device)
        self.local_network.prepare_loss()
        
        self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                           global_network.get_vars(),
                                                           self.local_network.get_vars())
        self.sync = self.local_network.sync_from(global_network)
        self.experience = Experience(self.experience_history_size)
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0
        self.summary_writer = None
        self.local_steps = 0
    
    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate
        
    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)
    
    def set_start_time(self, start_time):
        self.start_time = start_time
        
    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        logger.debug("pulled batch from rollout, length:{}".format(len(rollout.rewards)))
        return rollout
    
    
    #@TODO get this whole thing working
    def _process_base(self, sess, global_t, summary_writer, summary_op, score_input):
        # [Base A3C]
        states = []
        last_action_rewards = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        start_lstm_state = self.local_network.base_lstm_state_out

        # t_max times loop
        for _ in range(self.local_t_max):
            # Prepare last action reward
            last_action = self.environment.last_action
            last_reward = self.environment.last_reward
            last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                          self.action_size,
                                                                          last_reward)
            
            pi_, value_ = self.local_network.run_base_policy_and_value(sess,
                                                                       self.environment.last_state,
                                                                       last_action_reward)
            
            
            action = self.choose_action(pi_)

            states.append(self.environment.last_state)
            last_action_rewards.append(last_action_reward)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                logger.info("localtime={}".format(self.local_t))
                logger.info("pi={}".format(pi_))
                logger.info(" V={}".format(value_))

            prev_state = self.environment.last_state

            # Process game
            new_state, reward, terminal, pixel_change = self.environment.process(action)
            frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                                    last_action, last_reward)

            # Store to experience
            self.experience.add_frame(frame)

            self.episode_reward += reward

            rewards.append( reward )

            self.local_t += 1

            if terminal:
              terminal_end = True
              print("score={}".format(self.episode_reward))

              self._record_score(sess, summary_writer, summary_op, score_input,
                                 self.episode_reward, global_t)
                
              self.episode_reward = 0
              self.environment.reset()
              self.local_network.reset_state()
              break

        R = 0.0
        if not terminal_end:
          R = self.local_network.run_base_value(sess, new_state, frame.get_last_action_reward(self.action_size))

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_adv = []
        batch_R = []

        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
          R = ri + self.gamma * R
          adv = R - Vi
          a = np.zeros([self.action_size])
          a[ai] = 1.0

          batch_si.append(si)
          batch_a.append(a)
          batch_adv.append(adv)
          batch_R.append(R)

        batch_si.reverse()
        batch_a.reverse()
        batch_adv.reverse()
        batch_R.reverse()

        return batch_si, last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state

        
    
