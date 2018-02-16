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
from model.fc_model import UnrealModel
from train.experience import Experience, ExperienceFrame

logger = logging.getLogger("StRADRL.base_trainer")

LOG_INTERVAL = 2000
PERFORMANCE_LOG_INTERVAL = 10000

Batch = namedtuple("Batch", ["si", "a", "rewards", "adv", "discrewards", "terminal", "pc"])

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    batch_reward = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = batch_reward + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    batch_pc = np.asarray(rollout.pixel_changes)
    return Batch(batch_si, batch_a, batch_reward, batch_adv, batch_r, rollout.terminal, batch_pc)

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
               gamma,
               experience,
               max_global_time_step,
               device,
               value_lambda):
        self.runner = runner
        self.learning_rate_input = learning_rate_input
        self.env_type = env_type
        self.env_name = env_name
        self.gamma = gamma
        self.max_global_time_step = max_global_time_step
        self.action_size, self.is_discrete = Environment.get_action_size(env_type, env_name)
        self.obs_size = Environment.get_obs_size(env_type, env_name)
        self.global_network = global_network
        self.local_network = UnrealModel(self.action_size,
                                         self.is_discrete,
                                         self.obs_size,
                                         1,
                                         entropy_beta,
                                         device,
                                         value_lambda=value_lambda)

        self.local_network.prepare_loss()
        
        self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                                    self.global_network.get_vars(),
                                                                     self.local_network.get_vars())
        self.sync = self.local_network.sync_from(self.global_network, name="base_trainer")
        self.experience = experience
        self.local_t = 0
        self.next_log_t = 0
        self.next_performance_t = PERFORMANCE_LOG_INTERVAL
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0
        # trackers for the experience replay creation
        self.last_state = None
        self.last_action = 0
        self.last_reward = 0
        self.ep_ploss = 0.
        self.ep_vloss = 0.
        self.ep_entr = []
        self.ep_grad = []
        self.ep_l = 0
        
    
    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate
        
    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)
    
    def set_start_time(self, start_time, global_t):
        self.start_time = start_time
        self.local_t = global_t
        
    def pull_batch_from_queue(self):
        """
        take a rollout from the queue of the thread runner.
        """
        rollout_full = False
        count = 0
        while not rollout_full:
            if count == 0:
                rollout = self.runner.queue.get(timeout=600.0)
                count += 1
            else:
                try:
                    rollout.extend(self.runner.queue.get_nowait())
                    count += 1
                except queue.Empty:
                    #logger.warn("!!! queue was empty !!!")
                    continue
            if count == 5 or rollout.terminal:
                rollout_full = True
        #logger.debug("pulled batch from rollout, length:{}".format(len(rollout.rewards)))
        return rollout
        
    def _print_log(self, global_t):
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            logger.info("Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
            global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
    
    def _add_batch_to_exp(self, batch):
        # if we just started, copy the first state as last state
        if self.last_state is None:
                self.last_state = batch.si[0]
        #logger.debug("adding batch to exp. len:{}".format(len(batch.si)))
        for k in range(len(batch.si)):
            state = batch.si[k]
            action = batch.a[k]#np.argmax(batch.a[k])
            reward = batch.rewards[k]

            self.episode_reward += reward
            pixel_change = batch.pc[k]
            #logger.debug("k = {} of {} -- terminal = {}".format(k,len(batch.si), batch.terminal))
            if k == len(batch.si)-1 and batch.terminal:
                terminal = True
            else:
                terminal = False
            frame = ExperienceFrame(state, reward, action, terminal, pixel_change,
                            self.last_action, self.last_reward)
            self.experience.add_frame(frame)
            self.last_state = state
            self.last_action = action
            self.last_reward = reward
            
        if terminal:
            total_ep_reward = self.episode_reward
            self.episode_reward = 0
            return total_ep_reward
        else:
            return None
            
    
    def process(self, sess, global_t, summary_writer, summary_op, summary_values, base_lambda):
        sess.run(self.sync)
        cur_learning_rate = self._anneal_learning_rate(global_t)
        # Copy weights from shared to local
        #logger.debug("Syncing to global net -- current learning rate:{}".format(cur_learning_rate))
        #logger.debug("local_t:{} - global_t:{}".format(self.local_t,global_t))


        # get batch from process_rollout
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=base_lambda)
        self.local_t += len(batch.si)


        #logger.debug("si:{}".format(batch.si.shape))
        feed_dict = {
            self.local_network.base_input: batch.si,
            self.local_network.base_a: batch.a,
            self.local_network.base_adv: batch.adv,
            self.local_network.base_r: batch.discrewards,
            # [common]
            self.learning_rate_input: cur_learning_rate
        }
        #logger.debug(batch.__dict__)
        
        # Calculate gradients and copy them to global network.
        [_, grad], policy_loss, value_loss, entr, baseinput, policy, value = sess.run(
                                              [self.apply_gradients,
                                              self.local_network.policy_loss,
                                              self.local_network.value_loss,
                                              self.local_network.entropy, 
                                              self.local_network.base_input,
                                              self.local_network.base_pi,
                                              self.local_network.base_v],
                                     feed_dict=feed_dict )
        self.ep_l += batch.si.shape[0]
        self.ep_ploss += policy_loss
        self.ep_vloss += value_loss
        self.ep_entr.append(entr)

        self.ep_grad.append(grad)
        # add batch to experience replay
        total_ep_reward = self._add_batch_to_exp(batch)
        if total_ep_reward is not None:
            laststate = baseinput[np.newaxis,-1,...]
            summary_str = sess.run(summary_op, feed_dict={summary_values[0]: total_ep_reward,
                                                          summary_values[1]: self.ep_l,
                                                          summary_values[2]: self.ep_ploss/self.ep_l,
                                                          summary_values[3]: self.ep_vloss/self.ep_l,
                                                          summary_values[4]: np.mean(self.ep_entr),
                                                          summary_values[5]: np.mean(self.ep_grad),
                                                          summary_values[6]: cur_learning_rate})#,
                                                          #summary_values[7]: laststate})
            summary_writer.add_summary(summary_str, global_t)
            summary_writer.flush()
                    
            if self.local_t > self.next_performance_t:
                self._print_log(global_t)
                self.next_performance_t += PERFORMANCE_LOG_INTERVAL
                    
            if self.local_t >= self.next_log_t:
                logger.info("localtime={}".format(self.local_t))
                logger.info("action={}".format(self.last_action))
                logger.info("policy={}".format(policy[-1]))
                logger.info("V={}".format(np.mean(value)))
                logger.info("ep score={}".format(total_ep_reward))
                self.next_log_t += LOG_INTERVAL
            
            #try:
            #sess.run(self.sync)
            #except Exception:
            #    logger.warn("--- !! parallel syncing !! ---")
            self.ep_l = 0
            self.ep_ploss = 0.
            self.ep_vloss = 0.
            self.ep_entr = []
            self.ep_grad = []
            
        # Return advanced local step size
        diff_global_t = self.local_t - global_t
        return diff_global_t
        

