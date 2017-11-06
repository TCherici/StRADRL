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

logger = logging.getLogger("StRADRL.aux_trainer")


Batch = namedtuple("Batch", ["si", "a", "a_r", "adv", "r", "terminal", "features"])#, "pc"])

class AuxTrainer(object):
    def __init__(self,
                global_network,
                thread_index,
                use_pixel_change,
                use_value_replay,
                use_reward_prediction,
                use_temporal_coherence,
                pixel_change_lambda,
                temporal_coherence_lambda,
                initial_learning_rate,
                learning_rate_input,
                grad_applier,
                env_type,
                env_name,
                local_t_max,
                gamma,
                gamma_pc,
                experience,
                max_global_time_step,
                device):
                
                
        self.use_pixel_change = use_pixel_change   
        self.use_value_replay = use_value_replay
        self.use_reward_prediction = use_reward_prediction  
        self.use_temporal_coherence = use_temporal_coherence        
        self.learning_rate_input = learning_rate_input
        self.env_type = env_type
        self.env_name = env_name
        self.local_t_max = local_t_max
        self.gamma = gamma
        self.gamma_pc = gamma_pc
        self.experience = experience
        self.max_global_time_step = max_global_time_step
        self.action_size = Environment.get_action_size(env_type, env_name)
        self.local_network = UnrealModel(self.action_size,
                                         thread_index,
                                         0.,
                                         device,
                                         use_pixel_change,
                                         use_value_replay,
                                         use_reward_prediction,
                                         use_temporal_coherence,
                                         pixel_change_lambda,
                                         temporal_coherence_lambda,
                                         use_base=True)
        self.local_network.prepare_loss()
        
        #logger.debug("ln.total_loss:{}".format(self.local_network.total_loss))
        
        self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                           global_network.get_vars(),
                                                           self.local_network.get_vars())
        self.sync = self.local_network.sync_from(global_network)
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0
        # trackers for the experience replay creation
        self.last_action = np.zeros(self.action_size)
        self.last_reward = 0
        
    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate
        
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
        
    def _process_base(self, sess, policy, gamma, lambda_=1.0):
        # base A3C from experience replay
        experience_frames = self.experience.sample_sequence(self.local_t_max+1)
        batch_si = []
        batch_a = []
        rewards = []
        action_reward = []
        batch_features = []
        values = []
        last_state = experience_frames[0].state
        last_action_reward = experience_frames[0].concat_action_and_reward(experience_frames[0].action,
                                                                        self.action_size,
                                                                        experience_frames[0].reward)
        for frame in range(1,len(experience_frames)):
            state = experience_frames[frame].state
            batch_si.append(state)
            action = experience_frames[frame].action
            reward = experience_frames[frame].reward
            a_r = experience_frames[frame].concat_action_and_reward(action, self.action_size, reward)
            action_reward.append(a_r)
            batch_a.append(a_r[:-1])
            rewards.append(reward)
            #logger.debug("last_action_reward:{}".format(last_action_reward))
            _, value, features = policy.run_base_policy_and_value(sess, last_state, last_action_reward)
            batch_features.append(features)
            values.append(value)
            last_state = state
            last_action_reward = action_reward[-1]

        if not experience_frames[-1].terminal:
           r = policy.run_base_value(sess, last_state, last_action_reward)
        else:
           r = 0.
                
        vpred_t = np.asarray(values + [r])
        rewards_plus_v = np.asarray(rewards + [r])
        batch_r = self.discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = self.discount(delta_t, gamma * lambda_)

        
        start_features = batch_features[0]

        return Batch(batch_si, batch_a, action_reward, batch_adv, batch_r, experience_frames[-1].terminal, start_features)
        
    def _process_pc(self, sess):
        # [pixel change]
        # Sample 20+1 frame (+1 for last next state)
        pc_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
        # Revese sequence to calculate from the last
        pc_experience_frames.reverse()

        batch_pc_si = []
        batch_pc_a = []
        batch_pc_R = []
        batch_pc_last_action_reward = []
        
        pc_R = np.zeros([20,20], dtype=np.float32)
        if not pc_experience_frames[0].terminal:
            pc_R = self.local_network.run_pc_q_max(sess,
                                                 pc_experience_frames[0].state,
                                                 pc_experience_frames[0].get_last_action_reward(self.action_size))


        for frame in pc_experience_frames[1:]:
            pc_R = frame.pixel_change + self.gamma_pc * pc_R
            a = np.zeros([self.action_size])
            a[frame.action] = 1.0
            last_action_reward = frame.get_last_action_reward(self.action_size)
              
            batch_pc_si.append(frame.state)
            batch_pc_a.append(a)
            batch_pc_R.append(pc_R)
            batch_pc_last_action_reward.append(last_action_reward)

        batch_pc_si.reverse()
        batch_pc_a.reverse()
        batch_pc_R.reverse()
        batch_pc_last_action_reward.reverse()
        
        return batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R
        
    def _process_vr(self, sess):
        # [Value replay]
        # Sample 20+1 frame (+1 for last next state)
        vr_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
        # Revese sequence to calculate from the last
        vr_experience_frames.reverse()

        batch_vr_si = []
        batch_vr_R = []
        batch_vr_last_action_reward = []

        vr_R = 0.0
        if not vr_experience_frames[0].terminal:
            vr_R = self.local_network.run_vr_value(sess,
                                                 vr_experience_frames[0].state,
                                                 vr_experience_frames[0].get_last_action_reward(self.action_size))
        
        # t_max times loop
        for frame in vr_experience_frames[1:]:
            vr_R = frame.reward + self.gamma * vr_R
            batch_vr_si.append(frame.state)
            batch_vr_R.append(vr_R)
            last_action_reward = frame.get_last_action_reward(self.action_size)
            batch_vr_last_action_reward.append(last_action_reward)

        batch_vr_si.reverse()
        batch_vr_R.reverse()
        batch_vr_last_action_reward.reverse()

        return batch_vr_si, batch_vr_last_action_reward, batch_vr_R
        
    def _process_rp(self):
        # [Reward prediction]
        rp_experience_frames = self.experience.sample_rp_sequence()
        # 4 frames

        batch_rp_si = []
        batch_rp_c = []
        
        for i in range(3):
            batch_rp_si.append(rp_experience_frames[i].state)

        # one hot vector for target reward
        r = rp_experience_frames[3].reward
        rp_c = [0.0, 0.0, 0.0]
        if r == 0:
          rp_c[0] = 1.0 # zero
        elif r > 0:
          rp_c[1] = 1.0 # positive
        else:
          rp_c[2] = 1.0 # negative
        batch_rp_c.append(rp_c)
        return batch_rp_si, batch_rp_c
        
    def _process_tc(self):
        # [temporal coherence]
        tc_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
        # Revese sequence to calculate from the last
        batch_tc_input1 = []
        batch_tc_input2 = []
        for frame in range(len(tc_experience_frames)-1):
            batch_tc_input1.append(tc_experience_frames[frame+1].state)
            batch_tc_input2.append(tc_experience_frames[frame].state)
        return batch_tc_input1, batch_tc_input2

    def process(self, sess, global_t):

        cur_learning_rate = self._anneal_learning_rate(global_t)
        if global_t % 10000 == 0:
            logger.debug("Syncing to global net -- current learning rate:{}".format(cur_learning_rate))
            # Copy weights from shared to local
            sess.run( self.sync )
            
        batch = self._process_base(sess, self.local_network, self.gamma)
        
        feed_dict = {
                self.local_network.base_input: batch.si,
                self.local_network.base_last_action_reward_input: batch.a_r,
                self.local_network.base_a: batch.a,
                self.local_network.base_adv: batch.adv,
                self.local_network.base_r: batch.r,
                self.local_network.base_initial_lstm_state: batch.features,
                # [common]
                self.learning_rate_input: cur_learning_rate
        }
        
        # [Pixel change]
        if self.use_pixel_change:
            batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R = self._process_pc(sess)

            pc_feed_dict = {
                self.local_network.pc_input: batch_pc_si,
                self.local_network.pc_last_action_reward_input: batch_pc_last_action_reward,
                self.local_network.pc_a: batch_pc_a,
                self.local_network.pc_r: batch_pc_R,
                # [common]
                self.learning_rate_input: cur_learning_rate
            }
            feed_dict.update(pc_feed_dict)

        # [Value replay]
        if self.use_value_replay:
            batch_vr_si, batch_vr_last_action_reward, batch_vr_R = self._process_vr(sess)
          
            vr_feed_dict = {
                self.local_network.vr_input: batch_vr_si,
                self.local_network.vr_last_action_reward_input : batch_vr_last_action_reward,
                self.local_network.vr_r: batch_vr_R,
                # [common]
                self.learning_rate_input: cur_learning_rate
            }
            feed_dict.update(vr_feed_dict)

        # [Reward prediction]
        if self.use_reward_prediction:
            batch_rp_si, batch_rp_c = self._process_rp()
            rp_feed_dict = {
                self.local_network.rp_input: batch_rp_si,
                self.local_network.rp_c_target: batch_rp_c,
                # [common]
                self.learning_rate_input: cur_learning_rate
            }
            feed_dict.update(rp_feed_dict)
        
        # [Temporal coherence]
        if self.use_temporal_coherence:
            batch_tc_input1, batch_tc_input2 = self._process_tc()
            tc_feed_dict = {
                self.local_network.tc_input1: np.asarray(batch_tc_input1),
                self.local_network.tc_input2: np.asarray(batch_tc_input2)
            }
            
            feed_dict.update(tc_feed_dict)

        # Calculate gradients and copy them to global netowrk.
        sess.run( self.apply_gradients, feed_dict=feed_dict )


