# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time
import logging

from helper import logger_init, generate_id
from environment.environment import Environment
from model.model import UnrealModel
#from model.base import BaseModel
from train.experience import Experience
from train.adam_applier import AdamApplier
from train.base_trainer import BaseTrainer
from train.aux_trainer import AuxTrainer
from queuer import RunnerThread
from options import get_options

# get command line args
flags = get_options("training")
# setup logger
logger = logging.getLogger('StRADRL.newmain')


RUN_ID = generate_id()
if flags.training_name:
    TRAINING_NAME = flags.training_name
else:
    TRAINING_NAME = RUN_ID
LOG_LEVEL = 'debug'
CONTINUE_TRAINING = False

USE_GPU = True
visualise = False


class Application(object):
    def __init__(self):
        pass
        
    def base_train_function(self):
        """ Train routine for base_trainer. """
        
        trainer = self.base_trainer
        
        # set start_time
        trainer.set_start_time(self.start_time, self.global_t)
      
        while True:
            if self.stop_requested:
                break
            if self.terminate_requested:
                trainer.stop()
                break
            if self.global_t > flags.max_time_step:
                trainer.stop()
                break
            if self.global_t > self.next_save_steps:
                # Save checkpoint
                logger.debug("Steps:{}".format(self.global_t))
                logger.debug(self.next_save_steps)
                
                self.save()
            
            diff_global_t = trainer.process(self.sess,
                                          self.global_t,
                                          self.summary_writer,
                                          self.summary_op,
                                          self.summary_values)
            self.global_t += diff_global_t
            
    def aux_train_function(self, aux_index):
        """ Train routine for aux_trainer. """
        
        trainer = self.aux_trainers[aux_index]
        
        while True:
            if self.global_t < 500:
                continue
            if self.stop_requested:
                continue
            if self.terminate_requested:
                trainer.stop()
                break
            if self.global_t > flags.max_time_step:
                trainer.stop()
                break
            
            diff_aux_t = trainer.process(self.sess,
                                        self.global_t,
                                        self.aux_t,
                                        self.summary_writer,
                                        self.summary_op_aux,
                                        self.summary_aux)
            self.aux_t += diff_aux_t
            #logger.debug("aux_t:{}".format(self.aux_t))
            
            
    def run(self):
        device = "/cpu:0"
        if USE_GPU:
            device = "/gpu:0"
        logger.debug("start App")
        initial_learning_rate = flags.initial_learning_rate
        
        self.global_t = 0
        self.aux_t = 0
        self.stop_requested = False
        self.terminate_requested = False
        logger.debug("getting action size...")
        visinput = [flags.vision, flags.vis_h, flags.vis_w]
        action_size = Environment.get_action_size(flags.env_type,
                                                  flags.env_name)
        # Setup Global Network
        logger.debug("loading global model...")
        self.global_network = UnrealModel(action_size,
                                          visinput,
                                          -1,
                                          flags.entropy_beta,
                                          device,
                                          flags.use_pixel_change,
                                          flags.use_value_replay,
                                          flags.use_reward_prediction,
                                          flags.use_temporal_coherence,
                                          flags.pixel_change_lambda,
                                          flags.temporal_coherence_lambda)
        logger.debug("done loading global model")
        learning_rate_input = tf.placeholder("float")
        
        # Setup gradient calculator
        """
        grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = flags.rmsp_alpha,
                                  momentum = 0.0,
                                  epsilon = flags.rmsp_epsilon,
                                  clip_norm = flags.grad_norm_clip,
                                  device = device)
        """
        grad_applier = AdamApplier(learning_rate = learning_rate_input,
                                   clip_norm=flags.grad_norm_clip,
                                   device=device)
        # Start environment
        self.environment = Environment.create_environment(flags.env_type,
                                                      flags.env_name,
                                                      visinput)
        logger.debug("done loading environment")
        
        # Setup runner
        self.runner = RunnerThread(self.environment,
                                   self.global_network,
                                   action_size,
                                   visinput,
                                   flags.entropy_beta,
                                   device,
                                   flags.local_t_max, 
                                   visualise)
        logger.debug("done setting up RunnerTread")
        
        # Setup experience
        self.experience = Experience(flags.experience_history_size)
        
        #@TODO check device usage: should we build a cluster?
        # Setup Base Network
        self.base_trainer = BaseTrainer(self.runner,
                                        self.global_network,
                                        initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier,
                                        visinput,
                                        flags.env_type,
                                        flags.env_name,
                                        flags.entropy_beta,
                                        flags.gamma,
                                        self.experience,
                                        flags.max_time_step,
                                        device)
        
        # Setup Aux Networks
        self.aux_trainers = []
        for k in range(flags.parallel_size):
            self.aux_trainers.append(AuxTrainer(self.global_network,
                                                k+2, #-1 is global, 0 is runnerthread, 1 is base
                                                flags.use_pixel_change, 
                                                flags.use_value_replay,
                                                flags.use_reward_prediction,
                                                flags.use_temporal_coherence,
                                                flags.pixel_change_lambda,
                                                flags.temporal_coherence_lambda,
                                                initial_learning_rate,
                                                learning_rate_input,
                                                grad_applier,
                                                visinput,
                                                self.aux_t,
                                                flags.env_type,
                                                flags.env_name,
                                                flags.local_t_max,
                                                flags.gamma,
                                                flags.gamma_pc,
                                                self.experience,
                                                flags.max_time_step,
                                                device))
        
        # Start tensorflow session
        config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.init_tensorboard()

        # init or load checkpoint with saver
        self.saver = tf.train.Saver(self.global_network.get_vars())
        
        checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
        if CONTINUE_TRAINING and checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            checkpointpath = checkpoint.model_checkpoint_path.replace("/", "\\")
            logger.info("checkpoint loaded: {}".format(checkpointpath))
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            logger.info(">>> global step set: {}".format(self.global_t))
            logger.info(">>> aux step: {}".format(self.aux_t))
            # set wall time
            wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
            with open(wall_t_fname, 'r') as f:
                self.wall_t = float(f.read())
                self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
                logger.debug("next save steps:{}".format(self.next_save_steps))
        else:
            logger.info("Could not find old checkpoint")
            # set wall time
            self.wall_t = 0.0
            self.next_save_steps = flags.save_interval_step
        
       

        signal.signal(signal.SIGINT, self.signal_handler)

        # set start time
        self.start_time = time.time() - self.wall_t
        # Start runner
        self.runner.start_runner(self.sess)
        # Start base_network thread
        self.base_train_thread = threading.Thread(target=self.base_train_function, args=())
        self.base_train_thread.start()
        
        # Start aux_network threads
        self.aux_train_threads = []
        for k in range(flags.parallel_size):
            self.aux_train_threads.append(threading.Thread(target=self.aux_train_function, args=(k,)))
            self.aux_train_threads[k].start()
            
        logger.debug(threading.enumerate())

        logger.info('Press Ctrl+C to stop')
        signal.pause()
    
    
    def init_tensorboard(self):
        # tensorboard summary for base 
        self.score_input = tf.placeholder(tf.int32)
        self.epl_input = tf.placeholder(tf.int32)
        self.policy_loss = tf.placeholder(tf.float32)
        self.value_loss = tf.placeholder(tf.float32)
        self.base_entropy = tf.placeholder(tf.float32)
        self.base_gradient = tf.placeholder(tf.float32)
        self.laststate = tf.placeholder(tf.float32, [1, flags.vis_w, flags.vis_h, len(flags.vision)], name="laststate")
        score = tf.summary.scalar("env/score", self.score_input)
        epl = tf.summary.scalar("env/ep_length", self.epl_input)
        policy_loss = tf.summary.scalar("base/policy_loss", self.policy_loss)
        value_loss = tf.summary.scalar("base/value_loss", self.value_loss)
        entropy = tf.summary.scalar("base/entropy", self.base_entropy)
        gradient = tf.summary.scalar("base/gradient", self.base_gradient)
        laststate = tf.summary.image("base/laststate", self.laststate)

        self.summary_values = [self.score_input, self.epl_input, self.policy_loss, self.value_loss, self.base_entropy, self.base_gradient, self.laststate]
        self.summary_op = tf.summary.merge_all() # we want to merge model histograms as well here
        
        # tensorboard summary for aux
        self.summary_aux = []
        aux_losses = []
        self.aux_basep_loss = tf.placeholder(tf.float32)
        self.aux_basev_loss = tf.placeholder(tf.float32)
        self.summary_aux.append(self.aux_basep_loss)
        self.summary_aux.append(self.aux_basev_loss)
        aux_losses.append(tf.summary.scalar("aux/basep_loss", self.aux_basep_loss))
        aux_losses.append(tf.summary.scalar("aux/basev_loss", self.aux_basev_loss))
        
        if flags.use_pixel_change:
            self.pc_loss = tf.placeholder(tf.float32)
            self.summary_aux.append(self.pc_loss)
            aux_losses.append(tf.summary.scalar("aux/pc_loss", self.pc_loss))
        if flags.use_value_replay:
            self.vr_loss = tf.placeholder(tf.float32)
            self.summary_aux.append(self.vr_loss)
            aux_losses.append(tf.summary.scalar("aux/vr_loss", self.vr_loss))
        if flags.use_reward_prediction:
            self.rp_loss = tf.placeholder(tf.float32)
            self.summary_aux.append(self.rp_loss)
            aux_losses.append(tf.summary.scalar("aux/rp_loss", self.rp_loss))
        if flags.use_temporal_coherence:
            self.tc_loss = tf.placeholder(tf.float32)
            self.summary_aux.append(self.tc_loss)
            aux_losses.append(tf.summary.scalar("aux/tc_loss", self.tc_loss))
        
        self.summary_op_aux = tf.summary.merge(aux_losses)
        
        #self.summary_op = tf.summary.merge_all()
        tensorboard_path = flags.temp_dir+TRAINING_NAME+"/"
        logger.info("tensorboard path:"+tensorboard_path)
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.summary_writer = tf.summary.FileWriter(tensorboard_path)
        self.summary_writer.add_graph(self.sess.graph)

    def save(self):
        """ Save checkpoint. 
        Called from base_trainer.
        """
        self.stop_requested = True
        
      
        # Save
        if not os.path.exists(flags.checkpoint_dir):
            os.mkdir(flags.checkpoint_dir)
      
        # Write wall time
        wall_t = time.time() - self.start_time
        wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
        with open(wall_t_fname, 'w') as f:
            f.write(str(wall_t))
        logger.info('Start saving.')
        self.saver.save(self.sess,
                    flags.checkpoint_dir + '/' + 'checkpoint',
                    global_step = self.global_t)
        logger.info('End saving.')
    
        self.stop_requested = False
        self.next_save_steps += flags.save_interval_step

    def signal_handler(self, signal, frame):
        logger.warn('Ctrl+C detected, shutting down...')
        logger.info('run name: {} -- terminated'.format(TRAINING_NAME))
        self.terminate_requested = True


def main(argv):
    app = Application()
    app.run()

if __name__ == '__main__':
    logger = logger_init(flags.log_dir+TRAINING_NAME+'/', TRAINING_NAME, loglevel=LOG_LEVEL)
    
    tf.app.run()
    
