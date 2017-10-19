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
from train.rmsprop_applier import RMSPropApplier
from train.base_trainer import BaseTrainer
from queuer import RunnerThread
from options import get_options

logger = logging.getLogger('StRADRL.newmain')
LOG_DIR = u'./temp/run_id/'
LOG_LEVEL = 'debug'

LOCAL_ENV_STEPS = 20
USE_GPU = True
visualise = False

# get command line args
flags = get_options("training")



class Application(object):
    def __init__(self):
        pass
        
    def base_train_function(self):
        """ Train routine for base_trainer. """
        
        trainer = self.base_trainer
        
        # set start_time
        trainer.set_start_time(self.start_time)
      
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
                self.save()
            
            diff_global_t = trainer.process(self.sess,
                                          self.global_t,
                                          self.summary_writer,
                                          self.summary_op,
                                          self.score_input)
            self.global_t += diff_global_t
            
            
    def run(self):
        device = "/cpu:0"
        if USE_GPU:
            device = "/gpu:0"
        logger.debug("start App")
        initial_learning_rate = 0.001 #@TODO implement unreal method?
        
        self.global_t = 0
        self.stop_requested = False
        self.terminate_requested = False
        logger.debug("getting action size...") 
        action_size = Environment.get_action_size(flags.env_type,
                                                  flags.env_name)
        # Setup Global Network
        logger.debug("loading global model...")
        self.global_network = UnrealModel(action_size,
                                          -1,
                                          flags.use_pixel_change,
                                          flags.use_value_replay,
                                          flags.use_reward_prediction,
                                          flags.pixel_change_lambda,
                                          flags.entropy_beta,
                                          device)
        logger.debug("done loading global model")
        learning_rate_input = tf.placeholder("float")
        
        # Setup gradient calculator
        grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = flags.rmsp_alpha,
                                  momentum = 0.0,
                                  epsilon = flags.rmsp_epsilon,
                                  clip_norm = flags.grad_norm_clip,
                                  device = device)

        # Start environment
        self.environment = Environment.create_environment(flags.env_type,
                                                      flags.env_name)
        logger.debug("done loading environment")
        
        # Setup runner
        self.runner = RunnerThread(self.environment, self.global_network, LOCAL_ENV_STEPS, visualise)
        logger.debug("done setting up RunnerTread")
        
        #@TODO check device usage: should we build a cluster?
        # Setup Base Network
        self.base_trainer = BaseTrainer(self.runner,
                                        self.global_network,
                                        initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier,
                                        flags.env_type,
                                        flags.env_name,
                                        flags.entropy_beta,
                                        flags.local_t_max,
                                        flags.gamma,
                                        flags.experience_history_size,
                                        flags.max_time_step,
                                        device)
        
        # Start tensorflow session
        config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer())
        
        # summary for tensorboard
        self.score_input = tf.placeholder(tf.int32)
        tf.summary.scalar("score", self.score_input)

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(flags.log_file,
                                                    self.sess.graph)

        # init or load checkpoint with saver
        self.saver = tf.train.Saver(self.global_network.get_vars())

        checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            logger.info("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            logger.info(">>> global step set: ", self.global_t)
            # set wall time
            wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
            with open(wall_t_fname, 'r') as f:
                self.wall_t = float(f.read())
                self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
        
        else:
            logger.info("Could not find old checkpoint")
            # set wall time
            self.wall_t = 0.0
            self.next_save_steps = flags.save_interval_step
        
       

        signal.signal(signal.SIGINT, self.signal_handler)

        # set start time
        self.start_time = time.time() - self.wall_t
        # Start runner
        self.runner.start_runner(self.sess, self.summary_writer)
        # Start base_network
        self.base_train_thread = threading.Thread(target=self.base_train_function, args=())
        self.base_train_thread.start()
        logger.debug(threading.enumerate())

        logger.info('Press Ctrl+C to stop')
        signal.pause()


    def save(self):
        """ Save checkpoint. 
        Called from therad-0.
        """
        self.stop_requested = True
      
        # Wait for all other threads to stop
        for (i, t) in enumerate(self.train_threads):
            if i != 0:
                t.join()
      
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
        self.terminate_reqested = True


def main(argv):
    app = Application()
    app.run()

if __name__ == '__main__':
    run_id = generate_id()
    logger = logger_init(LOG_DIR, run_id, loglevel=LOG_LEVEL)
    tf.app.run()
    
