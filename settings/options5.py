# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_options(option_type):
  """
  option_type: string
    'training' or 'diplay' or 'visualize'
  """    
  # name
  tf.app.flags.DEFINE_string("training_name","caus_v1","name of next training in log")

    
  # Common
  tf.app.flags.DEFINE_string("env_type", "gym", "environment type (lab or gym or maze)")
  tf.app.flags.DEFINE_string("env_name", "CartPole-v1",  "environment name (for lab)")
  tf.app.flags.DEFINE_integer("env_max_steps", 400000, "max number of steps in environment")
  
  tf.app.flags.DEFINE_boolean("use_base", False, "whether to use base A3C for aux network")
  tf.app.flags.DEFINE_boolean("use_pixel_change", False, "whether to use pixel change")
  tf.app.flags.DEFINE_boolean("use_value_replay", False, "whether to use value function replay")
  tf.app.flags.DEFINE_boolean("use_reward_prediction", False, "whether to use reward prediction")
  tf.app.flags.DEFINE_boolean("use_temporal_coherence", False, "whether to use temporal coherence")
  tf.app.flags.DEFINE_boolean("use_proportionality", False, "whether to use proportionality")
  tf.app.flags.DEFINE_boolean("use_causality", True, "whether to use causality")
  tf.app.flags.DEFINE_boolean("use_repeatability", False, "whether to use repeatability")
  tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/StRADRL/checkpoints", "checkpoint directory")

  # For training
  if option_type == 'training':
    tf.app.flags.DEFINE_string("temp_dir", "/tmp/StRADRL/tensorboard/", "base directory for tensorboard")
    tf.app.flags.DEFINE_string("log_dir", "/tmp/StRADRL/log/", "base directory for logs")
    tf.app.flags.DEFINE_integer("max_time_step", 10**6, "max time steps")
    tf.app.flags.DEFINE_integer("save_interval_step", 10**4, "saving interval steps")
    tf.app.flags.DEFINE_boolean("grad_norm_clip", 40.0, "gradient norm clipping")

    #base
    tf.app.flags.DEFINE_float("initial_learning_rate", 1e-3, "learning rate")
    tf.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards")
    tf.app.flags.DEFINE_float("entropy_beta", 0.01, "entropy regurarlization constant")
    tf.app.flags.DEFINE_float("value_lambda", 0.5, "value ratio for base loss")
    tf.app.flags.DEFINE_float("base_lambda", 0.97, "generalized adv. est. lamba for short-long sight")
    
    
    # auxiliary
    tf.app.flags.DEFINE_integer("parallel_size", 1, "parallel thread size")
    tf.app.flags.DEFINE_float("aux_initial_learning_rate", 1e-3, "learning rate")
    tf.app.flags.DEFINE_float("aux_lambda", 0.0, "generalized adv. est. lamba for short-long sight (aux)")
    tf.app.flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pixel control")
    tf.app.flags.DEFINE_float("pixel_change_lambda", 0.0001, "pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
    tf.app.flags.DEFINE_float("temporal_coherence_lambda", 1., "temporal coherence lambda")
    tf.app.flags.DEFINE_float("proportionality_lambda", 100., "proportionality lambda")
    tf.app.flags.DEFINE_float("causality_lambda", 1., "causality lambda")
    tf.app.flags.DEFINE_float("repeatability_lambda", 100., "repeatability lambda")
    
    tf.app.flags.DEFINE_integer("experience_history_size", 100000, "experience replay buffer size")
    
    # queuer
    tf.app.flags.DEFINE_integer("local_t_max", 20, "repeat step size")
    tf.app.flags.DEFINE_integer("queue_length", 5, "max number of batches (of length local_t_max) in queue")
    tf.app.flags.DEFINE_integer("env_runner_sync", 1, "number of env episodes before sync to global")
    tf.app.flags.DEFINE_float("action_freq", 0,  "number of actions per second in env")
    

  # For display
  if option_type == 'display':
    tf.app.flags.DEFINE_string("frame_save_dir", "/tmp/StRADRL_frames", "frame save directory")
    tf.app.flags.DEFINE_boolean("recording", False, "whether to record movie")
    tf.app.flags.DEFINE_boolean("frame_saving", False, "whether to save frames")

  return tf.app.flags.FLAGS
