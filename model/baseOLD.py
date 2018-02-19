# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger('StRADRL.base')

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer
    
def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer


class BaseModel(object):
    """
    Base A3C model (no RNN)
    """
    def __init__(self,
                 visinput,
                 action_size,
                 thread_index, 
                 entropy_beta,
                 device):

        self._device = device
        self.ch_num = len(visinput[0]) # ch_num is 1 if D, 3 if RGB, 4 if RGBD
        self.vis_h = visinput[1]
        self.vis_w = visinput[2]
        self._action_size = action_size
        self._thread_index = thread_index
        self._entropy_beta = entropy_beta
        self.reuse_conv = False
        self.reuse_fc = False
        self.reuse_value = False
        self.reuse_policy = False
        self._create_network()
        
    def _create_network(self):
        scope_name = "net_base_{}".format(self._thread_index)
        logger.debug("creating base network -- device:{}".format(self._device))
        
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # State (Base image input)
            self.base_input = tf.placeholder("float", [None, self.vis_w, self.vis_h, self.ch_num], name="base_input")
            # Conv layers
            base_conv_output = self._base_conv_layers(self.base_input)
            
            # FC layer
            base_fc_output = self._base_fc_layer(base_conv_output)
            
            # Policy and Value layers
            self.base_pi, self.base_pi_log = self._base_policy_layer(base_fc_output) # policy output
            self.base_v  = self._base_value_layer(base_fc_output)  # value output
            
            self.reset_state()
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
            
            
    def _base_conv_layers(self, state_input):
        with tf.variable_scope("base_conv", reuse=self.reuse_conv) as scope:
            # Weights
            W_conv1, b_conv1 = self._conv_variable([8, 8, self.ch_num, 16],  "base_conv1")
            W_conv2, b_conv2 = self._conv_variable([4, 4, 16, 32], "base_conv2")
            
            # Nodes
            h_conv1 = tf.nn.elu(self._conv2d(state_input, W_conv1, 4) + b_conv1) # stride=4
            h_conv2 = tf.nn.elu(self._conv2d(h_conv1,     W_conv2, 2) + b_conv2) # stride=2
            
            # tensorboard summaries
            tf.summary.histogram("weights1", W_conv1)
            tf.summary.histogram("weights2", W_conv2)
            tf.summary.histogram("biases1", b_conv1)
            tf.summary.histogram("biases2", b_conv2)
            
            # set reuse to True to make other functions reuse the variables
            self.reuse_conv = True
            
            return h_conv2

    def _base_fc_layer(self, conv_output):
        with tf.variable_scope("base_fc", reuse=self.reuse_fc) as scope:
            # Weights and biases for fc layer
            W_fc1, b_fc1 = self._fc_variable([2592, 256], "base_fc1")
            
            # Flatten (bs*9*9*32 = bs*2592)
            conv_output_flat = tf.reshape(conv_output, [-1, 2592])
            
            # Make fc layer
            fc_output = tf.nn.elu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
            
            # set reuse to True to make aux tasks reuse the variables
            self.reuse_fc = False
            
            return fc_output            
            
    def _base_policy_layer(self, lstm_outputs):
        with tf.variable_scope("base_policy", reuse=self.reuse_policy) as scope:
            # Weight for policy output layer
            W_fc_p, b_fc_p = self._fc_variable([256, self._action_size], "base_fc_p")
            # Policy (output)
            logits = tf.matmul(lstm_outputs, W_fc_p) + b_fc_p
            base_pi = tf.nn.softmax(logits)
            base_pi_log = tf.nn.log_softmax(logits)
            
            # set reuse to True to make aux tasks reuse the variables
            self.reuse_policy = True
            
            return base_pi,base_pi_log

    def _base_value_layer(self, lstm_outputs):
        with tf.variable_scope("base_value", reuse=self.reuse_value) as scope:
            # Weight for value output layer
            W_fc_v, b_fc_v = self._fc_variable([256, 1], "base_fc_v")
            
            # Value (output)
            v_ = tf.matmul(lstm_outputs, W_fc_v) + b_fc_v
            base_v = tf.reshape( v_, [-1] )

            # set reuse to True to make aux tasks reuse the variables
            self.reuse_value = True            
            
            return base_v
            
    def _base_loss(self):
        # [base A3C]
        # Taken action (input for policy)
        self.base_a = tf.placeholder("float", [None, self._action_size])
        # Advantage (R-V) (input for policy)
        self.base_adv = tf.placeholder("float", [None])    
        # R (input for value target)
        self.base_r = tf.placeholder("float", [None])
        
        # Policy loss (output)
        self.policy_loss = -tf.reduce_sum(tf.reduce_sum(self.base_pi_log * self.base_a, [1]) * self.base_adv)

        # Value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.base_v - self.base_r))
        
        # Policy entropy
        self.entropy = -tf.reduce_sum(self.base_pi * self.base_pi_log)
        
        base_loss = self.policy_loss + 0.5 * self.value_loss - self.entropy * self._entropy_beta
        return base_loss

    def prepare_loss(self):
        with tf.device(self._device):
            self.total_loss = self._base_loss()
            
    def run_base_policy_and_value(self, sess, s_t):
        # This run_base_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out = sess.run( [self.base_pi, self.base_v],
                                    feed_dict = {self.base_input : [s_t]} )
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0])
        
    def run_base_value(self, sess, s_t):
        v_out = sess.run([self.base_v], feed_dict = {self.base_input: [s_t]})
        return v_out[0]
    
    def get_vars(self):
        return self.variables
        
    def sync_from(self, src_network, name=None):
        #logger.debug("sync {} from {}".format(name, src_network))
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "BaseModel",[]) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)
        
    def _fc_variable(self, weight_shape, name):
        name_w = "W_{0}".format(name)
        name_b = "b_{0}".format(name)
        
        input_channels  = weight_shape[0]
        output_channels = weight_shape[1]
        bias_shape = [output_channels]

        weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
        bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
        return weight, bias

  
    def _conv_variable(self, weight_shape, name, deconv=False):
        name_w = "W_{0}".format(name)
        name_b = "b_{0}".format(name)
        
        w = weight_shape[0]
        h = weight_shape[1]
        if deconv:
            input_channels  = weight_shape[3]
            output_channels = weight_shape[2]
        else:
            input_channels  = weight_shape[2]
            output_channels = weight_shape[3]
        bias_shape = [output_channels]

        weight = tf.get_variable(name_w, weight_shape,
                                 initializer=conv_initializer(w, h, input_channels))
        bias   = tf.get_variable(name_b, bias_shape,
                                 initializer=conv_initializer(w, h, input_channels))
        return weight, bias

  
    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")
        
    def reset_state(self):
        logger.debug("dummy function, resetting state")
        
