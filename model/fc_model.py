# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger('StRADRL.model')

SEED = 1337

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d, seed=SEED)
    return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d, seed=SEED)
    return _initializer


class UnrealModel(object):
    """
    UNREAL algorithm network model.
    """
    def __init__(self,
                action_size,
                obs_size,
                thread_index, # -1 for global
                entropy_beta,
                device,
                use_pixel_change=False,
                use_value_replay=False,
                use_reward_prediction=False,
                use_temporal_coherence=False,
                use_proportionality=False,
                value_lambda=0.5,
                pixel_change_lambda=0.,
                temporal_coherence_lambda=0.,
                for_display=False,
                use_base=True):
        self._device = device
        self._action_size = action_size
        self._obs_size = obs_size
        self.input_shape = [None, self._obs_size]     
        self._thread_index = thread_index
        self._use_pixel_change = use_pixel_change
        self._use_value_replay = use_value_replay
        self._use_reward_prediction = use_reward_prediction
        self._use_temporal_coherence = use_temporal_coherence
        self._use_proportionality = use_proportionality
        self._use_base = use_base
        self._pixel_change_lambda = pixel_change_lambda
        self._temporal_coherence_lambda = temporal_coherence_lambda
        self._value_lambda = value_lambda
        self._entropy_beta = entropy_beta
        self.reuse_conv = False
        self.reuse_lstm = False
        self.reuse_value = False
        self.reuse_policy = False
        self._create_network(for_display)


        
    def get_initial_features(self):
        return self.state_init
    
    def _create_network(self, for_display):
        scope_name = "net_{}".format(self._thread_index)
        logger.debug("creating network -- scope_name:{} -- device:{}".format(scope_name,self._device))
        logger.debug("base:{} -- pc:{} -- vr:{} -- rp:{} -- tc:{}".format(self._use_base,self._use_pixel_change,\
                            self._use_value_replay,self._use_reward_prediction,self._use_temporal_coherence))
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            ## lstm
            #self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
              
            # [base A3C network]
            #if self._use_base:
            self._create_base_network()

            # [Pixel change network]
            if self._use_pixel_change:
                self._create_pc_network()
                if for_display:
                    self._create_pc_network_for_display()

            # [Value replay network]
            if self._use_value_replay:
                self._create_vr_network()

            # [Reward prediction network]
            if self._use_reward_prediction:
                self._create_rp_network()
                
            # [Temporal Coherence network]
            if self._use_temporal_coherence:
                self._create_tc_network()
            
            # [Proportionality network]    
            if self._use_proportionality:
                self._create_prop_network()
            
            self.reset_state()

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


    def _create_base_network(self):
        # State (Base image input)
        self.base_input = tf.placeholder("float", self.input_shape, name="base_input")
        
        #self.base_flat = tf.reshape(self.base_input, [-1, 7*7*self._ch_num])
        # Last action and reward
        self.base_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])
        
        # Fully connected layers (we "borrow" the reuse_lstm boolean)
        self.base_fc_outputs = self._fc_layers(self.base_input, reuse=self.reuse_lstm)
        
        ## Conv layers
        #base_conv_output = self._base_conv_layers(self.base_input, reuse=self.reuse_conv)
        
        ## LSTM layer
        #self.base_initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
        #self.base_initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256], name='bils1')

        #self.base_initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.base_initial_lstm_state0,
        #                                                             self.base_initial_lstm_state1)
        
        #c_init = np.zeros((self.base_initial_lstm_state[0].shape), np.float32)
        #h_init = np.zeros((self.base_initial_lstm_state[1].shape), np.float32)
        self.state_init = []
        
        
        #self.base_lstm_outputs, self.base_lstm_state = \
        #    self._base_lstm_layer(base_conv_output,
        #                          self.base_last_action_reward_input,
        #                          self.base_initial_lstm_state,
        #                          reuse=self.reuse_lstm)

        self.base_pi, self.base_pi_log = self._base_policy_layer(self.base_fc_outputs, reuse=self.reuse_policy) # policy output
        self.base_v  = self._base_value_layer(self.base_fc_outputs, reuse=self.reuse_value)  # value output

    def _fc_layers(self, state_input, reuse=False):
        with tf.variable_scope("base_fc", reuse=reuse) as scope:
            # Weight for policy output layer
            #logger.debug(state_input.shape[1])
            W_fc_1, b_fc_1 = self._fc_variable([self._obs_size, 64], "base_fc_1")
            W_fc_2, b_fc_2 = self._fc_variable([64, 64], "base_fc_2")
            #W_fc_3, b_fc_3 = self._fc_variable([256, 256], "base_fc_3")
            
            out_fc_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(state_input, W_fc_1) + b_fc_1),0.5)            
            out_fc_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(out_fc_1, W_fc_2) + b_fc_2),0.5)
            #out_fc_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(out_fc_2, W_fc_3) + b_fc_3),0.5)
            
            self.reuse_lstm = True # "borrowed lstm reuse check"
        
            return out_fc_2
        

    def _base_policy_layer(self, lstm_outputs, reuse=False):
        with tf.variable_scope("base_policy", reuse=reuse) as scope:
            # Weight for policy output layer
            W_fc_p, b_fc_p = self._fc_variable([64, self._action_size], "base_fc_p")
            
            tf.summary.histogram("policyW", W_fc_p)
            tf.summary.histogram("policyb", b_fc_p)
            
            # Policy (output)
            #logger.warn(" !! doing some tricks with the policy layer, have a look !!")
            base_pi_linear = tf.matmul(lstm_outputs, W_fc_p) + b_fc_p
            base_pi = tf.nn.softmax(base_pi_linear)
            #base_pi = base_pi_linear
            base_pi_log = tf.nn.log_softmax(base_pi_linear)
            
            # set reuse to True to make aux tasks reuse the variables
            self.reuse_policy = True
            
            return base_pi, base_pi_log


    def _base_value_layer(self, lstm_outputs, reuse=False):
        with tf.variable_scope("base_value", reuse=reuse) as scope:
            # Weight for value output layer
            W_fc_v, b_fc_v = self._fc_variable([64, 1], "base_fc_v")
            
            tf.summary.histogram("valueW", W_fc_v)
            tf.summary.histogram("valueb", b_fc_v)
            
            # Value (output)
            #logger.warn("!! ELU set for value !!")
            #v_ = tf.nn.elu(tf.matmul(lstm_outputs, W_fc_v) + b_fc_v)
            v_ = tf.matmul(lstm_outputs, W_fc_v) + b_fc_v
            base_v = tf.reshape( v_, [-1] )

            # set reuse to True to make aux tasks reuse the variables
            self.reuse_value = True            
            
            return base_v



    def _create_vr_network(self):
        # State (Image input)
        self.vr_input = tf.placeholder("float", self.input_shape, name="vr_input")

        # Last action and reward
        self.vr_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])
        
        vr_lstm_outputs = self._fc_layers(self.vr_input, reuse=self.reuse_lstm)
        
        # value output
        self.vr_v  = self._base_value_layer(vr_lstm_outputs, reuse=self.reuse_value)

    
    def _create_rp_network(self):
        self.rp_input = tf.placeholder("float", [3, 84, 84, self._ch_num])

        # RP conv layers
        rp_conv_output = self._base_conv_layers(self.rp_input, reuse=self.reuse_conv)
        rp_conv_output_reshaped = tf.reshape(rp_conv_output, [1,9*9*32*3])
        
        with tf.variable_scope("rp_fc") as scope:
            # Weights
            W_fc1, b_fc1 = self._fc_variable([9*9*32*3, 3], "rp_fc1")

        # Reward prediction class output. (zero, positive, negative)
        self.rp_c = tf.nn.softmax(tf.matmul(rp_conv_output_reshaped, W_fc1) + b_fc1)
        # (1,3)
         
    # temporal coherence
    def _create_tc_network(self):
        # State (Image input)
        self.tc_input1 = tf.placeholder("float", self.input_shape, name="tc_input1")
        self.tc_input2 = tf.placeholder("float", self.input_shape, name="tc_input2")

        # tc conv layers
        tc_output1 = self._fc_layers(self.tc_input1, reuse=self.reuse_lstm)
        tc_output2 = self._fc_layers(self.tc_input2, reuse=self.reuse_lstm)
        
        # loss is norm of fc output difference
        self.tc_q = tf.reduce_mean(tf.subtract(tc_output2,tc_output1))
        
    # proportionality
    def _create_prop_network(self):
        # State (Image input)
        self.prop_input1 = tf.placeholder("float", self.input_shape, name="prop_input1")
        self.prop_input2 = tf.placeholder("float", self.input_shape, name="prop_input2")

        # tc conv layers
        prop_output1 = self._fc_layers(self.prop_input1, reuse=self.reuse_lstm)
        prop_output2 = self._fc_layers(self.prop_input2, reuse=self.reuse_lstm)
        
        # take diff of conv outputs
        self.prop_q = tf.norm(prop_output2-prop_output1)

    def _base_loss(self):
        # [base A3C]
        # Taken action (input for policy)
        self.base_a = tf.placeholder("float", [None, self._action_size])
        
        # Advantage (R-V) (input for policy)
        self.base_adv = tf.placeholder("float", [None])
               
        
        # Policy loss (output)
        self.policy_loss = -tf.reduce_sum( tf.reduce_sum( self.base_pi_log * self.base_a, [1] ) *
                                      self.base_adv )
        
        # R (input for value target)
        self.base_r = tf.placeholder("float", [None])
        
        # Value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        self.value_loss = self._value_lambda * tf.reduce_sum(tf.square(self.base_v - self.base_r))
        
        # Policy entropy
        self.entropy = -tf.reduce_sum(self.base_pi * self.base_pi_log) * self._entropy_beta
        
        base_loss = self.policy_loss + self.value_loss - self.entropy
        return base_loss

  
    def _pc_loss(self):
        # [pixel change]
        self.pc_a = tf.placeholder("float", [None, self._action_size])
        pc_a_reshaped = tf.reshape(self.pc_a, [-1, 1, 1, self._action_size])

        # Extract Q for taken action
        pc_qa_ = tf.multiply(self.pc_q, pc_a_reshaped)
        pc_qa = tf.reduce_sum(pc_qa_, reduction_indices=3, keep_dims=False)
        # (-1, 20, 20)
          
        # TD target for Q
        self.pc_r = tf.placeholder("float", [None, 20, 20])

        pc_loss = self._pixel_change_lambda * tf.nn.l2_loss(self.pc_r - pc_qa)
        return pc_loss

  
    def _vr_loss(self):
        # R (input for value)
        self.vr_r = tf.placeholder("float", [None])
        
        # Value loss (output)
        vr_loss = tf.nn.l2_loss(self.vr_r - self.vr_v)
        return vr_loss

    def _rp_loss(self):
        # reward prediction target. one hot vector
        self.rp_c_target = tf.placeholder("float", [1,3])
        
        # Reward prediction loss (output)
        rp_c = tf.clip_by_value(self.rp_c, 1e-20, 1.0)
        rp_loss = -tf.reduce_sum(self.rp_c_target * tf.log(rp_c))
        return rp_loss
    
    def _tc_loss(self):
        # temporal coherence loss
        tc_loss = self._temporal_coherence_lambda * self.tc_q
        return tc_loss
    
    def prepare_loss(self):
        with tf.device(self._device):
            loss = tf.Variable(tf.zeros([], dtype=np.float32), name="loss")
            loss_nullifier = tf.zeros([], dtype=np.float32)
            
            self.base_loss = self._base_loss()
            if not self._use_base:
                self.base_loss *= loss_nullifier
            loss = loss +  self.base_loss
      
            if self._use_pixel_change:
                self.pc_loss = self._pc_loss()
                loss = loss + self.pc_loss

            if self._use_value_replay:
                self.vr_loss = self._vr_loss()
                loss = loss + self.vr_loss

            if self._use_reward_prediction:
                self.rp_loss = self._rp_loss()
                loss = loss + self.rp_loss
                
            if self._use_temporal_coherence:
                self.tc_loss = self._tc_loss()
                loss = loss + self.tc_loss
            
            self.total_loss = loss


    def reset_state(self):
        self.base_lstm_state_out = []
                                                             
    def set_state(self, features):
        self.base_lstm_state_out = features#tf.contrib.rnn.LSTMStateTuple(features[0],features[1])

    def run_base_policy_and_value(self, sess, s_t, last_action_reward):
        # This run_base_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out = sess.run( [self.base_pi, self.base_v],
                                    feed_dict = {self.base_input : [s_t],
                                                 self.base_last_action_reward_input : [last_action_reward]} )
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0], [])
  
    def run_base_value(self, sess, s_t, last_action_reward):
        # This run_base_value() is used for calculating V for bootstrapping at the 
        # end of LOCAL_T_MAX time step sequence.
        # When the next sequence starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        v_out = sess.run( [self.base_v],
                             feed_dict = {self.base_input : [s_t],
                                          self.base_last_action_reward_input : [last_action_reward]} )
        return v_out[0]
  
    def run_vr_value(self, sess, s_t, last_action_reward):
        vr_v_out = sess.run( self.vr_v,
                         feed_dict = {self.vr_input : [s_t],
                                      self.vr_last_action_reward_input : [last_action_reward]} )
        return vr_v_out[0]

  
    def get_vars(self):
        return self.variables
  

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(None, "UnrealModel", []) as scopename:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)
        #"""

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

