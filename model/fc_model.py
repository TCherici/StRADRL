# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger('StRADRL.model')

SEED = 4444#3000

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        #d = np.sqrt(1/input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)#, seed=SEED)
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
                is_discrete,
                obs_size,
                thread_index, # -1 for global
                entropy_beta,
                device,
                use_pixel_change=False,
                use_value_replay=False,
                use_reward_prediction=False,
                use_temporal_coherence=False,
                use_proportionality=False,
                use_causality=False,
                use_repeatability=False,
                value_lambda=0.5,
                pixel_change_lambda=0.,
                temporal_coherence_lambda=0.,
                proportionality_lambda=0.,
                causality_lambda=0.,
                repeatability_lambda=0.,
                for_display=False,
                use_base=True):
        self._device = device
        self._action_size = action_size
        self._is_discrete = is_discrete
        self._obs_size = obs_size
        self._input_shape = [None, self._obs_size]     
        self._thread_index = thread_index
        self._use_pixel_change = use_pixel_change
        self._use_value_replay = use_value_replay
        self._use_reward_prediction = use_reward_prediction
        self._use_temporal_coherence = use_temporal_coherence
        self._use_proportionality = use_proportionality
        self._use_causality = use_causality
        self._use_repeatability = use_repeatability
        self._use_base = use_base
        self._pixel_change_lambda = pixel_change_lambda
        self._temporal_coherence_lambda = temporal_coherence_lambda
        self._proportionality_lambda = proportionality_lambda
        self._causality_lambda = causality_lambda
        self._repeatability_lambda = repeatability_lambda
        self._value_lambda = value_lambda
        self._entropy_beta = entropy_beta
        self.reuse_conv = False
        self.reuse_fc = False
        self.reuse_value = False
        self.reuse_policy = False
        self._create_network(for_display)

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
                
            # [Causality network]    
            if self._use_causality:
                self._create_caus_network()
                
            if self._use_repeatability:
                self._create_rep_network()
            

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


    def _create_base_network(self):
        # State (Base image input)
        self.base_input = tf.placeholder("float", self._input_shape, name="base_input")
        
        # Fully connected layers
        self.base_fc_outputs = self._fc_layers(self.base_input, reuse=self.reuse_fc)

        if self._is_discrete:
            self.base_pi, self.base_pi_log, self.base_pi_linear = self._base_policy_layer_discrete(self.base_fc_outputs, reuse=self.reuse_policy) # discrete policy output
        else:
            self.base_distr, self.base_pi = self._base_policy_layer(self.base_fc_outputs, reuse=self.reuse_policy) # policy output
        
        self.base_v  = self._base_value_layer(self.base_fc_outputs, reuse=self.reuse_value)  # value output

    def _fc_layers(self, state_input, reuse=False):
        with tf.variable_scope("base_fc", reuse=reuse) as scope:
            # Weight for policy output layer
            #logger.debug(state_input.shape[1])
            W_fc_1, b_fc_1 = self._fc_variable([self._obs_size, 64], "base_fc_1")
            W_fc_2, b_fc_2 = self._fc_variable([64, 64], "base_fc_2")
            #W_fc_3, b_fc_3 = self._fc_variable([256, 256], "base_fc_3")
            
            #out_fc_1 = tf.nn.relu(tf.matmul(state_input, W_fc_1) + b_fc_1)     
            #out_fc_2 = tf.nn.relu(tf.matmul(out_fc_1, W_fc_2) + b_fc_2)
            
            out_fc_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(state_input, W_fc_1) + b_fc_1),0.5)     
            out_fc_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(out_fc_1, W_fc_2) + b_fc_2),0.5)
            #out_fc_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(out_fc_2, W_fc_3) + b_fc_3),0.5)
            
            self.reuse_fc = True # "borrowed lstm reuse check"
        
            return out_fc_2
            
    def _base_policy_layer_discrete(self, fc_outputs, reuse=False):
        with tf.variable_scope("base_policy", reuse=reuse) as scope:
            # Weight for policy output layer
            W_fc_p, b_fc_p = self._fc_variable([64, self._action_size], "base_fc_p")
            
            tf.summary.histogram("policyW", W_fc_p)
            tf.summary.histogram("policyb", b_fc_p)
            
            # Policy (output)
            #logger.warn(" !! doing some tricks with the policy layer, have a look !!")
            base_pi_linear = tf.matmul(fc_outputs, W_fc_p) + b_fc_p
            base_pi = tf.nn.softmax(base_pi_linear)
            #base_pi = base_pi_linear
            base_pi_log = tf.nn.log_softmax(base_pi_linear)
            
            # set reuse to True to make aux tasks reuse the variables
            self.reuse_policy = True
            
        return base_pi, base_pi_log, base_pi_linear

    def _base_policy_layer(self, fc_outputs, reuse=False):
        with tf.variable_scope("base_policy", reuse=reuse) as scope:
            # Weight for policy output layer
            W_fc_p, b_fc_p = self._fc_variable([64, 2*self._action_size], "base_fc_p")
            tf.summary.histogram("policyW", W_fc_p)
            tf.summary.histogram("policyb", b_fc_p)

            # Policy (output)
            #logger.warn(" !! doing some tricks with the policy layer, have a look !!")
            base_pi_linear = tf.matmul(fc_outputs, W_fc_p) + b_fc_p
            
            base_pi_linear = tf.reshape(base_pi_linear,[-1,self._action_size,2])
            mu = tf.nn.tanh(base_pi_linear[...,0])
            sigma = base_pi_linear[...,1]
            sigmarelu = tf.nn.softmax(sigma)
            logger.debug(mu.shape)
            logger.debug(sigmarelu.shape)
            
            distr = tf.distributions.Normal(mu,sigmarelu,validate_args=False,allow_nan_stats=True)
            
            sample = distr.sample()
            
            #base_pi = tf.nn.softmax(base_pi_linear)
            #base_pi = base_pi_linear
            #base_pi_log = tf.nn.log_softmax(base_pi_linear)
            
            # set reuse to True to make aux tasks reuse the variables
            self.reuse_policy = True
            
            return distr, sample


    def _base_value_layer(self, fc_outputs, reuse=False):
        with tf.variable_scope("base_value", reuse=reuse) as scope:
            # Weight for value output layer
            W_fc_v, b_fc_v = self._fc_variable([64, 1], "base_fc_v")
            
            tf.summary.histogram("valueW", W_fc_v)
            tf.summary.histogram("valueb", b_fc_v)
            
            # Value (output)
            #logger.warn("!! ELU set for value !!")
            #v_ = tf.nn.elu(tf.matmul(fc_outputs, W_fc_v) + b_fc_v)
            v_ = tf.matmul(fc_outputs, W_fc_v) + b_fc_v
            base_v = tf.reshape( v_, [-1] )

            # set reuse to True to make aux tasks reuse the variables
            self.reuse_value = True            
            
            return base_v



    def _create_vr_network(self):
        # State (Image input)
        self.vr_input = tf.placeholder("float", self._input_shape, name="vr_input")

        
        vr_fc_outputs = self._fc_layers(self.vr_input, reuse=self.reuse_fc)
        
        # value output
        self.vr_v  = self._base_value_layer(vr_fc_outputs, reuse=self.reuse_value)

    
    def _create_rp_network(self):
        self.rp_input = tf.placeholder("float", self._input_shape)
        """
        # RP conv layers
        rp_conv_output = self._base_conv_layers(self.rp_input, reuse=self.reuse_conv)
        rp_conv_output_reshaped = tf.reshape(rp_conv_output, [1,9*9*32*3])
        """
        rp_fc_output = self._fc_layers(self.rp_input, reuse=self.reuse_fc)
        
        with tf.variable_scope("rp_fc") as scope:
            # Weights
            W_fc1, b_fc1 = self._fc_variable([64, 3], "rp_fc1")

        # Reward prediction class output. (zero, positive, negative)
        self.rp_c = tf.nn.softmax(tf.matmul(rp_fc_output, W_fc1) + b_fc1)
        # (1,3)
         
    # temporal coherence
    def _create_tc_network(self):
        # Observations (input)
        self.tc_input1 = tf.placeholder("float", self._input_shape, name="tc_input1")
        self.tc_input2 = tf.placeholder("float", self._input_shape, name="tc_input2")

        # fc output is our internal state s
        tc_output1 = self._fc_layers(self.tc_input1, reuse=self.reuse_fc)
        tc_output2 = self._fc_layers(self.tc_input2, reuse=self.reuse_fc)
        
        # loss is norm of fc output difference
        self.tc_q = tf.reduce_mean(tf.norm(tc_output2-tc_output1))
        
    # proportionality
    def _create_prop_network(self):
        # Observations (input)
        self.prop_input1_1 = tf.placeholder("float", self._input_shape, name="prop_input1_1")
        self.prop_input1_2 = tf.placeholder("float", self._input_shape, name="prop_input1_2")
        self.prop_input2_1 = tf.placeholder("float", self._input_shape, name="prop_input2_1")
        self.prop_input2_2 = tf.placeholder("float", self._input_shape, name="prop_input2_2")
        # Boolean vector check for if actions 1 and 2 are equal
        self.prop_actioncheck = tf.placeholder("float", [None,], name="prop_actioncheck")

        # get fc outputs (our internal state s)
        prop_output1_1 = self._fc_layers(self.prop_input1_1, reuse=self.reuse_fc)
        prop_output1_2 = self._fc_layers(self.prop_input1_2, reuse=self.reuse_fc)
        prop_output2_1 = self._fc_layers(self.prop_input2_1, reuse=self.reuse_fc)
        prop_output2_2 = self._fc_layers(self.prop_input2_2, reuse=self.reuse_fc)
        
        prop_ds1 = tf.norm(prop_output1_2-prop_output1_1,axis=1)
        prop_ds2 = tf.norm(prop_output2_2-prop_output2_1,axis=1)
        prop_statediff = tf.square(prop_ds2-prop_ds1)
        
        self.prop_q = tf.reduce_mean(prop_statediff*self.prop_actioncheck)
        
    def _create_caus_network(self):
        # Observations (input)
        self.caus_input1 = tf.placeholder("float", self._input_shape, name="caus_input1")
        self.caus_input2 = tf.placeholder("float", self._input_shape, name="caus_input2")
        # Boolean vector check for if actions 1 and 2 are equal
        self.caus_actioncheck = tf.placeholder("float", [None,], name="caus_actioncheck")
        # Boolean vector check for if reward 1 and 2 are different
        self.caus_rewardcheck = tf.placeholder("float", [None,], name="caus_rewardcheck")
        
        caus_out1 = self._fc_layers(self.caus_input1, reuse=self.reuse_fc)
        caus_out2 = self._fc_layers(self.caus_input2, reuse=self.reuse_fc)
        
        caus_state_distance = tf.exp(-tf.norm(caus_out2-caus_out1,axis=1))
        
        self.caus_q = tf.reduce_mean(caus_state_distance*self.caus_actioncheck*self.caus_rewardcheck)
        
        
    def _create_rep_network(self):
        # Observations (input)
        self.rep_input1_1 = tf.placeholder("float", self._input_shape, name="rep_input1_1")
        self.rep_input1_2 = tf.placeholder("float", self._input_shape, name="rep_input1_2")
        self.rep_input2_1 = tf.placeholder("float", self._input_shape, name="rep_input2_1")
        self.rep_input2_2 = tf.placeholder("float", self._input_shape, name="rep_input2_2")
        # Boolean vector check for if actions 1 and 2 are equal
        self.rep_actioncheck = tf.placeholder("float", [None,], name="rep_actioncheck")

        # get fc outputs (our internal state s)
        rep_out1_1 = self._fc_layers(self.rep_input1_1, reuse=self.reuse_fc)
        rep_out1_2 = self._fc_layers(self.rep_input1_2, reuse=self.reuse_fc)
        rep_out2_1 = self._fc_layers(self.rep_input2_1, reuse=self.reuse_fc)
        rep_out2_2 = self._fc_layers(self.rep_input2_2, reuse=self.reuse_fc)
        
        rep_sq_diff_s_change = tf.norm((rep_out2_2-rep_out2_1)-(rep_out1_2-rep_out1_1),axis=1)
        
        rep_state_distance = tf.exp(-tf.norm(rep_out2_1-rep_out1_1,axis=1))
        
        self.rep_q = tf.reduce_mean(rep_state_distance*rep_sq_diff_s_change*self.rep_actioncheck)

    def _base_loss(self):
        # [base A3C]
        # Taken action (input for policy)
        self.base_a = tf.placeholder("float", [None, self._action_size])
        
        # Advantage (R-V) (input for policy)
        self.base_adv = tf.placeholder("float", [None])
        if self._is_discrete:
            base_a_ind = tf.argmax(self.base_a, axis=1)
            self.log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.base_pi_linear, labels=base_a_ind)
        else:
            self.log_prob = tf.reduce_sum(self.base_distr.log_prob(self.base_a), axis=1)
            logger.debug("log_prob shape:{}".format(self.log_prob.shape))
        # Policy loss (output)
        self.policy_loss = tf.reduce_mean(self.base_adv * self.log_prob)
        logger.debug("self.policy_loss shape:{}".format(self.policy_loss.shape))
        
        # R (input for value target)
        self.base_r = tf.placeholder("float", [None])
        
        # Value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        self.value_loss = self._value_lambda * tf.reduce_sum(tf.square(self.base_v - self.base_r))
        
        
        if self._is_discrete:
            # Policy entropy
            self.entropy = self._entropy_beta - tf.reduce_sum(self.base_pi * self.base_pi_log) * self._entropy_beta
        else:
            self.entropy = tf.zeros([], dtype=np.float32)
            
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
    
    def _prop_loss(self):
        # temporal coherence loss
        prop_loss = self._proportionality_lambda * self.prop_q
        return prop_loss
        
    def _caus_loss(self):
        # temporal coherence loss
        caus_loss = self._causality_lambda * self.caus_q
        return caus_loss
    
    def _rep_loss(self):
        # repeatability loss
        rep_loss = self._repeatability_lambda * self.rep_q
        return rep_loss
        
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
                
            if self._use_proportionality:
                self.prop_loss = self._prop_loss()
                loss = loss + self.prop_loss
                
            if self._use_causality:
                self.caus_loss = self._caus_loss()
                loss = loss + self.caus_loss
                
            if self._use_repeatability:
                self.rep_loss = self._rep_loss()
                loss = loss + self.rep_loss
            
            self.total_loss = loss


    def run_base_policy_and_value(self, sess, s_t):
        # This run_base_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out = sess.run( [self.base_pi, self.base_v],
                                    feed_dict = {self.base_input : [s_t]} )
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0])
  
    def run_base_value(self, sess, s_t):
        # This run_base_value() is used for calculating V for bootstrapping at the 
        # end of LOCAL_T_MAX time step sequence.
        # When the next sequence starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        v_out = sess.run( [self.base_v],
                             feed_dict = {self.base_input : [s_t]} )
        return v_out[0]
  
    def run_vr_value(self, sess, s_t):
        vr_v_out = sess.run( self.vr_v,
                         feed_dict = {self.vr_input : [s_t]} )
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

