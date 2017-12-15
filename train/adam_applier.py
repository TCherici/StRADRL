# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator

logger = logging.getLogger("StRADRL.adam_applier")

class AdamApplier(object):

    def __init__(self,
                 learning_rate,
                 clip_norm=40.0,
                 device="/cpu:0",
                 name="AdamApplier"):
        self._name = name
        self._learning_rate = learning_rate
        self._clip_norm = clip_norm
        self._device = device
        self._opt = tf.train.AdamOptimizer(self._learning_rate)

    
    
    def minimize_local(self, loss, global_var_list, local_var_list):
        """
        minimize loss and apply gradients to global vars.
        """
        with tf.device(self._device):
            logger.debug("appling grads")
            var_refs = [v._ref() for v in local_var_list]
            local_gradients = tf.gradients(loss, var_refs,
                                gate_gradients=False,
                                aggregation_method=None,
                                colocate_gradients_with_ops=False)
            
            local_gradients, _ = tf.clip_by_global_norm(local_gradients, self._clip_norm)
            
            
            norms = tf.global_norm(local_gradients)
            
            #self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(local_var_list, global_var_list)])
            
            grads_and_vars = list(zip(local_gradients, global_var_list))

            return self._opt.apply_gradients(grads_and_vars), norms
