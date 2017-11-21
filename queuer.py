import tensorflow as tf
import numpy as np
import random
import time
import sys
import threading
import six.moves.queue as queue
import logging

from environment.environment import Environment
from model.model import UnrealModel
from model.base import BaseModel

logger = logging.getLogger('StRADRL.queuer')

QUEUE_LENGTH = 5
TIMESTEP_LIMIT = 2000


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.pixel_changes = []

    def add(self, state, action, reward, value, terminal, pixel_change):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.pixel_changes += [pixel_change]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.pixel_changes.extend(other.pixel_changes)

class RunnerThread(threading.Thread):
    def __init__(self, env, global_net, action_size, entropy_beta, device, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(QUEUE_LENGTH)        
        self.num_local_steps = num_local_steps
        self.env = env
        self.policy = BaseModel(3,
                                action_size,
                                0,
                                entropy_beta,
                                device)
        self.global_net = global_net
        self.sess = None
        self.visualise = visualise
        self.sync = self.policy.sync_from#(global_net, name="net_0")
    
    def start_runner(self, sess):
        logger.debug("starting runner")
        self.sess = sess
        self.start()
        
    def run(self):
        with self.sess.as_default():
            self._run()
    
    def _run(self):
        rollout_provider = env_runner(self.env, self.sess, self.policy, self.num_local_steps, \
            self.sync, self.global_net, self.visualise)
        while True:
            self.queue.put(next(rollout_provider), timeout=600.0)
            #logger.debug("added rollout. Approx queue length:{}".format(self.queue.qsize()))
            
def boltzmann(pi_values):
    # take action with chance equal to distribution
    return np.random.choice(range(len(pi_values)), p=pi_values)
    
    
#@TODO implement
def eps_greedy(pi_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(range(len(pi_values)))
    else:
        return np.argmax(pi_values)
        
def onehot(action, action_size, dtype="float32"):
    action_oh = np.zeros([action_size], dtype=dtype)
    action_oh[action] = 1.
    return action_oh
    
        
def env_runner(env, sess, policy, num_local_steps, syncfunc, global_net, render):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    logger.debug("resetting env in session {}".format(sess))
    last_state, last_action_reward = env.reset()
    #logger.debug(last_action_reward.shape)
    length = 0
    rewards = 0
    
    while True:
        terminal_end = False
        rollout = PartialRollout()
        for _ in range(num_local_steps):
            fetched = policy.run_base_policy_and_value(sess, last_state)
            pi, value_ = fetched[0], fetched[1]
            
            #@TODO decide if argmax or probability, if latter fix experience replay selection
            #chosenaction = boltzmann(pi)
            chosenaction = np.argmax(pi)
            action = pi
            #action = onehot(chosenaction, len(pi))
            
            state, reward, terminal, pixel_change = env.process(chosenaction)
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, pixel_change)
            length += 1
            rewards += reward
            
            
            last_state = state
            
            #timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= TIMESTEP_LIMIT:
                terminal_end = True
                rollout.terminal = True
                # the if condition below has been disabled because deepmind lab has no metadata
                #if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                last_state, _ = env.reset()
                #policy.reset_state()
                logger.info("Ep. finished. \nTot rewards: %d. Length: %d. Value: %f" % (rewards, length, value_))
                #logger.debug(syncfunc)
                sess.run(syncfunc(global_net, name="env_runner_net"))
                length = 0
                rewards = 0
                break
                
        if not terminal_end:
            rollout.r = policy.run_base_value(sess, last_state)
        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout
       
    
