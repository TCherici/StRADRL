import tensorflow as tf
import numpy as np
import random
import time
import sys
import threading
import six.moves.queue as queue
import logging

from environment.environment import Environment
from model.fc_model import UnrealModel

logger = logging.getLogger('StRADRL.queuer')


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
    def __init__(self, flags, env, global_net, obs_size, device, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(flags.queue_length)        
        self.num_local_steps = flags.local_t_max
        self.env = env
        self.action_size, self.is_discrete = self.env.get_action_size(flags.env_name)
        self.policy = UnrealModel(self.action_size,
                                  self.is_discrete,
                                  obs_size,
                                  0,
                                  flags.entropy_beta,
                                  device)
        self.sess = None
        self.visualise = visualise
        self.sync = self.policy.sync_from(global_net, name="env_runner")
        self.global_net = global_net
        self.env_max_steps = flags.env_max_steps
        self.action_freq = flags.action_freq
        self.env_runner_sync = flags.env_runner_sync
    
    def start_runner(self, sess):
        logger.debug("starting runner")
        self.sess = sess
        self.start()
        
    def run(self):
        with self.sess.as_default():
            self._run()
    
    def _run(self):
        
        rollout_provider = env_runner(self.env, self.sess, self.policy, self.num_local_steps, self.env_max_steps,\
            self.action_freq, self.env_runner_sync, self.sync, self.global_net, self.visualise, self.action_size, self.is_discrete)
            
        while True:
            self.queue.put(next(rollout_provider), timeout=600.0)
            #logger.debug("added rollout. Approx queue length:{}".format(self.queue.qsize()))
            
def boltzmann(pi_values):
    # take action with chance equal to distribution
    return np.random.choice(range(len(pi_values)), p=pi_values)
    
    
def eps_greedy(pi_values, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(range(len(pi_values)))
    else:
        return np.argmax(pi_values)
        
def onehot(action, action_size, dtype="float32"):
    action_oh = np.zeros([action_size], dtype=dtype)
    action_oh[action] = 1.
    return action_oh
    
        
def env_runner(env, sess, policy, num_local_steps, env_max_steps, action_freq, env_runner_sync, syncfunc, global_net, render, action_size, is_discrete):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    logger.debug("resetting env in session {}".format(sess))
    last_state = env.reset()
    sess.run(syncfunc)
    length = 0
    rewards = 0.
    itercount = 0
    
    while True:
        itercount += 1
        sess.run(syncfunc)
        terminal_end = False
        rollout = PartialRollout()
        for _ in range(num_local_steps):
            fetched = policy.run_base_policy_and_value(sess, last_state)
            pi, value_ = fetched[0], fetched[1]
            
            #logger.debug("pi:{}".format(pi))
            #logger.debug("action:{}".format(action))
            
            if is_discrete:
                #chosenaction = boltzmann(pi)
                chosenaction = eps_greedy(pi, epsilon=0.05)
                #chosenaction = np.argmax(pi)
                action = onehot(chosenaction, len(pi), dtype="int32")
            else:
                action = pi
            state, reward, terminal, pixel_change = env.process(action)
            if action_freq > 0.:
                time.sleep(1.0/action_freq)
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, pixel_change)
            length += 1
            rewards += reward
            
            
            last_state = state
            
            #timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= env_max_steps:
                terminal_end = True
                rollout.terminal = True
                # the if condition below has been disabled because deepmind lab has no metadata
                #if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                last_state = env.reset()
                #logger.info("Ep. finish. Tot rewards: %d. Length: %d" % (rewards, length))
                if itercount % env_runner_sync == 0:
                    # moved sync to start of cycle
                    assert True
                length = 0
                rewards = 0.
                break
                
        if not terminal_end:
            rollout.r = policy.run_base_value(sess, last_state)
        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout
        
       
    

