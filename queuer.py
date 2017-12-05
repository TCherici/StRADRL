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

logger = logging.getLogger('StRADRL.queuer')

QUEUE_LENGTH = 30


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
        self.features = []
        self.pixel_changes = []

    def add(self, state, action, reward, value, terminal, features, pixel_change):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.pixel_changes += [pixel_change]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        self.pixel_changes.extend(other.pixel_changes)

class RunnerThread(threading.Thread):
    def __init__(self, env, global_net, action_size, entropy_beta, device, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(QUEUE_LENGTH)        
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = UnrealModel(action_size,
                                         0,
                                         entropy_beta,
                                         device)
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.sync = self.policy.sync_from(global_net)
    
    def start_runner(self, sess, summary_writer):
        logger.debug("starting runner")
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()
        
    def run(self):
        with self.sess.as_default():
            self._run()
    
    def _run(self):
        rollout_provider = env_runner(self.env, self.sess, self.policy, self.num_local_steps, \
            self.sync, self.summary_writer, self.visualise)
        while True:
            self.queue.put(next(rollout_provider), timeout=600.0)
            #logger.debug("added rollout. Approx queue length:{}".format(self.queue.qsize()))
            
def boltzmann(pi_values):
    # take action with chance equal to distribution
    return np.random.choice(range(len(pi_values)), p=pi_values)
    
    
#@TODO implement
def eps_greedy(pi_values, epsilon):
    return None
        
def env_runner(env, sess, policy, num_local_steps, syncfunc, summary_writer, render):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    logger.debug("resetting env in session {} and syncing to global".format(sess))
    last_state, last_action_reward = env.reset()
    #logger.debug(last_action_reward.shape)
    length = 0
    rewards = 0
    
    while True:
        terminal_end = False
        rollout = PartialRollout()
        for _ in range(num_local_steps):
            fetched = policy.run_base_policy_and_value(sess, last_state, last_action_reward)
            action, value_, last_features = fetched[0], fetched[1], fetched[2:]
            
            #@TODO decide if argmax or probability, if latter fix experience replay selection
            chosenaction = boltzmann(action)
            #chosenaction = np.argmax(action)
            
            state, reward, terminal, pixel_change = env.process(chosenaction)
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features, pixel_change)
            length += 1
            rewards += reward
            
            last_state = state
            last_action_reward = np.append(action,reward)
            
            #@TODO fix information pipeline
            info = False 
            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()
            
            #timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            timestep_limit = 100000
            if terminal or length >= timestep_limit:
                terminal_end = True
                # the if condition below has been disabled because deepmind lab has no metadata
                #if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                last_state, last_action_reward = env.reset()
                policy.reset_state()
                last_features = policy.base_lstm_state_out
                logger.info("Episode finished (terminal:%s). Sum of rewards: %d. Length: %d" % (terminal,rewards, length))
                sess.run(syncfunc)
                length = 0
                rewards = 0
                break
                
        if not terminal_end:
            rollout.r = policy.run_base_value(sess, last_state, last_action_reward)
        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout
       
    
