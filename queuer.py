import tensorflow as tf
import numpy as np
import random
import time
import sys
import threading
import six.moves.queue as queue
import logging

from environment.environment import Environment

logger = logging.getLogger('StRADRL.queuer')

QUEUE_LENGTH = 15

def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

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

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    def __init__(self, env, policy, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(QUEUE_LENGTH)        
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
    
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
            self.summary_writer, self.visualise)
        while True:
            self.queue.put(next(rollout_provider), timeout=600.0)
            logger.debug("added rollout. Approx queue length:{}".format(self.queue.qsize()))

        
def env_runner(env, sess, policy, num_local_steps, summary_writer, render):
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
            fetched = policy.run_base_policy_and_value(sess, last_state, last_action_reward)
            action, value_, last_features = fetched[0], fetched[1], fetched[2:]
            # argmax to convert from one-hot
            state, reward, terminal, _ = env.process(action.argmax())
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward
            
            last_state = state
            last_action_reward = np.append(action,reward)
            #logger.debug("last_action_reward:{}".format(last_action_reward))
            
            #@TODO fix information pipeline
            info = False 
            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()
            
            #@TODO investigate timestep_limit
            #timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            timestep_limit = 1000
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit: #or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                logger.info("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break
                
        if not terminal_end:
            rollout.r = policy.run_base_value(sess, last_state, last_action_reward)
        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout
       
    
