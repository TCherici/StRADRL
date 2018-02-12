# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import logging
from collections import deque

logger = logging.getLogger("StRADRL.experience")

class ExperienceFrame(object):
  def __init__(self, state, reward, action, terminal, features, pixel_change, last_action, last_reward):
    self.state = state
    self.action = action # (Taken action with the 'state')
    self.reward = reward # Received reward with the 'state'.
    self.terminal = terminal # (Whether terminated when 'state' was inputted)
    self.features = features # LSTM C and H memory states to be used to start
    self.pixel_change = pixel_change
    self.last_action = last_action # (After this last action was taken, agent move to the 'state')
    self.last_reward = last_reward# (After this last reward was received, agent move to the 'state') (Clipped)

  def get_last_action_reward(self, action_size):
    """
    Return one hot vectored last action + last reward.
    """
    return ExperienceFrame.concat_action_and_reward(self.last_action, action_size,
                                                    self.last_reward)

  @staticmethod
  def concat_action_and_reward(action, action_size, reward):
    """
    Return one hot vectored action and reward.
    """
    action_reward = np.zeros([action_size+1])
    action_reward[action] = 1.0
    action_reward[-1] = float(reward)
    return action_reward
  

class Experience(object):
  def __init__(self, history_size):
    self._history_size = history_size
    self._frames = deque(maxlen=history_size)
    # frame indices for zero rewards
    self._zero_reward_indices = deque()
    # frame indices for non zero rewards
    self._non_zero_reward_indices = deque()
    self._top_frame_index = 0


  def add_frame(self, frame):
    if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
      # Discard if terminal frame continues
      logger.info("Terminal frames continued.")
      return

    frame_index = self._top_frame_index + len(self._frames)
    was_full = self.is_full()

    # append frame
    self._frames.append(frame)
    
    # append index
    if frame_index >= 3:
      if frame.reward == 0:
        self._zero_reward_indices.append(frame_index)
      else:
        self._non_zero_reward_indices.append(frame_index)
    
    if was_full:
      self._top_frame_index += 1

      cut_frame_index = self._top_frame_index + 3
      # Cut frame if its index is lower than cut_frame_index.
      if len(self._zero_reward_indices) > 0 and self._zero_reward_indices[0] < cut_frame_index:
        self._zero_reward_indices.popleft()
        
      if len(self._non_zero_reward_indices) > 0 and self._non_zero_reward_indices[0] < cut_frame_index:
        self._non_zero_reward_indices.popleft()


  def is_full(self):
    return len(self._frames) >= self._history_size


  def sample_sequence(self, sequence_size):
    # -1 for the case if start pos is the terminated frame.
    # (Then +1 not to start from terminated frame.)
    start_pos = np.random.randint(0, len(self._frames) - sequence_size -1)
    if self._frames[start_pos].terminal:
      start_pos += 1
      # Assuming that there are no successive terminal frames.

    sampled_frames = []
    
    for i in range(sequence_size):
      frame = self._frames[start_pos+i]
      sampled_frames.append(frame)
      if frame.terminal:
        break
    
    return sampled_frames
    
  def sample_b2b_sequence(self, sequence_size):
    start_pos = np.random.randint(0, len(self._frames) - sequence_size -1)
    while self._frames[start_pos].terminal or self._frames[start_pos+1].terminal:
      start_pos += 1
      
    seq1 = []
    for i in range(sequence_size):
      frame = self._frames[start_pos+i]
      seq1.append(frame)
      if frame.terminal:
        break
    #logger.debug("start_pos:{}".format(start_pos))
    #logger.debug("seq1 length:{}".format(len(seq1)))
    # get starting point for seq2 search (at least 100 steps further)
    search_start_2 = start_pos+i+100
    #logger.debug("search_start_2:{}".format(search_start_2))
    start_2 = None
    for k in range(100):
      if self._frames[search_start_2+k].terminal:
        start_2 = search_start_2+k+1
        break
    # if after 100steps no terminal state is found, set 100th step as start
    if start_2 is None:
      start_2 = search_start_2+k
    
    #logger.debug("start_2:{}".format(start_2))
    # try 10 times to get a sequence after seq1 of the same length
    for trynum in range(10):
      for l in range(len(seq1)):
        if self._frames[start_2+l].terminal:
          start_2 = start_2+l+1
          #logger.debug("terminal at {}".format(start_2))
          break
      # else is run when no break occured in the above for loop 
      #   (that is, no terminal states have been found, thus seq2 can be created)
      else:
        seq2 = []
        for i in range(len(seq1)):
          frame = self._frames[start_2+i]
          seq2.append(frame)
        assert len(seq1)==len(seq2)
        return seq1, seq2
    # else is run after 10 attempts to find a starting point for the second sequence
    raise TypeError("Couldn't find second start point")
        
  def sample_b2b_seq_recursive(self, sequence_size):
    # try getting two random parallel sequences 
    #   from two different episodes or at least 100 steps of distance
    # if an error is called, recusively call own function to try again
    k = 0
    while True:
      try:
        k += 1
        seq1 = []
        seq2 = []
        seq1, seq2 = self.sample_b2b_sequence(sequence_size)
        return seq1, seq2
      except:
        #logger.debug("{} try".format(k))
        if k > 10:
          logger.warn("!!! B2B sampling may be broken??? !!!")
    
    
  
  def sample_rp_sequence(self):
    """
    Sample 4 successive frames for reward prediction.
    """
    if np.random.randint(2) == 0:
      from_zero = True
    else:
      from_zero = False
    
    if len(self._zero_reward_indices) == 0:
      # zero rewards container was empty
      from_zero = False
    elif len(self._non_zero_reward_indices) == 0:
      # non zero rewards container was empty
      from_zero = True

    if from_zero:
      index = np.random.randint(len(self._zero_reward_indices))
      end_frame_index = self._zero_reward_indices[index]
    else:
      index = np.random.randint(len(self._non_zero_reward_indices))
      end_frame_index = self._non_zero_reward_indices[index]

    start_frame_index = end_frame_index-3
    raw_start_frame_index = start_frame_index - self._top_frame_index

    sampled_frames = []
    
    for i in range(4):
      frame = self._frames[raw_start_frame_index+i]
      sampled_frames.append(frame)

    return sampled_frames
