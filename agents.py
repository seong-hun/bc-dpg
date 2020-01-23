import numpy as np
from collections import deque
import random


class Agent:
    def __init__(self, maxlen=5, batch_size=5):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def get_action(self, obs):
        if len(self.buffer) < self.batch_size:
            return None
        else:
            batch = random.sample(self.buffer, self.batch_size)
            F = np.sum([np.outer(b[0], b[0]) for b in batch], axis=0)
            G = np.sum([np.outer(b[0], b[1]).ravel() for b in batch], axis=0)
            return F, G

    def append(self, obs):
        if obs is not None:
            self.buffer.append(obs)
