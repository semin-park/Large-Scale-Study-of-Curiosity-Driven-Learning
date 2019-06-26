from collections import deque
import random
import numpy as np

class ReplayMemory(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def push(self, state, action, reward, next_state):
        transition = (state, action, reward, next_state)
        self.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states      = np.stack(states)
        actions     = np.array(actions)
        rewards     = np.array(rewards)
        next_states = np.stack(next_states)

        return (states, actions, rewards, next_states)