from multiprocessing import Process
import numpy as np

from wrapper import make_env

class Worker(Process):
    def __init__(self, idx, env_name, queue, barrier, channel, state, mean, std):
        """mean: np.array of shape equal to the env state shape
        std: scalar, std of the obs collected by the random agent
        """
        super(Worker, self).__init__()
        self.daemon = True
        self.idx = idx
        self.env_name = env_name
        print(f"Agent {self.idx}\r", end='')

        self.queue = queue
        self.channel = channel
        self.barrier = barrier

        # state is shared memory
        self.state = state

        self.mean = mean
        self.std = std

    def run(self):
        self.env = make_env(self.env_name)

        score = 0
        while True:
            action = self.queue.get()
            if action is None:
                break
            elif action == -1: # reset
                state = np.array(self.env.reset())
                self.state[self.idx, :, :, :] = (state - self.mean) / self.std
            else:
                lazy_state, reward, done, _ = self.env.step(action)

                state = np.array(lazy_state)
                self.state[self.idx, :, :, :] = (state - self.mean) / self.std
                
                score += reward
                if done:
                    state = np.array(self.env.reset())
                    self.state[self.idx, :, :, :] = (state - self.mean) / self.std
                    self.channel.put(score)
                    score = 0

            self.barrier.put(None)