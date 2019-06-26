from multiprocessing import Process
import numpy as np

from wrapper import make_env

class Worker(Process):
    def __init__(self, idx, env_name, queue, barrier, channel, state, reward, done):
        super(Worker, self).__init__()
        self.daemon = True
        self.idx = idx
        print(f"Agent {self.idx}\r", end='')

        self.env = make_env(env_name)
        self.queue = queue
        self.channel = channel
        self.barrier = barrier

        # Everyone's sharing these
        self.state = state
        self.reward = reward
        self.done = done

    def run(self):
        score = 0
        while True:
            action = self.queue.get()
            if action is None:
                break
            elif action == -1: # reset
                self.state[self.idx, :, :, :] = np.array(self.env.reset())
            else:
                lazy_state, reward, done, _ = self.env.step(action)

                self.state[self.idx, :, :, :] = np.array(lazy_state)
                self.reward[self.idx, 0] = reward
                self.done[self.idx, 0] = done
                
                score += reward
                if self.done[self.idx, 0]:
                    self.state[self.idx, :, :, :] = np.array(self.env.reset())
                    self.channel.put(score)
                    score = 0

            self.barrier.put(None)