import torch, torchvision
import numpy as np
import os, sys, argparse
from collections import deque
from multiprocessing import Queue, RawArray
from ctypes import c_uint8, c_float

from tqdm import tqdm

from network import Policy, IntrinsicCuriosityModule
from wrapper import make_env
from replay import ReplayMemory
from util import plot
from worker import Worker

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch-size', default=32, type=int, help="Batch size")
parser.add_argument('-g', '--gamma', default=0.99, type=float, help="Gamma")
parser.add_argument('-t', '--tau', default=0.96, type=float, help="Tau for GAE")


class Trainer:
    def __init__(self, env_name, mode, batch_size, gamma, tau):

        assert mode.upper() in ['IDF', 'RANDOM']
        self.mode = mode.upper()
        self.batch_size = batch_size # batch_size == number of envs
        
        self.queues = [Queue() for i in range(batch_size)]
        self.barrier = Queue() # use to block Trainer until all envs finish updating
        self.channel = Queue() # envs send their total scores after each episode

        # sh_* variables are shared between processes
        self.sh_state, self.sh_reward, self.sh_done  = self.init_shared()
        self.workers = [
            Worker(i, env_name, self.queues[i], self.barrier, self.channel, self.sh_state, self.sh_reward, self.sh_done) for i in range(batch_size)
        ]
        self.start_workers()

        tmp_env = make_env(env_name)
        self.c_in = tmp_env.observation_space.shape[0]
        self.num_actions = tmp_env.action_space.n
        del tmp_env

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gamma  = gamma # reward discounting factor
        self.tau    = tau   # for GAE (Generalized Advantage Estimation)
        self.lmbda  = 0.1   # weight of the policy/value loss
        self.beta   = 0.2   # beta: forward; (1 - beta): inverse loss
        self.eta    = 0.01  # weight of intrinsic reward

        self.model  = Policy(self.c_in, self.num_actions).to(self.device)
        self.icm    = IntrinsicCuriosityModule(self.c_in, self.num_actions).to(self.device)

        self.optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.icm.parameters()),
            lr=1e-3
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss()
    
    def reset_workers(self):
        for q in self.queues:
            q.put(-1)

    def broadcast_actions(self, actions):
        for i in range(self.batch_size):
            self.queues[i].put(actions[i].item())

    def start_workers(self):
        for worker in self.workers:
            worker.start()

    def stop_workers(self):
        for q in self.queues:
            q.put(None)

    def wait_for_workers(self):
        for i in range(self.batch_size):
            self.barrier.get()

    def init_shared(self):
        shape = (self.batch_size, 4, 42, 42) # fixed

        state = np.zeros(shape, dtype=np.float32)
        state = RawArray(c_float, state.reshape(-1))
        state = np.frombuffer(state, c_float).reshape(shape)

        reward = np.zeros((self.batch_size, 1), dtype=np.float32)
        reward = RawArray(c_float, reward)
        reward = np.frombuffer(reward, c_float).reshape(self.batch_size, 1)

        done = np.zeros((self.batch_size, 1), dtype=np.uint8)
        done = RawArray(c_uint8, done)
        done = np.frombuffer(done, c_uint8).reshape(self.batch_size, 1)
        
        return state, reward, done
    
    
    def train(self, T_max):
        step = 0
        self.num_lookahead = 5
        

        self.reset_workers()
        self.wait_for_workers()

        stat = {
            'ploss': [],
            'vloss': [],
            'score': [],
            'int_reward': [],
            'ext_reward': [],
            'entropy': [],
            'running_loss': 0
        }

        while step < T_max:

            # these will keep tensors, which we'll use later for backpropagation
            values        = []
            log_probs     = []
            rewards       = []
            entropies     = []

            actions       = []
            actions_pred  = []
            features      = []
            features_pred = []


            state = torch.from_numpy(self.sh_state).to(self.device)

            for i in range(self.num_lookahead):
                step += self.batch_size

                logit, value = self.model(state)
                prob = torch.softmax(logit, dim=1)
                log_prob = torch.log_softmax(logit, dim=1)
                entropy = -(prob * log_prob).sum(1, keepdim=True)

                action = prob.multinomial(1)
                sampled_lp = log_prob.gather(1, action)

                # one-hot action
                oh_action = torch.zeros(self.batch_size, self.num_actions, device=self.device).scatter_(1,action,1)

                self.broadcast_actions(action)
                self.wait_for_workers()

                next_state = torch.from_numpy(self.sh_state).to(self.device)
                s1, s1_pred, action_pred = self.icm(state, oh_action, next_state)

                ext_reward = torch.from_numpy(np.clip(self.sh_reward, a_min=-1, a_max=1)).to(self.device)
                int_reward = 0.5 * self.eta * (s1.detach() - s1_pred.detach()).pow(2).sum(dim=1, keepdim=True)
                reward = ext_reward + int_reward


                done_mask = torch.tensor(1.0 - self.sh_done.astype(np.float32), dtype=torch.float).to(self.device)
                value *= done_mask

                state = next_state

                # save variables for gradient descent
                values.append(value)
                log_probs.append(sampled_lp)
                rewards.append(reward)
                entropies.append(entropy)

                actions.append(action.flatten())
                actions_pred.append(action_pred)
                features.append(s1)
                features_pred.append(s1_pred)

                stat['entropy'].append(entropy.sum(dim=1).mean().item())
                stat['int_reward'].append(int_reward.mean().item())
                stat['ext_reward'].append(ext_reward.mean().item())


            state = torch.from_numpy(self.sh_state.astype(np.float32)).to(self.device)
            with torch.no_grad():
                _, R = self.model(state) # R is the estimated return
            
            done_mask = torch.tensor(1.0 - self.sh_done.astype(np.float32), dtype=torch.float).to(self.device)
            R *= done_mask
            values.append(R)

            ploss = 0
            vloss = 0
            fwd_loss = 0
            inv_loss = 0

            # Generalized Advantage Estimation
            delta = torch.zeros((self.batch_size, 1), dtype=torch.float, device=self.device)
            for i in reversed(range(self.num_lookahead)):
                R = rewards[i] + self.gamma * R
                advantage = R - values[i]
                vloss += (0.5 * advantage.pow(2)).mean()

                delta = rewards[i] + self.gamma * values[i + 1].detach() - values[i].detach()
                ploss += -(log_probs[i] * delta + 0.01 * entropies[i]).mean() # beta = 0.01

                fwd_loss += 0.5 * (features[i] - features_pred[i]).pow(2).sum(dim=1).mean()
                inv_loss += self.cross_entropy(actions_pred[i], actions[i])


            while not self.channel.empty():
                # print("Getting score: ", end="")
                score = self.channel.get()
                stat['score'].append(score)
                # print(score, "| size:", len(stat['score']))

            self.optim.zero_grad()
            loss = self.lmbda * (ploss + 0.5 * vloss) + self.beta * fwd_loss + (1 - self.beta) * inv_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)

            self.optim.step()

            stat['ploss'].append(ploss.item() / self.num_lookahead)
            stat['vloss'].append(vloss.item() / self.num_lookahead)
            stat['running_loss'] = 0.99 * stat['running_loss'] + 0.01 * loss.item() / self.num_lookahead

            if len(stat['score']) > 20 and step % (self.batch_size * 1000) == 0:
                print(f"Step {step} | Running loss: {stat['running_loss']:.2f} | Running score: {np.mean(stat['score'][-10:]):.2f}")
                if step % (self.batch_size * 50000) == 0:
                    plot(step, stat['score'], stat['int_reward'], stat['ext_reward'], stat['ploss'], stat['vloss'], stat['entropy'], name="CuriosityBN.png")
        

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    args = parser.parse_args()

    batch_size    = args.batch_size
    gamma         = args.gamma
    tau           = args.tau

    # force
    batch_size = 16
    T_max = 8000000 # 20M steps


    trainer = Trainer(env_name="PongNoFrameskip-v4", mode="IDF", batch_size=batch_size, gamma=gamma, tau=tau)
    trainer.train(T_max)