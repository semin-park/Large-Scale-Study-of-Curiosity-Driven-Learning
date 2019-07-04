import torch
import numpy as np
import datetime
import os, sys, argparse
from collections import deque
from multiprocessing import Queue, RawArray
from ctypes import c_uint8, c_float
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import RunningMeanStd

from tqdm import tqdm

from network import Policy, FeatureEncoder, ForwardModel, InverseModel, IntrinsicCuriosityModule
from wrapper import make_env
from util import plot
from worker import Worker

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch-size', default=32, type=int, help="Batch size")
parser.add_argument('-g', '--gamma', default=0.99, type=float, help="Gamma")
parser.add_argument('-r', '--random', default=True, type=bool, help="Use random features")
parser.add_argument('-n', '--name', default='graph.png', type=str, help="Name of the graph (must append .png, .jpeg etc)")
parser.add_argument('-e', '--env', default='PongNoFrameskip-v4', type=str, help="Name of the environment to use")


class Trainer:
    def __init__(self, env_name, batch_size, gamma, use_random_features):

        self.random = use_random_features
        self.batch_size = batch_size # batch_size == number of envs
        
        self.queues = [Queue() for i in range(batch_size)]
        self.barrier = Queue() # use to block Trainer until all envs finish updating
        self.channel = Queue() # envs send their total scores after each episode

        tmp_env = make_env(env_name)
        self.c_in = tmp_env.observation_space.shape[0]
        self.num_actions = tmp_env.action_space.n
        mean, std = self.mean_std_from_random_agent(tmp_env, 10000)

        # sh_state is shared between processes
        self.sh_state  = self.init_shared(tmp_env.observation_space.shape)

        self.workers = [
            Worker(i, env_name, self.queues[i], self.barrier, self.channel, self.sh_state, mean, std) for i in range(batch_size)
        ]
        self.start_workers()


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gamma  = gamma # reward discounting factor

        self.model  = Policy(self.c_in, self.num_actions).to(self.device)
        self.icm    = IntrinsicCuriosityModule(self.c_in, self.num_actions, self.random).to(self.device)

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

    def init_shared(self, obs_shape):
        shape = (self.batch_size,) + obs_shape

        state = np.zeros(shape, dtype=np.float32)
        state = RawArray(c_float, state.reshape(-1))
        state = np.frombuffer(state, c_float).reshape(shape)
        
        return state

    @staticmethod
    def mean_std_from_random_agent(env, steps):
        obs = np.empty((steps,) + env.observation_space.shape, dtype=np.float32)
        
        env.reset()
        for i in range(steps):
            state, _, done, _ = env.step(env.action_space.sample())
            obs[i] = np.array(state)
            if done:
                env.reset()
        mean = np.mean(obs, 0)
        std = np.std(obs, 0).mean()
        return mean, std
    
    
    def train(self, T_max, graph_name=None):
        step = 0
        self.num_lookahead = 5
        

        self.reset_workers()
        self.wait_for_workers()

        stat = {
            'ploss': [],
            'vloss': [],
            'score': [],
            'int_reward': [],
            'entropy': [],
            'fwd_kl_div': [],
            'running_loss': 0
        }

        reward_tracker = RunningMeanStd()
        reward_buffer = np.empty((self.batch_size, self.num_lookahead),dtype=np.float32)
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
                

                with torch.no_grad():
                    int_reward = 0.5 * (s1 - s1_pred).pow(2).sum(dim=1, keepdim=True)
                reward_buffer[:, i] = int_reward.cpu().numpy().ravel()

                state = next_state

                # save variables for gradient descent
                values.append(value)
                log_probs.append(sampled_lp)
                rewards.append(int_reward)
                entropies.append(entropy)

                if not self.random:
                    actions.append(action.flatten())
                    actions_pred.append(action_pred)
                features.append(s1)
                features_pred.append(s1_pred)

                stat['entropy'].append(entropy.sum(dim=1).mean().item())
                stat['fwd_kl_div'].append(torch.kl_div(s1_pred, s1).mean().item())

            # may have to update reward_buffer with gamma first
            reward_mean, reward_std, count =  mpi_moments(reward_buffer.ravel())
            reward_tracker.update_from_moments(reward_mean, reward_std ** 2, count)
            std = np.sqrt(reward_tracker.var)
            rewards = [rwd / std for rwd in rewards]
            for rwd in rewards:
                stat['int_reward'].append(rwd.mean().item())

            state = torch.from_numpy(self.sh_state.astype(np.float32)).to(self.device)
            with torch.no_grad():
                _, R = self.model(state) # R is the estimated return
            
            values.append(R)

            ploss = 0
            vloss = 0
            fwd_loss = 0
            inv_loss = 0

            delta = torch.zeros((self.batch_size, 1), dtype=torch.float, device=self.device)
            for i in reversed(range(self.num_lookahead)):
                R = rewards[i] + self.gamma * R
                advantage = R - values[i]
                vloss += (0.5 * advantage.pow(2)).mean()

                delta = rewards[i] + self.gamma * values[i + 1].detach() - values[i].detach()
                ploss += -(log_probs[i] * delta + 0.01 * entropies[i]).mean() # beta = 0.01

                fwd_loss += 0.5 * (features[i] - features_pred[i]).pow(2).sum(dim=1).mean()
                if not self.random:
                    inv_loss += self.cross_entropy(actions_pred[i], actions[i])


            self.optim.zero_grad()

            # inv_loss is 0 if using random features
            loss = ploss + vloss + fwd_loss + inv_loss # 2018 Large scale curiosity paper simply sums them (no lambda and beta anymore)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.icm.parameters()), 40)
            self.optim.step()

            while not self.channel.empty():
                score = self.channel.get()
                stat['score'].append(score)


            stat['ploss'].append(ploss.item() / self.num_lookahead)
            stat['vloss'].append(vloss.item() / self.num_lookahead)
            stat['running_loss'] = 0.99 * stat['running_loss'] + 0.01 * loss.item() / self.num_lookahead

            if len(stat['score']) > 20 and step % (self.batch_size * 1000) == 0:
                now = datetime.datetime.now().strftime("%H:%M")
                print(
                    f"Step {step: <10} | Running loss: {stat['running_loss']:.4f} | Running score: {np.mean(stat['score'][-10:]):.2f} | Time: {now}"
                )
                if graph_name is not None and step % (self.batch_size * 10000) == 0:
                    plot(step, stat['score'], stat['int_reward'], stat['ploss'], stat['vloss'], stat['entropy'], name=graph_name)
        

if __name__ == '__main__':
    args = parser.parse_args()

    env_name      = args.env
    batch_size    = args.batch_size
    gamma         = args.gamma
    use_random    = args.random
    graph_name    = args.name

    # force
    batch_size = 16
    T_max = 100000000 # 100M steps

    trainer = Trainer(env_name=env_name, batch_size=batch_size, gamma=gamma, use_random_features=use_random)
    trainer.train(T_max, graph_name)
