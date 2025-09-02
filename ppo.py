import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

import safety_gymnasium

from buffer import Buffer

gamma = 0.99
lamda = 0.97
eps_clip = 0.2
epochs = 1000
steps_per_epoch = 15000
train_iters = 80

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(60, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_pi = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 1)
        self.fc_cv = nn.Linear(128, 1)

        log_std = -0.5 * np.ones(2, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def pi(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc_pi(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def v(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.fc_v(x)
        cv = self.fc_cv(x)
        return v, cv

    def step(self, x):
        with torch.no_grad():
            pi = self.pi(x)
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v, cv = self.v(x)

        return a, logp_a, v, cv
    
    def train_net(self, buf):
        data = buf.get()
        s, a, logp_old, advantage = data['obs'], data['act'], data['logp'], data['adv']
        
        for i in range(train_iters):
            dist = self.pi(s)
            logp_a = dist.log_prob(a).sum(axis=-1)
            ratio = torch.exp(logp_a - logp_old)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1- eps_clip, 1 + eps_clip) * advantage

            v, cv = self.v(s)

            loss = -torch.min(surr1, surr2).mean() + ((v - data['ret']) ** 2).mean() + ((cv - data['cret']) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def main():
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    model = PPO()
    buf = Buffer(obs_dim=60, act_dim=2, size=steps_per_epoch, gamma=gamma, lamda=lamda)

    for epoch in range(epochs):
        o, _ = env.reset()

        ep_ret, ep_cret = 0, 0
        ep_ret_lst, ep_cret_lst = [], []

        for t in range(steps_per_epoch):
            a, logp_a, v, cv = model.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, c, d, truncated, info = env.step(a)

            buf.store(o, a, r, c, v, cv, logp_a)

            ep_ret += r
            ep_cret += c

            o = next_o

            if d or truncated:
                buf.finish_path(last_val=v, last_cval=cv)

                ep_ret_lst.append(ep_ret)
                ep_cret_lst.append(ep_cret)

                o, _ = env.reset()
                ep_ret, ep_cret = 0, 0

        model.train_net(buf)

        print(f"[Epoch {epoch}] Reward: {np.mean(ep_ret_lst):.2f}, Cost: {np.mean(ep_cret_lst):.2f}")

if __name__ == '__main__':
    main()