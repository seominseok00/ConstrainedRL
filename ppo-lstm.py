import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

import safety_gymnasium

from buffer_lstm import LSTMBuffer, split_total_steps_to_chunks

gamma = 0.99
lamda = 0.97
eps_clip = 0.2
epochs = 1000
steps_per_epoch = 15000
seq_len = 100
train_iters = 80

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(60, 128)
        self.fc2 = nn.Linear(128, 128)
        self.lstm = nn.LSTM(128, 128)
        self.fc_pi = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 1)
        self.fc_cv = nn.Linear(128, 1)

        log_std = -0.5 * np.ones(2, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def encode(self, x):
        x1 = F.tanh(self.fc1(x))
        x2 = F.tanh(self.fc2(x1))
        return x2
    
    def lstm_forward(self, x, h_in):
        x_lstm, h_out = self.lstm(x, h_in)
        return x_lstm, h_out

    def heads(self, x):
        mu = self.fc_pi(x)
        std = torch.exp(self.log_std)
        v = self.fc_v(x)
        cv = self.fc_cv(x)
        return mu, std, v, cv
    
    def pi(self, x, h_in):
        T, B, _ = x.shape
        xmlp = self.encode(x.view(T*B, -1))
        xlstm, h_out = self.lstm_forward(xmlp.view(T, B, -1), h_in)
        xlstm = xlstm.view(T*B, -1)
        x = xmlp + xlstm
        mu = self.fc_pi(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std), h_out

    def v(self, x, h_in):
        T, B, _ = x.shape
        x_mlp = self.encode(x.view(T*B, -1))
        x_lstm, _ = self.lstm_forward(x_mlp.view(T, B, -1), h_in)
        x_lstm = x_lstm.view(T*B, -1)
        x = x_mlp + x_lstm
        v = self.fc_v(x)
        cv = self.fc_cv(x)
        return v.squeeze(-1), cv.squeeze(-1)

    def step(self, x, h_in):
        with torch.no_grad():
            x = x.view(1, 1, -1)
            dist, h_out = self.pi(x, h_in)
            a = dist.sample()
            logp_a = dist.log_prob(a).sum(axis=-1)
            v, cv = self.v(x, h_in)
        return a.squeeze(0), logp_a, v, cv, h_out


    def train_net(self, buf):
        data = buf.get()
        
        chunks = split_total_steps_to_chunks(data, episode_steps=1000, chunk_T=seq_len, drop_last_incomplete=True)

        for i in range(train_iters):
            for data in chunks:
                T, B, _ = data['obs'].shape
                s, a, logp_old, advantage = data['obs'], data['act'], data['logp'], data['adv']

                a = a.view(T*B, -1)
                logp_old = logp_old.view(T*B)
                advantage = advantage.view(T*B)
                ret = data['ret'].view(T*B)
                cret = data['cret'].view(T*B)

                dist, _ = self.pi(s, (data['h0_h'], data['h0_c']))
                logp_a = dist.log_prob(a).sum(axis=-1)
                ratio = torch.exp(logp_a - logp_old)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1- eps_clip, 1 + eps_clip) * advantage

                v, cv = self.v(s, (data['h0_h'], data['h0_c']))

                loss = -torch.min(surr1, surr2).mean() + ((v - ret) ** 2).mean() + ((cv - cret) ** 2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def main():
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    model = PPO()
    buf = LSTMBuffer(obs_dim=60, act_dim=2, hid_dim=128, size=steps_per_epoch, gamma=gamma, lamda=lamda)

    for epoch in range(epochs):
        h_out = (torch.zeros([1, 1, 128], dtype=torch.float), torch.zeros([1, 1, 128], dtype=torch.float))
        o, _ = env.reset()

        ep_ret, ep_cret = 0, 0
        ep_ret_lst, ep_cret_lst = [], []

        for t in range(steps_per_epoch):
            h_in = h_out
            a, logp_a, v, cv, h_out = model.step(torch.as_tensor(o, dtype=torch.float32), h_in)

            next_o, r, c, d, truncated, info = env.step(a)
            
            buf.store(o, a, r, c, v, cv, logp_a, h_in[0], h_in[1])

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