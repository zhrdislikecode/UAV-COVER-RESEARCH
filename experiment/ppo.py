import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  PPO 网络
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)  # logits


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state)


# ============================================================
#  PPO Agent — 支持多 UAV 独立轨迹
# ============================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=1e-4, gamma=0.99, clip_eps=0.2,
                 gae_lambda=0.95, update_epochs=8, batch_size=128,
                 horizon=2048, ent_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.ent_coef = ent_coef

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # 主缓冲区
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

        # 当前 episode 的临时缓冲: list of (uav_idx, state, action, reward, next, lp, v)
        self._ep_buffer = []
        self._current_uav = 0
        self._last_log_prob = None
        self._last_value = None

    # ---- DQN 兼容接口 ----
    def choose_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.actor(state_t)
            value = self.critic(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self._last_log_prob = dist.log_prob(action).item()
        self._last_value = value.item()
        return action.item()

    def set_uav_index(self, idx: int):
        """标记当前处理的 UAV 索引，用于按 UAV 拆分轨迹"""
        self._current_uav = idx

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        self._ep_buffer.append((
            self._current_uav, state, action, reward, next_state,
            self._last_log_prob, self._last_value,
        ))

    def end_episode(self, num_uavs: int):
        """Episode 结束：按 UAV 拆分轨迹，末尾标记 done=True，移入主缓冲区"""
        if not self._ep_buffer:
            return

        # 按 UAV 分组
        trajs = {i: [] for i in range(num_uavs)}
        for item in self._ep_buffer:
            uav_idx, s, a, r, ns, lp, v = item
            trajs[uav_idx].append((s, a, r, ns, lp, v))

        for uav_idx in range(num_uavs):
            traj = trajs[uav_idx]
            if not traj:
                continue
            for j, (s, a, r, ns, lp, v) in enumerate(traj):
                self.states.append(s)
                self.actions.append(a)
                self.rewards.append(r)
                self.dones.append(1.0 if j == len(traj) - 1 else 0.0)
                self.log_probs.append(lp)
                self.values.append(v)

        self._ep_buffer.clear()

    # ---- 训练 ----
    def train(self):
        if len(self.states) < self.horizon:
            return

        returns, advantages = self._compute_gae()

        states_t = torch.FloatTensor(np.array(self.states)).to(device)
        actions_t = torch.LongTensor(self.actions).to(device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)

        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / adv_std

        dataset_size = len(self.states)
        for _ in range(self.update_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start:start + self.batch_size]
                s = states_t[idx]; a = actions_t[idx]
                old_lp = old_log_probs_t[idx]
                ret = returns_t[idx]; adv = advantages_t[idx]

                logits = self.actor(s)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                log_ratio = new_lp - old_lp
                ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                values = self.critic(s).squeeze(-1)
                critic_loss = nn.MSELoss()(values, ret)

                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
                self.optimizer.step()

        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.dones.clear(); self.values.clear()

    def _compute_gae(self):
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float64)
        returns = np.zeros(T, dtype=np.float64)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            delta = (self.rewards[t]
                     + self.gamma * next_value * (1.0 - self.dones[t])
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t] if self.dones[t] == 0.0 else 0.0
        return returns, advantages

    def update_target(self):
        pass

    def save_model(self, path):
        torch.save({'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict()}, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
