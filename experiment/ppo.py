import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_ppo_state(uav):
    """构建 PPO 专用状态向量 (6维)：[dx, dy, cvx, cvy, dist, battery_ratio]

    相比原始绝对坐标 [cx, cy, cvx, cvy, ux, uy]，使用 UAV 与集群的
    相对位置，使状态平移不变，更容易泛化。
    """
    raw = uav.get_state()
    if not np.any(raw):
        return raw
    dx = raw[4] - raw[0]
    dy = raw[5] - raw[1]
    dist = np.sqrt(dx * dx + dy * dy)
    battery_ratio = uav.current_battery_capacity / uav.total_battery_capacity
    return np.array([dx, dy, raw[2], raw[3], dist, battery_ratio], dtype=np.float32)


# ============================================================
#  RunningMeanStd — 运行时均值/标准差归一化
# ============================================================
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        """用一批数据更新统计量 (Welford 在线算法)"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


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
        # 正交初始化，有助于训练稳定
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state)


# ============================================================
#  PPO Agent — 支持多 UAV 独立轨迹
# ============================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=1e-4, gamma=0.99, clip_eps=0.2,
                 gae_lambda=0.95, update_epochs=12, batch_size=128,
                 horizon=1024, ent_coef=0.5, ent_decay=0.98, ent_min=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.ent_coef = ent_coef
        self.ent_decay = ent_decay
        self.ent_min = ent_min

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.9)

        # 状态 & 奖励归一化器
        self.state_norm = RunningMeanStd((state_dim,))
        self.reward_norm = RunningMeanStd((1,))

        # 主缓冲区
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

        # 当前 episode 的临时缓冲
        self._ep_buffer = []
        self._current_uav = 0
        self._last_log_prob = None
        self._last_value = None

    def choose_action(self, state):
        s = self.state_norm.normalize(state)
        state_t = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.actor(state_t)
            value = self.critic(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self._last_log_prob = dist.log_prob(action).item()
        self._last_value = value.item()
        return action.item()

    def set_uav_index(self, idx: int):
        self._current_uav = idx

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        self._ep_buffer.append((
            self._current_uav, state, action, reward, next_state,
            self._last_log_prob, self._last_value,
        ))

    def end_episode(self, num_uavs: int):
        if not self._ep_buffer:
            return

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

        # 更新状态 & 奖励归一化器统计量
        states_arr = np.array(self.states)
        rewards_arr = np.array(self.rewards)
        self.state_norm.update(states_arr)
        self.reward_norm.update(rewards_arr.reshape(-1, 1))

        returns, advantages = self._compute_gae()

        # 归一化状态
        states_norm = self.state_norm.normalize(states_arr)
        states_t = torch.FloatTensor(states_norm).to(device)
        actions_t = torch.LongTensor(self.actions).to(device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)

        # 优势归一化
        adv_std = advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / adv_std

        # 回报裁剪（防止 value loss 爆炸）
        returns_t = torch.clamp(returns_t, -50.0, 50.0)

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
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                values = self.critic(s).squeeze(-1)
                critic_loss = nn.MSELoss()(values, ret)

                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
                self.optimizer.step()

        self.lr_scheduler.step()
        self.ent_coef = max(self.ent_coef * self.ent_decay, self.ent_min)

        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.dones.clear(); self.values.clear()

    def _compute_gae(self):
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float64)
        returns = np.zeros(T, dtype=np.float64)

        # 奖励归一化：除以运行时标准差，稳定训练
        rewards_arr = np.array(self.rewards)
        reward_std = np.sqrt(self.reward_norm.var + 1e-8)
        scaled_rewards = rewards_arr / max(float(reward_std[0]), 1.0)

        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            delta = (scaled_rewards[t]
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
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'ent_coef': self.ent_coef,
            'state_norm_mean': self.state_norm.mean,
            'state_norm_var': self.state_norm.var,
            'state_norm_count': self.state_norm.count,
            'reward_norm_mean': self.reward_norm.mean,
            'reward_norm_var': self.reward_norm.var,
            'reward_norm_count': self.reward_norm.count,
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if 'ent_coef' in ckpt:
            self.ent_coef = ckpt['ent_coef']
        if 'state_norm_mean' in ckpt:
            self.state_norm.mean = ckpt['state_norm_mean']
            self.state_norm.var = ckpt['state_norm_var']
            self.state_norm.count = ckpt['state_norm_count']
        if 'reward_norm_mean' in ckpt:
            self.reward_norm.mean = ckpt['reward_norm_mean']
            self.reward_norm.var = ckpt['reward_norm_var']
            self.reward_norm.count = ckpt['reward_norm_count']
