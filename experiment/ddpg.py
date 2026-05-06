import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_ddpg_state(uav):
    """构建 DDPG 专用状态向量 (6维)：[dx, dy, cvx, cvy, dist, battery_ratio]

    使用 UAV 与集群的相对位置，平移不变，更适合连续控制。
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
#  DDPG 网络
# ============================================================
class DDPGActor(nn.Module):
    """策略网络：输出连续动作 (dx, dy) ∈ [-1, 1]"""
    def __init__(self, state_dim, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state)


class DDPGCritic(nn.Module):
    """Q 网络：输入 state + action，输出 Q 值"""
    def __init__(self, state_dim, action_dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
#  DDPG Agent（连续动作空间）
# ============================================================
class DDPGAgent:
    def __init__(self, state_dim, action_dim=2,
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=128,
                 memory_size=100000,
                 noise_std=0.3, noise_decay=0.98, noise_min=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min

        # 主网络
        self.actor = DDPGActor(state_dim, action_dim).to(device)
        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        # 目标网络
        self.actor_target = DDPGActor(state_dim, action_dim).to(device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放
        self.buffer = []
        self.memory_size = memory_size

    @staticmethod
    def _norm_state(state):
        """手工状态归一化，基于各维度的预期范围"""
        s = np.array(state, dtype=np.float32)
        s[0] /= 15.0   # dx ∈ [-15, 15]
        s[1] /= 15.0   # dy ∈ [-15, 15]
        s[2] /= 0.1    # cvx ~0.04
        s[3] /= 0.1    # cvy ~0.04
        s[4] /= 21.0   # dist ∈ [0, ~21]
        # s[5] = battery_ratio ∈ [0, 1]，无需缩放
        return s

    def choose_action(self, state):
        s = self._norm_state(state)
        state_t = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.noise_std, size=self.action_dim)
        action = action + noise
        action = np.clip(action, -1.0, 1.0)
        return action  # shape (2,), 值域 [-1, 1]

    def store_transition(self, transition):
        if len(self.buffer) >= self.memory_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def decay_noise(self):
        """每个 episode 结束时调用，逐步减少探索噪声"""
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_min)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(
            np.array([self._norm_state(s) for s in states])).to(device)
        actions_t = torch.FloatTensor(np.array(actions)).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(
            np.array([self._norm_state(s) for s in next_states])).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Critic 更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = rewards_t + (1 - dones_t) * self.gamma * \
                       self.critic_target(next_states_t, next_actions)
        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor 更新
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def update_target(self):
        pass  # DDPG 用软更新，不需要硬更新

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'noise_std': self.noise_std,
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if 'noise_std' in ckpt:
            self.noise_std = ckpt['noise_std']
