import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  PPO 网络
# ============================================================
class Actor(nn.Module):
    """策略网络：输出 logits"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)  # logits, 不做 softmax


class Critic(nn.Module):
    """价值网络"""
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state)


# ============================================================
#  PPO Agent
# ============================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=1e-4, gamma=0.99, clip_eps=0.2,
                 gae_lambda=0.95, update_epochs=8, batch_size=128,
                 horizon=1024, ent_coef=0.01):
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

        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

        self._last_log_prob = None
        self._last_value = None

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

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.log_probs.append(self._last_log_prob)
        self.values.append(self._last_value)

    def train(self):
        if len(self.states) < self.horizon:
            return

        returns, advantages = self._compute_gae()

        states_t = torch.FloatTensor(np.array(self.states)).to(device)
        actions_t = torch.LongTensor(self.actions).to(device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)

        # 优势归一化
        adv_mean, adv_std = advantages_t.mean(), advantages_t.std()
        if adv_std > 1e-8:
            advantages_t = (advantages_t - adv_mean) / adv_std

        dataset_size = len(self.states)
        for _ in range(self.update_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start:start + self.batch_size]
                s = states_t[idx]
                a = actions_t[idx]
                old_lp = old_log_probs_t[idx]
                ret = returns_t[idx]
                adv = advantages_t[idx]

                # Actor loss
                logits = self.actor(s)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                # clamp log-diff 防止 ratio 溢出
                log_ratio = new_lp - old_lp
                ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                # Critic loss
                values = self.critic(s).squeeze(-1)
                critic_loss = nn.MSELoss()(values, ret)

                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 1.0
                )
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
            delta = (self.rewards[t] + self.gamma * next_value * (1.0 - self.dones[t])
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]
        return returns, advantages

    def update_target(self):
        pass

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
