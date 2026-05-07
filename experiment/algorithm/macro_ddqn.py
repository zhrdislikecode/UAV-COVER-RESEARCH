"""
宏观调度 DDQN — 分布式 UAV-集群切换

每架 UAV 独立运行一个 DDQN 模型（同构共享参数），每 10 步决策：
  keep: 保持当前集群（继续服务）
  switch-to-top-N: 切换到匈牙利 benefit 评分最高的 N 个候选集群之一

State (15 维): top-3 候选 × 5 特征
训练: 在线 epsilon-greedy，边仿真边学
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#  常量
# ============================================================
STATE_DIM = 15     # top-3 × 5
ACTION_DIM = 4     # keep, top-1, top-2, top-3
TOP_K = 3
HUN_WEIGHT = 0.5   # 匈牙利权重（score vs distance）


# ============================================================
#  DDQN 网络
# ============================================================
class MacroQNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)


# ============================================================
#  状态构建（15 维：top-3 × 5 特征）
# ============================================================
def build_macro_state(uav, env, other_uavs, scene_size=15.0):
    """构建宏观调度状态向量 (15 维)

    候选评分 = 匈牙利 benefit: 0.5*score_norm + 0.5*dist_benefit
    取 top-3（排除已被兄弟覆盖的集群，当前集群强制保留）

    每个候选 5 维:
      [0] score_norm        — 集群等待时间（归一化）
      [1] dist_benefit      — 距离收益 = 1 - dist/max_dist_uav
      [2] is_current        — 0/1，是否当前服务的集群
      [3] covered_by_other  — 0/1，是否被其他 UAV 覆盖
      [4] density_norm      — 用户密度（归一化）
    """
    clusters = env.clusters
    uav_pos = uav.position[:2]
    n_clusters = len(clusters)

    # ── 收集各集群指标 ──
    dists = [float(np.linalg.norm(uav_pos - c.center[:2])) for c in clusters]
    max_dist = max(max(dists), 1e-6)
    max_score = max(max(c.score for c in clusters), 1.0)
    max_users = max(max(len(c.users) for c in clusters), 1)

    # 已被兄弟覆盖的集群
    covered_by_others = set()
    for other in other_uavs:
        if other.follow_cluster is not None and other.id != uav.id:
            covered_by_others.add(other.follow_cluster.id)

    cur_id = uav.follow_cluster.id if uav.follow_cluster is not None else -1

    # ── 候选评分 ──
    scored = []
    for j, c in enumerate(clusters):
        score_norm = c.score / max_score
        dist_benefit = 1.0 - dists[j] / max_dist
        density_norm = len(c.users) / max_users
        benefit = HUN_WEIGHT * score_norm + (1 - HUN_WEIGHT) * dist_benefit
        covered = 1.0 if j in covered_by_others else 0.0
        scored.append((j, benefit, score_norm, dist_benefit, covered, density_norm))

    # ── 选 top-3（当前集群强制保留）──
    candidates = []
    if cur_id >= 0:
        for item in scored:
            if item[0] == cur_id:
                candidates.append(item)
                break
    for item in scored:
        if item[0] not in [c[0] for c in candidates]:
            candidates.append(item)
    candidates.sort(key=lambda x: x[1], reverse=True)
    top3 = candidates[:TOP_K]

    # 补足到 3 个
    while len(top3) < TOP_K:
        top3.append((-1, -999., 0., 0., 0., 0.))

    # ── 构建 state ──
    state = np.zeros(STATE_DIM, dtype=np.float32)
    for k, (cid, benefit, sn, db, cov, den) in enumerate(top3):
        base = k * 5
        state[base + 0] = sn          # score_norm
        state[base + 1] = db          # dist_benefit
        state[base + 2] = 1.0 if cid == cur_id else 0.0  # is_current
        state[base + 3] = cov         # covered_by_other
        state[base + 4] = den         # density_norm

    return np.nan_to_num(state, nan=0.0), top3


# ============================================================
#  单 UAV 宏观飞行执行
# ============================================================
def execute_macro_switch(uav, target_cluster, env_slot=4.0):
    """执行单架 UAV 到目标集群的宏观切换"""
    move_dis = float(np.linalg.norm(target_cluster.center[:2] - uav.position[:2]))
    uav.macro_fly_time = int(move_dis * 100 / (20 * env_slot))
    uav.current_battery_capacity -= uav.macro_fly_time * uav.high_speed_energy
    uav.follow_cluster = target_cluster
    uav.position = np.array([target_cluster.center[0], target_cluster.center[1]],
                            dtype=np.float32)
    uav.steps_since_last_switch = 0


# ============================================================
#  DDQN Agent
# ============================================================
class MacroAgent:
    """宏观调度 DDQN Agent（Double DQN + 经验回放）"""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, batch_size=128, memory_size=50000,
                 target_update_interval=100, hidden=128, reward_clip=100.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.reward_clip = reward_clip

        self.q_net = MacroQNet(state_dim, action_dim, hidden).to(device)
        self.target_net = MacroQNet(state_dim, action_dim, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = []
        self.memory_size = memory_size
        self.train_step_count = 0

    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        r = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        if len(self.buffer) >= self.memory_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, r, next_state, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q
        target_q = torch.clamp(target_q, -self.reward_clip * 2, self.reward_clip * 2)

        q_values = self.q_net(states_t).gather(1, actions_t)
        loss = nn.MSELoss()(q_values, target_q)

        if torch.isnan(loss) or torch.isinf(loss):
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        with torch.no_grad():
            mean_q = q_values.mean().item()
            max_q = q_values.max().item()

        return {'loss': loss.item(), 'mean_q': mean_q, 'max_q': max_q,
                'target_q_mean': target_q.mean().item()}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def save_model(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.epsilon = ckpt.get('epsilon', self.epsilon_end)
