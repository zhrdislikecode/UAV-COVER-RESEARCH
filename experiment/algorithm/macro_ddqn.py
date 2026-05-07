"""
宏观调度 DDQN — 分布式 UAV-集群切换

每架 UAV 独立运行一个 DDQN 模型（同构共享参数），在决策步决定：
  keep: 保持当前集群
  switch-to-top-N: 切换到评分最高的 N 个候选集群之一

训练与现有 RL 完全解耦，不修改 trigger.py / train.py。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#  常量
# ============================================================
STATE_DIM = 33  # 5(self) + 3(sibling) + 1(same_count) + 3*8(top3)
ACTION_DIM = 4          # keep, top-1, top-2, top-3
TOP_K = 3

# 候选评分权重
W_WAIT = 0.4
W_DIST = 0.4
W_DENSITY = 0.2


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
#  状态构建
# ============================================================
def build_macro_state(uav, env, other_uavs,
                      decision_interval=30, min_switch_interval=60,
                      scene_size=15.0, max_vel=0.1):
    """为单架 UAV 构建宏观调度状态向量 (32 维)

    Returns:
        state: np.ndarray shape [STATE_DIM]
        candidates: list of (cluster_id, score)  top-3 候选集群
    """
    clusters = env.clusters
    n_clusters = len(clusters)
    n_uavs = len(other_uavs) + 1
    uav_pos = uav.position[:2]

    # ── 候选评分（排除已被兄弟覆盖的集群）──
    covered_by_others = set()
    for other in other_uavs:
        if other.follow_cluster is not None and other.id != uav.id:
            covered_by_others.add(other.follow_cluster.id)

    # 收集各集群的归一化指标
    all_dists = []
    scores_raw = []
    for j, c in enumerate(clusters):
        d = float(np.linalg.norm(uav_pos - c.center[:2]))
        all_dists.append(d)
    max_dist = max(max(all_dists), 1e-6)
    max_score = max(max(c.score for c in clusters), 1.0)
    max_users = max(max(len(c.users) for c in clusters), 1)

    for j, c in enumerate(clusters):
        d = all_dists[j]
        wait_norm = c.score / max_score
        dist_norm = d / max_dist
        density_norm = len(c.users) / max_users
        score = W_WAIT * wait_norm + W_DIST * (1.0 - dist_norm) + W_DENSITY * density_norm
        scores_raw.append((j, score, d, wait_norm, dist_norm, density_norm))

    # 排除已被覆盖 vs 自己当前集群
    candidates = []
    if uav.follow_cluster is not None:
        cur_id = uav.follow_cluster.id
        # 当前集群一定在候选里
        for j, s, d, wn, dn, den in scores_raw:
            if j == cur_id:
                candidates.append((j, s, d, wn, dn, den))
                break

    for j, s, d, wn, dn, den in scores_raw:
        if j not in covered_by_others and not any(c[0] == j for c in candidates):
            candidates.append((j, s, d, wn, dn, den))

    # 按分数排序，取 top-3
    candidates.sort(key=lambda x: x[1], reverse=True)
    top3 = candidates[:TOP_K]

    # 补足到 3 个（集群数可能不够）
    while len(top3) < TOP_K:
        # 填充虚拟候选
        top3.append((-1, -999.0, 0.0, 0.0, 0.0, 0.0))

    # ── 构建状态 ──
    state = np.zeros(STATE_DIM, dtype=np.float32)

    # 自身 (5)
    state[0] = uav.position[0] / scene_size
    state[1] = uav.position[1] / scene_size
    state[2] = uav.current_battery_capacity / max(uav.total_battery_capacity, 1e-6)
    state[3] = (uav.follow_cluster.id / 10.0
                if uav.follow_cluster is not None else -0.1)
    state[4] = min(uav.steps_since_last_switch / max(min_switch_interval, 1), 1.0) \
        if hasattr(uav, 'steps_since_last_switch') else 0.0

    # 兄弟 UAV 分配 (3)
    idx = 5
    for other in other_uavs:
        if other.id != uav.id:
            state[idx] = (other.follow_cluster.id / 10.0
                          if other.follow_cluster is not None else -0.1)
            idx += 1
            if idx >= 8:
                break
    # 与自己同集群的 UAV 数
    same_count = 0
    if uav.follow_cluster is not None:
        for other in other_uavs:
            if (other.id != uav.id and other.follow_cluster is not None
                    and other.follow_cluster.id == uav.follow_cluster.id):
                same_count += 1
    state[8] = same_count / max(n_uavs, 1)

    # Top-3 候选集群 × 8 = 24
    base = 9
    for k, (cid, score, dist, wn, dn, den) in enumerate(top3):
        if cid >= 0:
            c = clusters[cid]
            state[base + k*8 + 0] = (c.center[0] - uav_pos[0]) / scene_size  # rel_x
            state[base + k*8 + 1] = (c.center[1] - uav_pos[1]) / scene_size  # rel_y
            dir_x = c.direction[0] if c.direction is not None else 0.0
            dir_y = c.direction[1] if c.direction is not None else 0.0
            state[base + k*8 + 2] = dir_x / max_vel
            state[base + k*8 + 3] = dir_y / max_vel
            state[base + k*8 + 4] = wn       # wait_norm
            state[base + k*8 + 5] = dn       # dist_norm
            state[base + k*8 + 6] = den      # density_norm
            state[base + k*8 + 7] = 1.0 if (uav.follow_cluster is not None
                                           and uav.follow_cluster.id == cid) else 0.0
        # 虚拟候选保持 0

    return state, top3


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
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, batch_size=128, memory_size=50000,
                 target_update_interval=100, hidden=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval

        self.q_net = MacroQNet(state_dim, action_dim, hidden).to(device)
        self.target_net = MacroQNet(state_dim, action_dim, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = []
        self.memory_size = memory_size
        self.train_step_count = 0

    def choose_action(self, state, training=True):
        """epsilon-greedy 选动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.memory_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

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

        # Double DQN
        q_values = self.q_net(states_t).gather(1, actions_t)
        next_actions = self.q_net(next_states_t).argmax(1).unsqueeze(1)
        next_q = self.target_net(next_states_t).gather(1, next_actions)
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        return loss.item()

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
