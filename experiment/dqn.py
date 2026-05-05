import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 默认超参数
LR = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  经验回放池
# ============================================================
class ReplayBuffer:
    """共享经验回放池，支持多 UAV 共用"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


# ============================================================
#  DQN 网络
# ============================================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
#  Double DQN Agent
# ============================================================
class DQNAgent:
    """Double DQN Agent —— online 网络选动作，target 网络评估，减少过估计"""
    def __init__(self, state_dim: int, action_dim: int,
                 buffer: ReplayBuffer = None,
                 lr: float = LR, gamma: float = GAMMA,
                 epsilon_start: float = EPSILON_START,
                 epsilon_end: float = EPSILON_END,
                 epsilon_decay: float = EPSILON_DECAY,
                 batch_size: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_end

        self.q_net = DQN(state_dim, action_dim).to(device)
        self.target_q_net = DQN(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 支持外部共享 buffer
        self.buffer = buffer if buffer is not None else ReplayBuffer()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def store_transition(self, transition):
        self.buffer.push(transition)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        # Double DQN: online 网络选动作，target 网络评估 Q 值
        next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_q_net(next_states).gather(1, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.epsilon = checkpoint.get('epsilon', 0.0)
