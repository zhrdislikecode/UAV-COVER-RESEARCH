from dataclasses import dataclass


@dataclass
class EnvConfig:
    """环境相关配置"""
    scene_size: float = 15.0
    slot: int = 6
    cluster_num: int = 6
    cluster_radius: float = 0.3
    users_per_cluster: int = 30
    uav_num: int = 3
    uav_fly_multiple: int = 3


@dataclass
class DQNConfig:
    """DQN agent 相关配置"""
    state_dim: int = 6
    action_dim: int = 9
    lr: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99
    batch_size: int = 128
    memory_size: int = 10000


@dataclass
class TrainConfig:
    """训练流程相关配置"""
    episodes: int = 200
    steps: int = 300               # 1800 // slot
    all_user_num: int = 120
    cluster_num: int = 4
    weight: float = 0.5          # 匈牙利算法权重
    threshold: float = 1.0         # 匈牙利算法阈值
    target_update_interval: int = 30
    epsilon_decay_interval: int = 100
    train_freq_early: int = 1      # 前半段训练：每步训练
    train_freq_late: int = 5       # 后半段训练：每5步训练
    log_interval: int = 10         # 打印日志间隔
    verbose_trigger: bool = False   # 是否打印匈牙利/GCN 触发日志
    rerandomize_interval: int = 100  # 每隔 N 轮重新随机化环境

    @property
    def step_change(self):
        return self.steps / 10
