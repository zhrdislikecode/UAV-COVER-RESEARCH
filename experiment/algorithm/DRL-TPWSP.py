import numpy as np
from matplotlib import rcParams

from experiment.config import EnvConfig, DQNConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.dqn import DQNAgent, ReplayBuffer
from experiment.algorithm.train import run_training
from experiment.algorithm.evaluate import run_evaluation

np.random.seed(10)
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def train():
    """训练模式：每轮随机化环境 → 训练 → 保存模型"""
    env_config = EnvConfig()
    dqn_config = DQNConfig()
    train_config = TrainConfig()

    env = Environment(
        slot=env_config.slot,
        cluster_num=env_config.cluster_num,
        scene_size=env_config.scene_size,
        cluster_radius=env_config.cluster_radius,
        users_per_cluster=env_config.users_per_cluster,
    )

    uavs = [UAV(uav_id=i + 1, slot=env_config.slot, multiple=env_config.uav_fly_multiple)
            for i in range(env_config.uav_num)]

    shared_buffer = ReplayBuffer(capacity=dqn_config.memory_size)
    agent = DQNAgent(
        state_dim=dqn_config.state_dim, action_dim=dqn_config.action_dim,
        buffer=shared_buffer, lr=dqn_config.lr, gamma=dqn_config.gamma,
        epsilon_start=dqn_config.epsilon_start, epsilon_end=dqn_config.epsilon_end,
        epsilon_decay=dqn_config.epsilon_decay, batch_size=dqn_config.batch_size,
    )

    run_training(env, uavs, agent, train_config)


def evaluate():
    """评估模式：随机化环境 → 加载模型 → 评估"""
    env_config = EnvConfig()
    dqn_config = DQNConfig()
    train_config = TrainConfig()

    value, total_com, jain_index = run_evaluation(
        env_config, train_config, dqn_config,
        model_path="models/drl_tpwsp_model.pth", randomize=True
    )
    return value, total_com, jain_index


if __name__ == "__main__":
    # train()   # 需要训练时取消注释
    evaluate()
