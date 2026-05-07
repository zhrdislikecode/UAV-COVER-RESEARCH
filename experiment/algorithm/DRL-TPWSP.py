import numpy as np
from matplotlib import rcParams

from experiment.config import EnvConfig, DQNConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.algorithm.train import (
    run_training_dqn, run_training_ppo, run_training_ddpg)
from experiment.algorithm.evaluate import run_evaluation

np.random.seed(11)
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

AGENT_TYPE = 'dqn'  # 'dqn' | 'ppo' | 'ddpg'


def _create_agent(agent_type, dqn_config):
    if agent_type == 'ppo':
        from experiment.ppo import PPOAgent
        return PPOAgent(state_dim=dqn_config.state_dim,
                        action_dim=dqn_config.action_dim)
    elif agent_type == 'ddpg':
        from experiment.ddpg import DDPGAgent
        return DDPGAgent(state_dim=dqn_config.state_dim)
    else:
        from experiment.dqn import DQNAgent, ReplayBuffer
        buffer = ReplayBuffer(capacity=dqn_config.memory_size)
        return DQNAgent(
            state_dim=dqn_config.state_dim, action_dim=dqn_config.action_dim,
            buffer=buffer, lr=dqn_config.lr, gamma=dqn_config.gamma,
            epsilon_start=dqn_config.epsilon_start, epsilon_end=dqn_config.epsilon_end,
            epsilon_decay=dqn_config.epsilon_decay, batch_size=dqn_config.batch_size,
        )


def train(agent_type=AGENT_TYPE, macro_scheduler='hungarian'):
    """训练模式：每轮随机化环境 → 训练 → 保存模型

    Args:
        agent_type: 'dqn' | 'ppo' | 'ddpg'
        macro_scheduler: 'hungarian' | 'gcn' | 'macro_ddqn'
    """
    env_config = EnvConfig()
    dqn_config = DQNConfig()
    train_config = TrainConfig()

    env = Environment(
        slot=env_config.slot, cluster_num=env_config.cluster_num,
        scene_size=env_config.scene_size, cluster_radius=env_config.cluster_radius,
        users_per_cluster=env_config.users_per_cluster,
    )
    uavs = [UAV(uav_id=i + 1, slot=env_config.slot, multiple=env_config.uav_fly_multiple)
            for i in range(env_config.uav_num)]

    agent = _create_agent(agent_type, dqn_config)
    print(f"Training with {agent_type.upper()} agent + {macro_scheduler} macro scheduler...")

    train_funcs = {'dqn': run_training_dqn, 'ppo': run_training_ppo,
                   'ddpg': run_training_ddpg}
    train_funcs[agent_type](env, uavs, agent, train_config, macro_scheduler=macro_scheduler)


def evaluate(agent_type=AGENT_TYPE, macro_scheduler='hungarian'):
    """评估模式：随机化环境 → 加载模型 → 评估

    Args:
        agent_type: 'dqn' | 'ppo' | 'ddpg'
        macro_scheduler: 'hungarian' | 'gcn' | 'macro_ddqn'
    """
    env_config = EnvConfig()
    dqn_config = DQNConfig()
    train_config = TrainConfig()

    model_names = {'dqn': 'drl_tpwsp_dqn.pth', 'ppo': 'drl_tpwsp_ppo.pth',
                   'ddpg': 'drl_tpwsp_ddpg.pth'}
    model_path = f"models/{model_names.get(agent_type, model_names['dqn'])}"

    value, total_com, jain_index = run_evaluation(
        env_config, train_config, dqn_config,
        model_path=model_path, randomize=True, agent_type=agent_type,
        macro_scheduler=macro_scheduler,
    )
    return value, total_com, jain_index


if __name__ == "__main__":
    # --- 训练（匈牙利调度）---
    # train('dqn', macro_scheduler='hungarian')
    # train('ppo', macro_scheduler='hungarian')
    # train('ddpg', macro_scheduler='hungarian')

    # --- 训练（GCN 调度，需先运行 train_gcn.py）---
    # train('ppo', macro_scheduler='gcn')

    # --- 训练（Macro DDQN 调度，需先运行 train_macro_ddqn.py）---
    # train('ppo', macro_scheduler='macro_ddqn')

    # --- 评估 ---
    evaluate('ppo', macro_scheduler='hungarian')



