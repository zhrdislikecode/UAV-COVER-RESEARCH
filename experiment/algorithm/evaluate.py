import numpy as np
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.dqn import DQNAgent
from experiment.ppo import make_ppo_state
from experiment.ddpg import make_ddpg_state
from experiment.utils import calculate_jain_index, plot_uav_trajectory_3d
from experiment.algorithm.trigger import try_trigger_deployment as _hungarian_trigger


def _get_trigger_fn(macro_scheduler='hungarian'):
    if macro_scheduler == 'hungarian':
        return _hungarian_trigger
    elif macro_scheduler == 'macro_ddqn':
        from experiment.algorithm.trigger_macro_ddqn import \
            try_trigger_deployment_macro_ddqn
        return try_trigger_deployment_macro_ddqn
    elif macro_scheduler == 'greedy':
        from experiment.algorithm.trigger_greedy import \
            try_trigger_deployment_greedy
        return try_trigger_deployment_greedy
    else:
        raise ValueError(f"Unknown macro_scheduler: {macro_scheduler}")


def _create_agent(agent_type, dqn_config, model_path):
    """根据类型创建并加载 agent"""
    if agent_type == 'ppo':
        from experiment.ppo import PPOAgent
        agent = PPOAgent(state_dim=dqn_config.state_dim,
                         action_dim=dqn_config.action_dim)
    elif agent_type == 'ddpg':
        from experiment.ddpg import DDPGAgent
        agent = DDPGAgent(state_dim=dqn_config.state_dim)
    else:
        agent = DQNAgent(state_dim=dqn_config.state_dim,
                         action_dim=dqn_config.action_dim)
    agent.load_model(model_path)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0
    return agent


def run_evaluation(env_config, train_config, dqn_config,
                   model_path="models/drl_tpwsp_model.pth",
                   agent_type='dqn', randomize=True, macro_scheduler='hungarian'):
    """评估：随机化环境，加载模型，运行评估

    Args:
        agent_type: 'dqn' | 'ppo' | 'ddpg'
        macro_scheduler: 'hungarian' | 'gcn' | 'macro_ddqn'
    """
    trigger_fn = _get_trigger_fn(macro_scheduler)
    env = Environment(
        slot=env_config.slot, cluster_num=env_config.cluster_num,
        scene_size=env_config.scene_size, cluster_radius=env_config.cluster_radius,
        users_per_cluster=env_config.users_per_cluster,
    )
    if randomize:
        env.randomize()
        print(f"Environment randomized: {env_config.cluster_num} clusters, "
              f"{len(env.interest_points)} POIs")

    uavs = [UAV(uav_id=i + 1, slot=env_config.slot, multiple=env_config.uav_fly_multiple)
            for i in range(env_config.uav_num)]

    agent = _create_agent(agent_type, dqn_config, model_path)
    print(f"Model loaded ({agent_type}): {model_path}")

    continuous = (agent_type == 'ddpg')
    steps = train_config.steps
    uav_position = np.zeros((steps, len(uavs), 2), dtype=np.float64)

    deploy_idx = 0
    total_covered, total_com, total_step = 0, 0, 0
    env.reset()
    for uav in uavs:
        uav.position[:] = [0, 0]
        uav.current_battery_capacity = uav.total_battery_capacity
        uav.follow_cluster = None
        uav.follow_cluster_list = []
        uav.cluster_coverage_time = 0
        uav.last_cluster_id = None
        uav.macro_fly_time = 0

    # 记录轨迹用于可视化
    cluster_traj = np.zeros((steps, len(env.clusters), 2), dtype=np.float64)

    for step in range(steps):
        env.step(True)  # 在线轨迹（含扰动）

        # 记录集群轨迹
        for ci, c in enumerate(env.clusters):
            cluster_traj[step, ci] = c.center

        deploy_idx, _ = trigger_fn(
            env, uavs, step, deploy_idx, train_config
        )

        total_step += 1
        for i, uav in enumerate(uavs):
            uav_position[step][i] = (uav.follow_cluster.center
                                     if uav.follow_cluster else uav.position)
            if hasattr(uav, 'steps_since_last_switch'):
                uav.steps_since_last_switch += 1
            if uav.macro_fly_time > 0:
                uav.macro_fly_time -= 1
                continue
            if uav.current_battery_capacity <= 0:
                continue
            if agent_type == 'ppo':
                state = make_ppo_state(uav)
            elif agent_type == 'ddpg':
                state = make_ddpg_state(uav)
            else:
                state = uav.get_state()
            action = agent.choose_action(state)

            if continuous:
                next_state, reward, done, covered, com = uav.step_continuous(action)
            else:
                next_state, reward, done, covered, com = uav.step(action)

            if uav.current_battery_capacity <= 0:
                uav.is_consume_energy = True
                continue
            total_covered += covered
            total_com += com

    jain_index = calculate_jain_index(env, total_step)
    plot_uav_trajectory_3d(uav_position, cluster_traj)

    print(f"Evaluation result: value={jain_index * total_com:.4f}, "
          f"total_com={total_com:.4f}, jain_index={jain_index:.4f}")
    return jain_index * total_com, total_com, jain_index
