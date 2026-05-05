import numpy as np
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.dqn import DQNAgent, ReplayBuffer
from experiment.utils import generate_positions, calculate_jain_index, plot_uav_trajectory_3d
from experiment.algorithm.trigger import compute_trigger_steps, deploy_uavs_at_trigger_step


def run_evaluation(env_config, train_config, dqn_config,
                   model_path="models/drl_tpwsp_model.pth", randomize=True):
    """评估：创建随机化环境，加载模型并评估

    Args:
        env_config: EnvConfig 实例
        train_config: TrainConfig 实例
        dqn_config: DQNConfig 实例
        model_path: 训练好的模型路径
        randomize: 是否随机化环境
    """
    # 1. 创建环境并随机化
    env = Environment(
        slot=env_config.slot,
        cluster_num=env_config.cluster_num,
        scene_size=env_config.scene_size,
        cluster_radius=env_config.cluster_radius,
        users_per_cluster=env_config.users_per_cluster,
    )
    if randomize:
        env.randomize()
        print(f"Environment randomized: {env_config.cluster_num} clusters, "
              f"{len(env.interest_points)} POIs")

    # 2. 创建 UAV
    uavs = [UAV(uav_id=i + 1, slot=env_config.slot, multiple=env_config.uav_fly_multiple)
            for i in range(env_config.uav_num)]

    # 3. 加载通用模型
    agent = DQNAgent(state_dim=dqn_config.state_dim, action_dim=dqn_config.action_dim)
    agent.load_model(model_path)
    agent.epsilon = 0.0
    print(f"Model loaded from: {model_path}")

    # 4. 预生成确定性轨迹并计算触发步
    user_positions, cluster_positions, cluster_direction = \
        generate_positions(env, False, train_config.steps)
    trigger_steps = compute_trigger_steps(
        env, uavs, user_positions, cluster_positions, cluster_direction, train_config
    )
    print(f"Trigger steps: {trigger_steps}")

    # 5. 在线轨迹评估
    steps = train_config.steps
    uav_position = np.zeros((steps, len(uavs), 2), dtype=np.float64)
    user_positions_online, cluster_positions_online, cluster_direction_online = \
        generate_positions(env, True, steps)

    deploy_idx = 0
    total_covered, total_com, total_step = 0, 0, 0
    env.reset()
    for uav in uavs:
        uav.position[:] = [0, 0]
        uav.current_battery_capacity = uav.total_battery_capacity
        uav.follow_cluster = None
        uav.cluster_coverage_time = 0
        uav.last_cluster_id = None
        uav.macro_fly_time = 0

    for step in range(steps):
        env.set_positions_from_array(
            user_positions_online[step], cluster_positions_online[step],
            cluster_direction_online[step]
        )
        if step in trigger_steps:
            deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx)

        total_step += 1
        for i, uav in enumerate(uavs):
            uav_position[step][i] = uav.follow_cluster.center
            if uav.macro_fly_time > 0:
                uav.macro_fly_time -= 1
                continue
            if uav.current_battery_capacity <= 0:
                continue
            state = uav.get_state()
            action = agent.choose_action(state)
            action = uav.revise_direction(action)
            next_state, reward, done, covered, com = uav.step(action)
            if uav.current_battery_capacity <= 0:
                uav.is_consume_energy = True
                continue
            total_covered += covered
            total_com += com

    jain_index = calculate_jain_index(env, total_step)
    plot_uav_trajectory_3d(uav_position, cluster_positions_online)

    print(f"Evaluation result: value={jain_index * total_com:.4f}, "
          f"total_com={total_com:.4f}, jain_index={jain_index:.4f}")
    return jain_index * total_com, total_com, jain_index
