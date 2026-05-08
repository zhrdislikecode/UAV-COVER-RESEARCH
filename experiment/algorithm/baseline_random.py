"""
随机游走基线：大步随机移动 + 每 20 步随机拉回到不同集群

- 微观: 大步长 (×10) 随机移动，覆盖范围大
- 宏观: 每 n 步随机选 n_uav 个不同集群，瞬移回去
- 不依赖任何 RL 模型
"""
import math
import numpy as np
from experiment.config import EnvConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV, COVERAGE_RADIUS, UAV_HEIGHT, compute_comm_rate
from experiment.utils import calculate_jain_index, plot_uav_trajectory_3d


def run_random_baseline(randomize=True, seed=None):
    """随机游走基线

    每步: 从 9 个离散动作中随机选一个，大步移动 (×10)
    每 n 步: 随机分配到 n_uav 个不同集群，瞬移到集群中心

    Returns:
        value, total_com, jain_index
    """
    if seed is not None:
        np.random.seed(seed)

    env_config = EnvConfig()
    train_config = TrainConfig()

    env = Environment(
        slot=env_config.slot, cluster_num=env_config.cluster_num,
        scene_size=env_config.scene_size, cluster_radius=env_config.cluster_radius,
        users_per_cluster=env_config.users_per_cluster,
    )
    uavs = [UAV(uav_id=i + 1, slot=env_config.slot,
                multiple=env_config.uav_fly_multiple)
            for i in range(env_config.uav_num)]

    if randomize:
        env.randomize()

    n_uav = len(uavs)
    n_cluster = len(env.clusters)
    steps = train_config.steps
    PULL_INTERVAL = 100  # 每 n 步拉回一次

    uav_position = np.zeros((steps, n_uav, 2), dtype=np.float64)
    cluster_traj = np.zeros((steps, n_cluster, 2), dtype=np.float64)

    total_com, total_step = 0, 0
    env.reset()
    for uav in uavs:
        uav.position[:] = [0, 0]
        uav.current_battery_capacity = uav.total_battery_capacity
        uav.follow_cluster = None
        uav.macro_fly_time = 0

    for step in range(steps):
        env.step(True)

        for ci, c in enumerate(env.clusters):
            cluster_traj[step, ci] = c.center

        # ── 每 n 步：随机拉回到不同集群 ──
        if step % PULL_INTERVAL == 0:
            targets = np.random.choice(n_cluster, size=min(n_uav, n_cluster),
                                       replace=False)
            for i, uav in enumerate(uavs):
                cid = targets[i % len(targets)]
                uav.follow_cluster = env.clusters[cid]
                uav.position = np.array(
                    [env.clusters[cid].center[0],
                     env.clusters[cid].center[1]], dtype=np.float32)

        total_step += n_uav
        for i, uav in enumerate(uavs):
            uav_position[step][i] = uav.position.copy()
            if uav.current_battery_capacity <= 0:
                continue

            # ── 微观：大步随机移动 ──
            action = np.random.randint(0, len(uav.moves))
            # 扣能（和旧逻辑一致）
            if action == 0:
                uav.current_battery_capacity -= uav.hover_energy
            elif action <= 4:
                uav.current_battery_capacity -= uav.low_speed_energy
            else:
                uav.current_battery_capacity -= uav.mid_speed_energy

            # 大步长 ×10
            uav.position[:2] += uav.moves[action] * 10
            uav.position[:2] = np.clip(uav.position[:2], 0, env_config.scene_size)

            # 计算通信质量
            com = 0.0
            uav_pos_2d = uav.position[:2]
            for cluster in env.clusters:
                for user in cluster.users:
                    d = float(np.linalg.norm(uav_pos_2d - user.position[:2]))
                    if d <= COVERAGE_RADIUS:
                        user.cover_num += 1
                        dis_3d = math.sqrt((d * 100) ** 2 + UAV_HEIGHT ** 2)
                        com += compute_comm_rate(UAV_HEIGHT, d,
                                                 num_users=len(cluster.users))
            com = com / 40.0 * (uav.slot / 4.0)

            if uav.current_battery_capacity <= 0:
                uav.is_consume_energy = True
                continue
            total_com += com

    jain_index = calculate_jain_index(env, total_step)
    plot_uav_trajectory_3d(uav_position, cluster_traj, pois=env.interest_points)

    value = jain_index * total_com
    print(f"Random baseline: value={value:.4f}, total_com={total_com:.4f}, "
          f"jain_index={jain_index:.4f}")
    return value, total_com, jain_index


if __name__ == "__main__":
    run_random_baseline(seed=None)
