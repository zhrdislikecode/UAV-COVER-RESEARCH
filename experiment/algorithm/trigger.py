import numpy as np
from experiment.hungarian import HungarianAssigner


def get_all_cluster_scores(env):
    return np.array([c.score for c in env.clusters])


def calculate_uav_to_cluster_distances(uavs, cluster_centers):
    distances = np.zeros((len(uavs), cluster_centers.shape[0]))
    uav_positions = np.array([u.get_position() for u in uavs])
    for i, up in enumerate(uav_positions):
        for j in range(cluster_centers.shape[0]):
            distances[i, j] = np.linalg.norm(up - cluster_centers[j])
    return distances


def assign_uavs_to_clusters(env, uavs, assignment_vector, cluster_centers):
    for c in env.clusters:
        c.is_selected = False
    for uav_idx, cluster_id in enumerate(assignment_vector):
        c = env.clusters[cluster_id]
        c.is_selected = True
        c.score = 0
        uavs[uav_idx].position = np.array(
            [cluster_centers[cluster_id][0], cluster_centers[cluster_id][1]])
        uavs[uav_idx].follow_cluster = c


def deploy_uavs_at_trigger_step(env, uavs, deploy_idx):
    for uav in uavs:
        target = env.clusters[uav.follow_cluster_list[deploy_idx]]
        move_dis = np.linalg.norm(target.center - uav.position)
        uav.macro_fly_time = int(move_dis * 100 / (20 * env.slot))
        uav.current_battery_capacity -= uav.macro_fly_time * uav.high_speed_energy
        uav.follow_cluster = target
        uav.position = np.array([target.center[0], target.center[1]])
    return deploy_idx + 1


def compute_trigger_steps(env, uavs, user_positions, cluster_positions,
                           cluster_direction, config):
    """用匈牙利算法预计算宏观调度触发步，同时填充每架 UAV 的 follow_cluster_list"""
    trigger_steps = []
    for step in range(config.steps):
        env.set_positions_from_array(
            user_positions[step], cluster_positions[step], cluster_direction[step]
        )
        cluster_centers = np.array([c.center for c in env.clusters])
        distance_matrix = calculate_uav_to_cluster_distances(uavs, cluster_centers)
        distance_matrix = 1 - distance_matrix / distance_matrix.max(axis=1, keepdims=True)
        score_matrix = get_all_cluster_scores(env)
        if step != 0:
            score_matrix = score_matrix / score_matrix.max()
        hungarian = HungarianAssigner(
            distance_matrix, score_matrix, step, config.step_change,
            config.weight, config.threshold
        )
        if hungarian.should_assign():
            trigger_steps.append(step)
            assign_vector = hungarian.assign()
            assign_uavs_to_clusters(env, uavs, assign_vector, cluster_centers)
            for uav in uavs:
                uav.follow_cluster_list.append(uav.follow_cluster.id)
        for cluster in env.clusters:
            if not cluster.is_selected:
                cluster.score += 1
    return trigger_steps


def try_trigger_deployment(env, uavs, step, deploy_idx, config):
    """在线检查匈牙利触发条件，满足则分配并部署

    Returns: (new_deploy_idx, triggered)
    """
    cluster_centers = np.array([c.center for c in env.clusters])
    distance_matrix = calculate_uav_to_cluster_distances(uavs, cluster_centers)
    distance_matrix = 1 - distance_matrix / distance_matrix.max(axis=1, keepdims=True)
    score_matrix = get_all_cluster_scores(env)
    if step != 0:
        score_matrix = score_matrix / score_matrix.max()

    hungarian = HungarianAssigner(
        distance_matrix, score_matrix, step, config.step_change,
        config.weight, config.threshold
    )

    triggered = False
    if hungarian.should_assign():
        assign_vector = hungarian.assign()
        assign_uavs_to_clusters(env, uavs, assign_vector, cluster_centers)
        for u in uavs:
            u.follow_cluster_list.append(u.follow_cluster.id)
        deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx)
        triggered = True

    # 未选中集群分值累加
    for c in env.clusters:
        if not c.is_selected:
            c.score += 1

    return deploy_idx, triggered
