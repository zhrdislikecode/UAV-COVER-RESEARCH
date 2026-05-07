"""
贪婪触发：仅 step=0 用匈牙利做一次分配，之后不再做宏观切换。
"""
import numpy as np
from experiment.hungarian import HungarianAssigner
from experiment.algorithm.trigger import (
    get_all_cluster_scores,
    calculate_uav_to_cluster_distances,
    assign_uavs_to_clusters,
    deploy_uavs_at_trigger_step,
)


def try_trigger_deployment_greedy(env, uavs, step, deploy_idx, config, verbose=False):
    """step=0 匈牙利分配一次，之后永不触发"""
    if step != 0 or deploy_idx > 0:
        return deploy_idx, False

    cluster_centers = np.array([c.center for c in env.clusters])
    distance_matrix = calculate_uav_to_cluster_distances(uavs, cluster_centers)
    distance_benefit = 1 - distance_matrix / distance_matrix.max(axis=1, keepdims=True)
    score_matrix = get_all_cluster_scores(env)

    hungarian = HungarianAssigner(
        distance_benefit, score_matrix, step,
        config.step_change, config.weight, config.threshold)
    assign_vector = hungarian.assign()

    if verbose:
        print(f"\n[Greedy] Step=0, initial assignment: "
              + ", ".join(f"UAV{i}→C{assign_vector[i]}" for i in range(len(uavs))))

    assign_uavs_to_clusters(env, uavs, assign_vector)
    for u in uavs:
        u.follow_cluster_list.append(u.follow_cluster.id)
    deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx)

    for c in env.clusters:
        if not c.is_selected:
            c.score += 1

    return deploy_idx, True
