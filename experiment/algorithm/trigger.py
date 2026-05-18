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


def assign_uavs_to_clusters(env, uavs, assignment_vector):
    for c in env.clusters:
        c.is_selected = False
    for uav_idx, cluster_id in enumerate(assignment_vector):
        c = env.clusters[cluster_id]
        c.is_selected = True
        c.score = 0
        uavs[uav_idx].position = np.array([c.center[0], c.center[1]])
        uavs[uav_idx].follow_cluster = c


def deploy_uavs_at_trigger_step(env, uavs, deploy_idx, training=False):
    for uav in uavs:
        target = env.clusters[uav.follow_cluster_list[deploy_idx]]
        move_dis = np.linalg.norm(target.center - uav.position)
        if np.isnan(move_dis) or np.isinf(move_dis):
            move_dis = 0.0
        if training:
            uav.macro_fly_time = 0
        else:
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
            score_matrix = score_matrix / max(score_matrix.max(), 1.0)
        hungarian = HungarianAssigner(
            distance_matrix, score_matrix, step, config.step_change,
            config.weight, config.threshold
        )
        if hungarian.should_assign():
            trigger_steps.append(step)
            assign_vector = hungarian.assign()
            assign_uavs_to_clusters(env, uavs, assign_vector)
            for uav in uavs:
                uav.follow_cluster_list.append(uav.follow_cluster.id)
        for cluster in env.clusters:
            if not cluster.is_selected:
                cluster.score += 1
    return trigger_steps


# ============================================================
#  日志格式化
# ============================================================
def _fmt_vec(x, y):
    return f"({x:5.1f},{y:5.1f})"


def _log_cluster_state(clusters):
    """打印集群状态"""
    print("  Clusters:")
    for c in clusters:
        sel = "← SELECTED" if c.is_selected else ""
        dir_x = c.direction[0] if c.direction is not None else 0.0
        dir_y = c.direction[1] if c.direction is not None else 0.0
        print(f"    C{c.id}: pos={_fmt_vec(c.center[0], c.center[1])}, "
              f"score={c.score:5.1f}, dir={_fmt_vec(dir_x, dir_y)}  {sel}")


def _log_benefit_matrix(distance_benefit, score_vec, weight, uavs, clusters, assignment):
    """打印匈牙利 benefit 矩阵和分配结果"""
    n_u, n_c = distance_benefit.shape
    benefit = weight * score_vec + (1 - weight) * distance_benefit

    # 表头
    header = "        " + "".join(f"  C{j}   " for j in range(n_c))
    print(f"  Benefit Matrix (w={weight}):")
    print(header)
    # 每行
    for i in range(n_u):
        row = f"  UAV{i}: "
        for j in range(n_c):
            marker = "*" if assignment[i] == j else " "
            row += f"{benefit[i, j]:6.3f}{marker} "
        uav_pos = uavs[i].position
        row += f"  (UAV at {_fmt_vec(uav_pos[0], uav_pos[1])})"
        print(row)

    # 分配结果
    assign_str = ", ".join(f"UAV{i}→C{assignment[i]}" for i in range(n_u))
    print(f"  Assignment: {assign_str}")


# ============================================================
#  在线触发部署
# ============================================================
def try_trigger_deployment(env, uavs, step, deploy_idx, config, verbose=None,
                            training=False):
    """在线检查匈牙利触发条件，满足则分配并部署

    Args:
        verbose: True=打印宏观调控详细日志
        training: True=瞬移不扣能, False=计算飞行距离和能耗

    Returns: (new_deploy_idx, triggered)
    """
    if verbose is None:
        verbose = getattr(config, 'verbose_trigger', False)
    cluster_centers = np.array([c.center for c in env.clusters])
    distance_matrix = calculate_uav_to_cluster_distances(uavs, cluster_centers)
    distance_benefit = 1 - distance_matrix / distance_matrix.max(axis=1, keepdims=True)
    score_matrix = get_all_cluster_scores(env)
    if step != 0:
        score_matrix = score_matrix / max(score_matrix.max(), 1.0)

    hungarian = HungarianAssigner(
        distance_benefit, score_matrix, step, config.step_change,
        config.weight, config.threshold
    )

    triggered = False
    if hungarian.should_assign():
        assign_vector = hungarian.assign()

        if verbose:
            n_triggered = deploy_idx + 1
            print(f"\n{'═'*70}")
            print(f"[Hungarian Trigger #{n_triggered}] Step={step}, "
                  f"DeployIdx={deploy_idx}→{n_triggered}")
            print(f"{'═'*70}")
            _log_cluster_state(env.clusters)
            print()
            _log_benefit_matrix(distance_benefit, score_matrix, config.weight,
                               uavs, env.clusters, assign_vector)
            print(f"{'═'*70}\n")

        assign_uavs_to_clusters(env, uavs, assign_vector)
        for u in uavs:
            u.follow_cluster_list.append(u.follow_cluster.id)
        deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx,
                                                  training=training)
        triggered = True

    # 未选中集群分值累加
    for c in env.clusters:
        if not c.is_selected:
            c.score += 1

    return deploy_idx, triggered
