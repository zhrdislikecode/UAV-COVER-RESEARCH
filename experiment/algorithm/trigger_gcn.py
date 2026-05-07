"""
GCN-based 触发部署模块

使用训练好的 GCN 模型替代匈牙利算法进行 UAV-集群匹配。
触发条件仍使用匈牙利逻辑（何时触发），GCN 只替换分配决策（分配给谁）。

用法:
  # 替换一行 import 即可切换
  from experiment.algorithm.trigger_gcn import try_trigger_deployment_gcn as try_trigger

  # 或者直接调用
  from experiment.algorithm.trigger_gcn import try_trigger_deployment_gcn
  deploy_idx, triggered = try_trigger_deployment_gcn(env, uavs, step, deploy_idx, config)
"""
import numpy as np
import torch
from experiment.hungarian import HungarianAssigner
from experiment.algorithm.trigger import (
    get_all_cluster_scores,
    calculate_uav_to_cluster_distances,
    assign_uavs_to_clusters,
    deploy_uavs_at_trigger_step,
    _log_cluster_state,
)
from experiment.algorithm.gcn_matcher import GCNAssigner


# 全局 GCN 分配器缓存（避免每次触发都重新加载模型）
_gcn_assigner_cache = {}


def _get_gcn_assigner(model_path="models/gcn_matcher.pth"):
    """懒加载 GCN 分配器，缓存避免重复 IO"""
    if model_path not in _gcn_assigner_cache:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        hidden = ckpt.get("hidden", 32)
        _gcn_assigner_cache[model_path] = GCNAssigner(
            model_path, hidden=hidden)
    return _gcn_assigner_cache[model_path]


def _fmt_vec(x, y):
    return f"({x:5.1f},{y:5.1f})"


def _log_gcn_scores(scores, uavs, clusters, hungarian_assign, gcn_assign, alpha=None):
    """打印 GCN 评分矩阵、匈牙利约束后的分配、以及对比"""
    n_u, n_c = scores.shape

    # GCN argmax 分配（无约束，可能冲突）
    argmax_assign = scores.argmax(axis=1)

    header = "        " + "".join(f"  C{j}   " for j in range(n_c))
    print(f"  GCN Score Matrix (higher = better):")
    print(header)
    for i in range(n_u):
        row = f"  UAV{i}: "
        for j in range(n_c):
            # 标注：* = GCN+Hungarian, # = argmax, ! = both
            marker = " "
            if gcn_assign[i] == j and argmax_assign[i] == j:
                marker = "!"
            elif gcn_assign[i] == j:
                marker = "*"
            elif argmax_assign[i] == j:
                marker = "#"
            row += f"{scores[i, j]:7.3f}{marker} "
        uav_pos = uavs[i].position
        row += f"  (UAV at {_fmt_vec(uav_pos[0], uav_pos[1])})"
        print(row)

    print()
    print(f"  Legend: !=argmax&Hungarian  *=Hungarian-forced  #=argmax(no constraint)")

    # 各分配结果
    argmax_str = ", ".join(f"UAV{i}→C{argmax_assign[i]}" for i in range(n_u))
    gcn_str = ", ".join(f"UAV{i}→C{gcn_assign[i]}" for i in range(n_u))
    hun_str = ", ".join(f"UAV{i}→C{hungarian_assign[i]}" for i in range(n_u))
    print(f"  GCN argmax (unconstrained): {argmax_str}")
    print(f"  GCN+Hungarian (constrained): {gcn_str}")
    print(f"  Hungarian original:           {hun_str}")
    if alpha is not None:
        print(f"  GCN alpha (score vs distance weight): {alpha:.4f}")


def try_trigger_deployment_gcn(env, uavs, step, deploy_idx, config,
                                gcn_model_path="models/gcn_matcher.pth",
                                verbose=None):
    """使用 GCN+Hungarian 进行 UAV-集群分配（触发条件沿用匈牙利逻辑）

    GCN 提供偏好评分，匈牙利算法在此基础上强制一对一约束。

    Args:
        env: 环境对象
        uavs: UAV 列表
        step: 当前步数
        deploy_idx: 当前部署索引
        config: TrainConfig 实例
        gcn_model_path: 训练好的 GCN 模型路径
        verbose: True=打印宏观调控详细日志

    Returns:
        (new_deploy_idx, triggered)
    """
    if verbose is None:
        verbose = getattr(config, 'verbose_trigger', False)
    cluster_centers = np.array([c.center for c in env.clusters])
    distance_matrix = calculate_uav_to_cluster_distances(uavs, cluster_centers)
    distance_benefit = 1 - distance_matrix / distance_matrix.max(
        axis=1, keepdims=True)
    score_matrix = get_all_cluster_scores(env)
    if step != 0:
        score_matrix = score_matrix / score_matrix.max()

    # 触发条件仍用匈牙利逻辑（决定何时触发）
    hungarian = HungarianAssigner(
        distance_benefit, score_matrix, step, config.step_change,
        config.weight, config.threshold
    )

    triggered = False
    if hungarian.should_assign():
        # ★ GCN 评分 + 匈牙利约束 = 一对一最优分配
        gcn = _get_gcn_assigner(gcn_model_path)
        assign_vector = gcn.assign(uavs, env.clusters)  # 已内置 Hungarian 约束

        if verbose:
            n_triggered = deploy_idx + 1
            gcn_scores = gcn.get_scores(uavs, env.clusters)
            hungarian_assign = hungarian.assign()
            alpha_val = gcn.model.alpha.item()

            print(f"\n{'═'*70}")
            print(f"[GCN Trigger #{n_triggered}] Step={step}, "
                  f"DeployIdx={deploy_idx}→{n_triggered}")
            print(f"{'═'*70}")
            _log_cluster_state(env.clusters)
            print()
            _log_gcn_scores(gcn_scores, uavs, env.clusters,
                           hungarian_assign, assign_vector, alpha=alpha_val)

            # 与纯匈牙利对比
            match = (assign_vector == hungarian_assign).all()
            if match:
                print(f"  [OK] GCN+Hungarian agrees with pure Hungarian")
            else:
                diff_uavs = [i for i in range(len(uavs))
                            if assign_vector[i] != hungarian_assign[i]]
                diff_str = ", ".join(
                    f"UAV{i}: GCN+Hun→C{assign_vector[i]} vs Hun→C{hungarian_assign[i]}"
                    for i in diff_uavs)
                print(f"  [DIFF] GCN diverges from Hungarian: {diff_str}")
            print(f"{'═'*70}\n")

        assign_uavs_to_clusters(env, uavs, assign_vector)
        for u in uavs:
            u.follow_cluster_list.append(u.follow_cluster.id)
        deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx)
        triggered = True

    # 未选中集群分值累加
    for c in env.clusters:
        if not c.is_selected:
            c.score += 1

    return deploy_idx, triggered
