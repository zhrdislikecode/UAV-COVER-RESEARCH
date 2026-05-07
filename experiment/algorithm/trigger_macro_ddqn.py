"""
Macro DDQN 触发部署模块（推理）

每 10 步决策，min_stay=20（切换后至少服务 20 步 = 跳过 1 个决策周期）。
首次分配用匈牙利算法保证互斥。
"""
import numpy as np
import torch
from experiment.algorithm.macro_ddqn import (
    MacroAgent, build_macro_state, execute_macro_switch,
    STATE_DIM, ACTION_DIM,
)
from experiment.hungarian import HungarianAssigner
from experiment.algorithm.trigger import (
    get_all_cluster_scores,
    calculate_uav_to_cluster_distances,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_INTERVAL = 10
INFER_MIN_STAY = 10

_agent_cache = {}


def _get_macro_agent(model_path="models/macro_ddqn.pth"):
    if model_path not in _agent_cache:
        agent = MacroAgent()
        agent.load_model(model_path)
        agent.epsilon = 0.0
        _agent_cache[model_path] = agent
    return _agent_cache[model_path]


def try_trigger_deployment_macro_ddqn(env, uavs, step, deploy_idx, config,
                                       model_path="models/macro_ddqn.pth",
                                       verbose=None):
    if verbose is None:
        verbose = getattr(config, 'verbose_trigger', False)
    agent = _get_macro_agent(model_path)
    triggered = False
    n_triggered = deploy_idx

    if step % INFER_INTERVAL != 0:
        return deploy_idx, False

    decisions = []

    # ── 首次分配：匈牙利保证互斥 ──
    unassigned = [uav for uav in uavs if uav.follow_cluster is None]
    if unassigned and step == 0:
        cluster_centers = np.array([c.center for c in env.clusters])
        dist_mat = calculate_uav_to_cluster_distances(uavs, cluster_centers)
        dist_benefit = 1 - dist_mat / dist_mat.max(axis=1, keepdims=True)
        score_mat = get_all_cluster_scores(env)
        hung = HungarianAssigner(dist_benefit, score_mat, step,
                                  config.step_change, config.weight, config.threshold)
        assign_vec = hung.assign()
        for i, uav in enumerate(uavs):
            cid = assign_vec[i]
            execute_macro_switch(uav, env.clusters[cid], env_slot=env.slot)
            triggered = True
            decisions.append((i, cid, 'init(hun)'))
        # 按顺序处理后续逻辑
        for c in env.clusters:
            c.is_selected = False
        for uav in uavs:
            if uav.follow_cluster is not None:
                uav.follow_cluster.is_selected = True
        if verbose and decisions:
            print(f"\n{'─'*60}")
            print(f"[MacroDDQN Trigger] Step={step}, Decisions:")
            for uav_idx, cluster, reason in decisions:
                print(f"  UAV{uav_idx}: → C{cluster} ({reason})")
            print(f"{'─'*60}\n")
        return (1, True)

    for i, uav in enumerate(uavs):
        if uav.macro_fly_time > 0:
            continue
        if uav.current_battery_capacity <= 0:
            continue

        if hasattr(uav, 'steps_since_last_switch'):
            uav.steps_since_last_switch += 1
        else:
            uav.steps_since_last_switch = 999

        in_stay = uav.steps_since_last_switch < INFER_MIN_STAY

        if in_stay:
            decisions.append((i, -1, f'stay({uav.steps_since_last_switch}/{INFER_MIN_STAY})'))
            continue

        # DDQN 决策
        other_uavs = [u for u in uavs if u.id != uav.id]
        state, top3 = build_macro_state(uav, env, other_uavs, scene_size=15.0)
        action = agent.choose_action(state, training=False)

        if action != 0:
            k = action - 1
            if k < len(top3) and top3[k][0] >= 0:
                target = env.clusters[top3[k][0]]
                if uav.follow_cluster != target:
                    execute_macro_switch(uav, target, env_slot=env.slot)
                    triggered = True
                    decisions.append((i, top3[k][0], f'sw{k+1}'))
                else:
                    decisions.append((i, -1, 'keep(same)'))
            else:
                decisions.append((i, -1, 'keep(invalid)'))
        else:
            decisions.append((i, -1, 'keep'))

    if triggered:
        n_triggered = deploy_idx + 1
        for uav in uavs:
            if uav.follow_cluster is not None:
                if not hasattr(uav, 'follow_cluster_list'):
                    uav.follow_cluster_list = []
                if not uav.follow_cluster_list or \
                   uav.follow_cluster_list[-1] != uav.follow_cluster.id:
                    uav.follow_cluster_list.append(uav.follow_cluster.id)

    if verbose and decisions:
        print(f"\n{'─'*60}")
        print(f"[MacroDDQN Trigger] Step={step}, Decisions:")
        for uav_idx, cluster, reason in decisions:
            if cluster >= 0:
                print(f"  UAV{uav_idx}: → C{cluster} ({reason})")
            else:
                print(f"  UAV{uav_idx}: {reason}")
        print(f"{'─'*60}\n")

    for c in env.clusters:
        c.is_selected = False
    for uav in uavs:
        if uav.follow_cluster is not None:
            uav.follow_cluster.is_selected = True

    return (n_triggered if triggered else deploy_idx), triggered
