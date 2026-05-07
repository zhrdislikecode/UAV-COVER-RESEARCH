"""
宏观调度 DDQN — 在线训练脚本

每 10 步一个决策周期，min_stay=20（切换后至少服务 20 步）。
在线 epsilon-greedy：边仿真边训练，epsilon 从 1.0 衰减到 0.05。
奖励 = 匈牙利 benefit + 覆盖累计 - 切换惩罚。
"""
import os
import numpy as np
import torch

from experiment.config import EnvConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.algorithm.macro_ddqn import (
    MacroAgent, build_macro_state, execute_macro_switch,
    STATE_DIM, ACTION_DIM, TOP_K, HUN_WEIGHT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

# ============================================================
#  超参数
# ============================================================
DECISION_INTERVAL = 10    # 每 10 步做一次宏观决策
MIN_STAY = 20             # 切换后最少停留步数（20 = 2 个决策周期）
SWITCH_COST = -5.0        # 切换惩罚


def _reset_uavs(uavs):
    for uav in uavs:
        uav.position[:] = [0, 0]
        uav.current_battery_capacity = uav.total_battery_capacity
        uav.follow_cluster = None
        uav.follow_cluster_list = []
        uav.cluster_coverage_time = 0
        uav.last_cluster_id = None
        uav.macro_fly_time = 0
        uav.is_consume_energy = False
        uav.steps_since_last_switch = 999


def _in_min_stay(uav):
    if not hasattr(uav, 'steps_since_last_switch'):
        uav.steps_since_last_switch = 999
    return uav.steps_since_last_switch < MIN_STAY


def _track_cluster_action(uav):
    """返回追踪集群中心的慢速动作"""
    if uav.follow_cluster is None:
        return 0
    direction = uav.follow_cluster.center[:2] - uav.position[:2]
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return 0
    angle = np.arctan2(direction[1], direction[0])
    if -np.pi/4 <= angle < np.pi/4:       return 3   # east
    elif np.pi/4 <= angle < 3*np.pi/4:    return 1   # north
    elif -3*np.pi/4 <= angle < -np.pi/4:  return 2   # south
    else:                                  return 4   # west


def _compute_hun_reward(uav, clusters, max_score):
    """计算当前 UAV-集群分配的匈牙利 benefit，缩放到 [-10, 30]"""
    if uav.follow_cluster is None:
        return -10.0
    uav_pos = uav.position[:2]
    c = uav.follow_cluster
    d = float(np.linalg.norm(uav_pos - c.center[:2]))
    max_d = max(float(np.linalg.norm(uav_pos - cl.center[:2])) for cl in clusters)
    dist_benefit = 1.0 - d / max(max_d, 1e-6)
    score_norm = c.score / max(max_score, 1.0)
    benefit = HUN_WEIGHT * score_norm + (1 - HUN_WEIGHT) * dist_benefit
    return benefit * 40.0 - 10.0


# ============================================================
#  在线训练
# ============================================================
def train_online(num_episodes=500, model_save_path="models/macro_ddqn.pth"):
    """在线训练：边仿真边学，epsilon 逐 episode 衰减"""
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

    agent = MacroAgent()
    n_uav = len(uavs)
    n_clusters = len(env.clusters)

    print(f"[在线训练] {num_episodes} episodes")
    print(f"  interval={DECISION_INTERVAL}, min_stay={MIN_STAY}, "
          f"UAV={n_uav}, Clusters={n_clusters}, State={STATE_DIM}, Action={ACTION_DIM}")
    print(f"{'Ep':>5s}  {'eps':>6s}  {'loss':>8s}  {'mean_q':>8s}  "
          f"{'avg_r':>8s}  {'switch':>7s}  {'actions':>s}")

    best_avg_r = -float('inf')

    for episode in range(1, num_episodes + 1):
        env.randomize()
        env.reset()
        _reset_uavs(uavs)

        # 集群分值
        cluster_score = [0] * n_clusters
        # pending[i] = (prev_state, prev_action, reward_acc)
        pending = [None] * n_uav
        ep_rewards = []
        ep_switches = 0
        ep_decisions = 0
        ep_losses = []
        action_counts = [0] * ACTION_DIM

        for step in range(train_config.steps):
            env.step(False)

            # 更新集群分值
            for j, c in enumerate(env.clusters):
                if not c.is_selected:
                    cluster_score[j] += 1
                else:
                    cluster_score[j] = max(0, cluster_score[j] - 1)
                c.score = cluster_score[j]

            max_score_ep = max(max(c.score for c in env.clusters), 1.0)

            for i, uav in enumerate(uavs):
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue

                if hasattr(uav, 'steps_since_last_switch'):
                    uav.steps_since_last_switch += 1

                # ── 宏观决策（仅非停留期）──
                is_decision = (step % DECISION_INTERVAL == 0)
                in_stay = _in_min_stay(uav)

                if is_decision and not in_stay:
                    other_uavs = [u for u in uavs if u.id != uav.id]
                    cur_state, top3 = build_macro_state(
                        uav, env, other_uavs, scene_size=env_config.scene_size)

                    # 首次决策
                    if uav.follow_cluster is None:
                        if top3[0][0] >= 0:
                            execute_macro_switch(uav, env.clusters[top3[0][0]],
                                                 env_config.slot)

                    # 完成上一个 pending
                    if pending[i] is not None:
                        prev_s, prev_a, prev_r = pending[i]
                        agent.store_transition(
                            prev_s, prev_a, prev_r, cur_state.copy(), False)
                        ep_rewards.append(prev_r)

                    # 在线 DDQN 更新
                    m = agent.train()
                    if m is not None:
                        ep_losses.append(m['loss'])

                    # epsilon-greedy 决策
                    action = agent.choose_action(cur_state, training=True)
                    action_counts[action] += 1
                    ep_decisions += 1

                    switched = False
                    if action != 0:
                        k = action - 1
                        if k < len(top3) and top3[k][0] >= 0:
                            target = env.clusters[top3[k][0]]
                            if uav.follow_cluster != target:
                                execute_macro_switch(uav, target, env_config.slot)
                                switched = True
                                ep_switches += 1

                    # 匈牙利 benefit 奖励
                    hun_r = _compute_hun_reward(uav, env.clusters, max_score_ep)
                    if switched:
                        hun_r += SWITCH_COST

                    pending[i] = (cur_state.copy(), action, hun_r)

                # ── 微观移动 ──
                if uav.follow_cluster is not None:
                    if _in_min_stay(uav):
                        micro_action = _track_cluster_action(uav)
                    else:
                        micro_action = np.random.randint(0, 9)
                    _, r, _, _, _ = uav.step(micro_action)
                else:
                    r = -10.0

                if pending[i] is not None:
                    s, a, acc = pending[i]
                    pending[i] = (s, a, acc + r)

            # 更新 is_selected
            for c in env.clusters:
                c.is_selected = False
            for uav in uavs:
                if uav.follow_cluster is not None:
                    uav.follow_cluster.is_selected = True

        # episode 结束
        for i in range(n_uav):
            if pending[i] is not None:
                s, a, r = pending[i]
                agent.store_transition(s, a, r,
                                       np.zeros(STATE_DIM, dtype=np.float32), True)
                ep_rewards.append(r)

        agent.decay_epsilon()

        # 日志
        if episode % 50 == 0 or episode == 1:
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0
            avg_r = np.mean(ep_rewards) if ep_rewards else 0.0
            avg_q = np.mean([t[2] for t in agent.buffer[-1000:]])
            act_str = f"k:{action_counts[0]} s1:{action_counts[1]} " \
                      f"s2:{action_counts[2]} s3:{action_counts[3]}"
            print(f"  {episode:4d}  {agent.epsilon:6.3f}  {avg_loss:8.4f}  "
                  f"{avg_q:8.4f}  {avg_r:+8.2f}  {ep_switches:6d}  {act_str}")

            if avg_r > best_avg_r:
                best_avg_r = avg_r
                os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.',
                            exist_ok=True)
                agent.save_model(model_save_path)

    print(f"[完成] best_avg_r={best_avg_r:+.2f}, model={model_save_path}")
    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model-path", type=str, default="models/macro_ddqn.pth")
    args = parser.parse_args()

    train_online(num_episodes=args.episodes, model_save_path=args.model_path)
