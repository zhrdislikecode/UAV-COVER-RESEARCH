"""
宏观调度 DDQN — 在线训练脚本

每步决策（stay 除外），min_stay=10。
Reward = 0.5*score_norm + 0.5*dist_benefit（完全对齐匈牙利）。
Score 更新: 未被覆盖 +1/步, 被覆盖 -2/步（拉大差距, 鼓励切换）。
训练时瞬移, 3 架 UAV 共享 buffer。
"""
import os
import numpy as np
import torch

from experiment.config import EnvConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.algorithm.macro_ddqn import (
    MacroAgent, build_macro_state,
    STATE_DIM, ACTION_DIM, TOP_K,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

# ============================================================
#  训练超参数
# ============================================================
MIN_STAY = 10              # 切换后最少停留步数


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
        uav.stay_remaining = 0


# ============================================================
#  在线训练
# ============================================================
def train_online(num_episodes=500, model_save_path="models/macro_ddqn.pth"):
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
    print(f"  min_stay={MIN_STAY}, UAV={n_uav}, Clusters={n_clusters}, "
          f"State={STATE_DIM}, Action={ACTION_DIM}")
    print(f"{'Ep':>5s}  {'eps':>6s}  {'loss':>8s}  {'mean_q':>8s}  "
          f"{'avg_r':>8s}  {'sw':>5s}  {'actions':>s}")

    best_avg_r = -float('inf')

    for episode in range(1, num_episodes + 1):
        env.randomize()
        env.reset()
        _reset_uavs(uavs)

        cluster_score = [0] * n_clusters
        ep_rewards = []
        ep_switches = 0
        ep_losses = []
        action_counts = [0] * ACTION_DIM

        for step in range(train_config.steps):
            env.step(False)

            # ── 更新集群分值（训练专用，不影响匈牙利）──
            # 被覆盖: -2/步；未被覆盖: +1/步；下限 0
            for j, c in enumerate(env.clusters):
                if not c.is_selected:
                    cluster_score[j] += 1
                else:
                    cluster_score[j] = max(0, cluster_score[j] - 2)
                c.score = cluster_score[j]

            max_score = max(max(c.score for c in env.clusters), 1.0)

            for i, uav in enumerate(uavs):
                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue

                # ── 停留期递减 ──
                if hasattr(uav, 'stay_remaining') and uav.stay_remaining > 0:
                    uav.stay_remaining -= 1
                    # 停留期奖励累加
                    if uav.follow_cluster is not None:
                        sn = uav.follow_cluster.score / max_score
                        uav.stay_reward_acc += 0.5 * sn + 0.5 * 1.0
                    continue

                # ── 停留期刚结束 → push transition ──
                if hasattr(uav, 'stay_pending') and uav.stay_pending is not None:
                    prev_s, prev_a = uav.stay_pending
                    other_uavs = [u for u in uavs if u.id != uav.id]
                    cur_state, _ = build_macro_state(uav, env, other_uavs,
                                                     scene_size=env_config.scene_size)
                    agent.store_transition(
                        prev_s, prev_a, uav.stay_reward_acc,
                        cur_state.copy(), False)
                    ep_rewards.append(uav.stay_reward_acc)
                    uav.stay_pending = None
                    # 在线训练
                    m = agent.train()
                    if m is not None:
                        ep_losses.append(m['loss'])

                # ── 宏观决策 ──
                other_uavs = [u for u in uavs if u.id != uav.id]
                cur_state, top3 = build_macro_state(uav, env, other_uavs,
                                                    scene_size=env_config.scene_size)

                # 首次分配
                if uav.follow_cluster is None:
                    if top3[0][0] >= 0:
                        uav.follow_cluster = env.clusters[top3[0][0]]
                        uav.position = np.array(
                            [uav.follow_cluster.center[0],
                             uav.follow_cluster.center[1]], dtype=np.float32)
                    continue

                # epsilon-greedy
                action = agent.choose_action(cur_state, training=True)
                action_counts[action] += 1
                uav_pos = uav.position[:2]

                if action == 0:
                    # ── Keep ──
                    sn = uav.follow_cluster.score / max_score
                    reward = 0.5 * sn + 0.5 * 1.0
                    agent.store_transition(
                        cur_state.copy(), 0, reward,
                        cur_state.copy(), False)
                    ep_rewards.append(reward)

                else:
                    # ── Switch ──
                    k = action - 1
                    if k < len(top3) and top3[k][0] >= 0:
                        target = env.clusters[top3[k][0]]
                        if uav.follow_cluster != target:
                            # 锁定切换瞬间的分数和距离
                            sn = target.score / max_score
                            d = float(np.linalg.norm(
                                uav_pos - target.center[:2]))
                            max_d = max(float(np.linalg.norm(
                                uav_pos - c.center[:2])) for c in env.clusters)
                            db = 1.0 - d / max(max_d, 1e-6)
                            reward = 0.5 * sn + 0.5 * db

                            # 瞬移到目标集群
                            uav.follow_cluster = target
                            uav.position = np.array(
                                [target.center[0], target.center[1]],
                                dtype=np.float32)
                            uav.stay_remaining = MIN_STAY - 1
                            uav.stay_reward_acc = reward
                            uav.stay_pending = (cur_state.copy(), action)
                            ep_switches += 1
                        else:
                            # 目标就是当前集群 → 等同于 keep
                            sn = uav.follow_cluster.score / max_score
                            reward = 0.5 * sn + 0.5 * 1.0
                            agent.store_transition(
                                cur_state.copy(), 0, reward,
                                cur_state.copy(), False)
                            ep_rewards.append(reward)

                # 在线训练
                m = agent.train()
                if m is not None:
                    ep_losses.append(m['loss'])

            # ── 更新 is_selected ──
            for c in env.clusters:
                c.is_selected = False
            for uav in uavs:
                if uav.follow_cluster is not None:
                    uav.follow_cluster.is_selected = True

        # ── episode 结束 ──
        # 清空所有 pending
        for uav in uavs:
            if hasattr(uav, 'stay_pending') and uav.stay_pending is not None:
                prev_s, prev_a = uav.stay_pending
                agent.store_transition(
                    prev_s, prev_a, uav.stay_reward_acc,
                    np.zeros(STATE_DIM, dtype=np.float32), True)
                ep_rewards.append(uav.stay_reward_acc)
                uav.stay_pending = None

        agent.decay_epsilon()

        if episode % 50 == 0 or episode == 1:
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0
            avg_r = np.mean(ep_rewards) if ep_rewards else 0.0
            avg_q = np.mean([t[2] for t in agent.buffer[-1000:]]) if agent.buffer else 0.0
            act_str = (f"k:{action_counts[0]} s1:{action_counts[1]} "
                       f"s2:{action_counts[2]} s3:{action_counts[3]}")
            print(f"  {episode:4d}  {agent.epsilon:6.3f}  {avg_loss:8.4f}  "
                  f"{avg_q:8.4f}  {avg_r:+8.4f}  {ep_switches:4d}  {act_str}")

            if avg_r > best_avg_r:
                best_avg_r = avg_r
                os.makedirs(os.path.dirname(model_save_path)
                            if os.path.dirname(model_save_path) else '.',
                            exist_ok=True)
                agent.save_model(model_save_path)

    print(f"[完成] best_avg_r={best_avg_r:+.4f}, model={model_save_path}")
    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--model-path", type=str, default="models/macro_ddqn.pth")
    args = parser.parse_args()

    train_online(num_episodes=args.episodes, model_save_path=args.model_path)
