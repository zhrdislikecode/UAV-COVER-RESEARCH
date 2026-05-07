"""
宏观调度 DDQN — 离线训练脚本

流程:
  1. 数据收集: 环境 + 宏观 DDQN（epsilon-greedy）+ 随机微观移动
  2. 离线训练: Double DQN 更新
  3. 保存模型

训练时 decision_interval=1（每步决策），推理时才用 30。
与现有 RL 训练完全解耦，不修改任何已有文件。

使用:
  python -m experiment.algorithm.train_macro_ddqn
"""
import os
import numpy as np
import torch

from experiment.config import EnvConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.algorithm.macro_ddqn import (
    MacroAgent, build_macro_state, execute_macro_switch,
    STATE_DIM, ACTION_DIM, TOP_K,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

# ============================================================
#  训练超参数（与推理不同！）
# ============================================================
TRAIN_DECISION_INTERVAL = 1     # 训练时每步做决策
TRAIN_MIN_SWITCH_INTERVAL = 2   # 训练时几乎无冷却


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
        uav.steps_since_last_switch = 999  # 动态属性，初始无冷却


def _is_in_cooldown(uav, min_switch_interval):
    """检查 UAV 是否在切换冷却期"""
    if not hasattr(uav, 'steps_since_last_switch'):
        uav.steps_since_last_switch = 999
    return uav.steps_since_last_switch < min_switch_interval


def _ensure_has_cluster(uav, env):
    """确保 UAV 至少有一个集群（首次决策时强制分配 top-1 候选）"""
    if uav.follow_cluster is None:
        other_uavs = []  # 首次分配不需要排除兄弟
        _, top3 = build_macro_state(uav, env, other_uavs,
                                    decision_interval=TRAIN_DECISION_INTERVAL,
                                    min_switch_interval=TRAIN_MIN_SWITCH_INTERVAL,
                                    scene_size=15.0)
        if top3[0][0] >= 0:
            target = env.clusters[top3[0][0]]
            execute_macro_switch(uav, target, env_slot=4.0)


# ============================================================
#  数据收集
# ============================================================
def collect_data(num_episodes=300, save_path="models/macro_ddqn_buffer.npz"):
    """运行环境 + epsilon-greedy 宏观 DDQN，收集训练数据

    每个 episode:
      - 每步决策（interval=1）
      - 随机微观移动（聚焦宏观学习）
      - 记录 (state, action, accumulated_reward, next_state) transitions
    """
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
    total_transitions = 0

    print(f"[数据收集] 开始 {num_episodes} episodes")
    print(f"  decision_interval={TRAIN_DECISION_INTERVAL}, "
          f"min_switch_interval={TRAIN_MIN_SWITCH_INTERVAL}")
    print(f"  UAV={n_uav}, Clusters={len(env.clusters)}, "
          f"StateDim={STATE_DIM}, ActionDim={ACTION_DIM}")

    for episode in range(1, num_episodes + 1):
        env.randomize()
        env.reset()
        _reset_uavs(uavs)

        # pending[i] = (state, action, reward_acc)  — 等待下个决策步完成
        pending = [None] * n_uav
        # 集群分值
        cluster_score = [0] * len(env.clusters)

        for step in range(train_config.steps):
            env.step(False)

            # 更新集群分值（模拟匈牙利 score）
            for j, c in enumerate(env.clusters):
                if not c.is_selected:
                    cluster_score[j] += 1
                else:
                    cluster_score[j] = 0
                c.score = cluster_score[j]

            for i, uav in enumerate(uavs):
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue

                if hasattr(uav, 'steps_since_last_switch'):
                    uav.steps_since_last_switch += 1

                # ── 宏观决策 ──
                if step % TRAIN_DECISION_INTERVAL == 0 and \
                   not _is_in_cooldown(uav, TRAIN_MIN_SWITCH_INTERVAL):

                    other_uavs = [u for u in uavs if u.id != uav.id]
                    cur_state, top3 = build_macro_state(
                        uav, env, other_uavs,
                        decision_interval=TRAIN_DECISION_INTERVAL,
                        min_switch_interval=TRAIN_MIN_SWITCH_INTERVAL,
                        scene_size=env_config.scene_size,
                    )

                    # 首次决策：确保 UAV 有集群
                    if uav.follow_cluster is None:
                        _ensure_has_cluster(uav, env)

                    # 完成上一个 pending → push transition
                    if pending[i] is not None:
                        prev_state, prev_action, prev_reward = pending[i]
                        agent.store_transition(
                            prev_state, prev_action, prev_reward,
                            cur_state.copy(), False)
                        total_transitions += 1

                    # 新决策
                    action = agent.choose_action(cur_state, training=True)
                    if action != 0:
                        k = action - 1
                        if k < len(top3) and top3[k][0] >= 0:
                            target = env.clusters[top3[k][0]]
                            if uav.follow_cluster != target:
                                execute_macro_switch(uav, target, env_config.slot)

                    pending[i] = (cur_state.copy(), action, 0.0)

                # ── 微观移动（随机）──
                if uav.follow_cluster is not None:
                    micro_action = np.random.randint(0, 9)
                    _, reward, _, _, _ = uav.step(micro_action)
                else:
                    reward = -10.0

                if pending[i] is not None:
                    s, a, r = pending[i]
                    pending[i] = (s, a, r + reward)

            # 更新 is_selected
            for c in env.clusters:
                c.is_selected = False
            for uav in uavs:
                if uav.follow_cluster is not None:
                    uav.follow_cluster.is_selected = True

        # episode 结束，完成所有 pending
        for i in range(n_uav):
            if pending[i] is not None:
                s, a, r = pending[i]
                agent.store_transition(s, a, r,
                                       np.zeros(STATE_DIM, dtype=np.float32),
                                       True)
                total_transitions += 1

        if episode % 50 == 0:
            print(f"  Episode {episode}: {total_transitions} transitions, "
                  f"epsilon={agent.epsilon:.4f}")

    print(f"[数据收集] 完成，共 {total_transitions} 条 transitions")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)

    # 保存 replay buffer
    buffer_data = {
        'states': np.array([t[0] for t in agent.buffer], dtype=np.float32),
        'actions': np.array([t[1] for t in agent.buffer], dtype=np.int64),
        'rewards': np.array([t[2] for t in agent.buffer], dtype=np.float32),
        'next_states': np.array([t[3] for t in agent.buffer], dtype=np.float32),
        'dones': np.array([t[4] for t in agent.buffer], dtype=np.bool_),
    }
    np.savez_compressed(save_path, **buffer_data)
    print(f"[数据收集] buffer 已保存: {save_path}")
    return agent


# ============================================================
#  离线训练
# ============================================================
def train_macro_ddqn(buffer_path="models/macro_ddqn_buffer.npz",
                     model_save_path="models/macro_ddqn.pth",
                     epochs=200, batch_size=128, lr=1e-3):
    """从保存的 buffer 数据训练 DDQN"""
    print(f"[训练] 加载 buffer: {buffer_path}")
    data = np.load(buffer_path, allow_pickle=True)

    agent = MacroAgent(batch_size=batch_size, lr=lr)
    # 恢复 buffer
    n = len(data['states'])
    for i in range(n):
        agent.store_transition(
            data['states'][i],
            data['actions'][i],
            data['rewards'][i],
            data['next_states'][i],
            bool(data['dones'][i]),
        )
    print(f"  加载 {n} 条 transitions")
    agent.epsilon = 0.5  # 从中间值开始继续衰减

    best_loss = float('inf')
    print(f"[训练] 开始 {epochs} epochs")

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        # 每个 epoch 做多次更新
        n_updates = min(len(agent.buffer) // batch_size, 200)
        for _ in range(n_updates):
            loss = agent.train()
            if loss is not None:
                epoch_losses.append(loss)

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            if avg_loss < best_loss:
                best_loss = avg_loss

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: avg_loss={np.mean(epoch_losses):.4f}, "
                  f"epsilon={agent.epsilon:.4f}, buffer={len(agent.buffer)}")

    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.',
                exist_ok=True)
    agent.save_model(model_save_path)
    print(f"[训练] 模型已保存: {model_save_path}")
    return agent


# ============================================================
#  主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="宏观调度 DDQN — 数据收集 & 训练")
    parser.add_argument("--collect-episodes", type=int, default=300,
                        help="数据收集的 episode 数")
    parser.add_argument("--buffer-path", type=str,
                        default="models/macro_ddqn_buffer.npz")
    parser.add_argument("--model-path", type=str,
                        default="models/macro_ddqn.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-collect", action="store_true",
                        help="跳过数据收集，直接训练已有 buffer")
    args = parser.parse_args()

    # Step 1: 数据收集
    if not args.skip_collect:
        print("=" * 60)
        print("  Step 1/2: 宏观调度数据收集")
        print("=" * 60)
        collect_data(num_episodes=args.collect_episodes,
                     save_path=args.buffer_path)

    # Step 2: 离线训练
    print("\n" + "=" * 60)
    print("  Step 2/2: DDQN 离线训练")
    print("=" * 60)
    train_macro_ddqn(buffer_path=args.buffer_path,
                     model_save_path=args.model_path,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     lr=args.lr)

    print("\n[完成] 宏观调度 DDQN 模型已就绪: " + args.model_path)
    print("  推理时导入: from experiment.algorithm.macro_ddqn import MacroAgent")
    print("  加载模型:    agent = MacroAgent()")
    print("              agent.load_model('" + args.model_path + "')")
