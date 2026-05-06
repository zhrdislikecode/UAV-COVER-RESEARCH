"""
GCN 匹配网络 — 离线训练脚本

流程:
  1. 数据收集: 运行环境 + 匈牙利算法，在每个触发步收集 (图状态, 匈牙利分配标签)
  2. 离线训练: 用收集的数据训练 GCN 模型（监督学习）
  3. 保存模型: 供后续推理使用

使用:
  python -m experiment.algorithm.train_gcn

与 RL 训练完全解耦，不修改已有代码。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from experiment.config import EnvConfig, TrainConfig
from experiment.domain.env import Environment
from experiment.domain.uav import UAV
from experiment.hungarian import HungarianAssigner
from experiment.algorithm.trigger import (
    get_all_cluster_scores,
    calculate_uav_to_cluster_distances,
    assign_uavs_to_clusters,
    deploy_uavs_at_trigger_step,
)
from experiment.algorithm.gcn_matcher import (
    GCNMatcher, GCNDataCollector,
    UAV_FEAT_DIM, CLUSTER_FEAT_DIM, EDGE_FEAT_DIM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)


# ============================================================
#  数据收集
# ============================================================
def collect_training_data(num_episodes=300, save_path="models/gcn_training_data.npz",
                          use_random_actions=True, rerandomize_every=100):
    """运行环境 + 匈牙利，收集 GCN 训练数据

    Args:
        num_episodes: 收集数据的 episode 数
        save_path: 数据保存路径
        use_random_actions: True=随机动作（更多样）
        rerandomize_every: 每隔 N 个 episode 重新随机化环境（增加数据多样性）

    Returns:
        collector 对象（含所有样本）
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

    collector = GCNDataCollector()
    n_uav, n_cluster = len(uavs), len(env.clusters)
    total_samples = 0

    print(f"[数据收集] 开始 {num_episodes} episodes (每 {rerandomize_every} 集重随机化)...")
    print(f"  UAV 数: {n_uav}, 集群数: {n_cluster}")

    for episode in range(1, num_episodes + 1):
        deploy_idx = 0
        # 更频繁地重随机化环境 → 更多样的路径配置
        if (episode - 1) % rerandomize_every == 0:
            env.randomize()
            for uav in uavs:
                uav.follow_cluster_list = []
        env.reset()

        # 重置 UAV
        for uav in uavs:
            uav.position[:] = [0, 0]
            uav.current_battery_capacity = uav.total_battery_capacity
            uav.follow_cluster = None
            uav.follow_cluster_list = []
            uav.cluster_coverage_time = 0
            uav.last_cluster_id = None
            uav.macro_fly_time = 0

        for step in range(train_config.steps):
            env.step(True)  # 在线轨迹（含扰动），增加数据多样性

            # ── 匈牙利触发逻辑（复现 trigger.py）──
            cluster_centers = np.array([c.center for c in env.clusters])
            distance_matrix = calculate_uav_to_cluster_distances(uavs, cluster_centers)
            distance_matrix = 1 - distance_matrix / distance_matrix.max(
                axis=1, keepdims=True)
            score_matrix = get_all_cluster_scores(env)
            if step != 0:
                score_matrix = score_matrix / score_matrix.max()

            hungarian = HungarianAssigner(
                distance_matrix, score_matrix, step, train_config.step_change,
                train_config.weight, train_config.threshold,
            )

            if hungarian.should_assign():
                assign_vector = hungarian.assign()

                # ★ 在匈牙利修改状态之前，保存图状态和标签
                collector.collect_at_trigger(
                    uavs, env.clusters, assign_vector,
                    scene_size=env_config.scene_size,
                )
                total_samples += 1

                # 执行匈牙利分配（与 trigger.py 一致）
                assign_uavs_to_clusters(env, uavs, assign_vector, cluster_centers)
                for u in uavs:
                    u.follow_cluster_list.append(u.follow_cluster.id)
                deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx)

            # 未选中集群分值累加
            for c in env.clusters:
                if not c.is_selected:
                    c.score += 1

            # ── UAV 微行动（随机或不动）──
            for uav in uavs:
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                if use_random_actions and uav.follow_cluster is not None:
                    action = np.random.randint(0, 9)  # 随机离散动作
                    uav.step(action)

        if episode % 50 == 0:
            print(f"  已收集 {total_samples} 条样本 (episode {episode})")

    print(f"[数据收集] 完成，共 {total_samples} 条样本")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    collector.save(save_path)
    return collector


# ============================================================
#  GCN 训练
# ============================================================
def train_gcn(data_path="models/gcn_training_data.npz",
              model_save_path="models/gcn_matcher.pth",
              epochs=200, batch_size=64, lr=1e-3, hidden=64):
    """加载数据，训练 GCN 匹配网络

    Args:
        data_path: 训练数据 .npz 路径
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        hidden: GCN 隐层维度
    """
    # 1. 加载数据 + 清洗 NaN/Inf
    print(f"[GCN训练] 加载数据: {data_path}")
    data = GCNDataCollector.load(data_path)

    # 清洗：将 NaN/Inf 替换为 0（集群方向除零可能导致 NaN）
    for key in ['uav_features', 'cluster_features', 'edge_features']:
        arr = data[key]
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        if nan_mask.any() or inf_mask.any():
            print(f"  ⚠ {key}: {nan_mask.sum()} NaN, {inf_mask.sum()} Inf → 替换为 0")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            data[key] = arr

    uav_f = torch.FloatTensor(data['uav_features'])
    clu_f = torch.FloatTensor(data['cluster_features'])
    edge_f = torch.FloatTensor(data['edge_features'])
    labels = torch.LongTensor(data['assignments'])

    # 验证数据无 NaN
    for name, t in [('uav', uav_f), ('cluster', clu_f), ('edge', edge_f), ('label', labels)]:
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError(f"数据中仍有 NaN/Inf: {name}")

    n_samples, n_uav, n_cluster = labels.shape[0], labels.shape[1], clu_f.shape[1]
    print(f"  样本数: {n_samples}, UAV: {n_uav}, 集群: {n_cluster}")

    # 2. 创建 DataLoader
    dataset = TensorDataset(uav_f, clu_f, edge_f, labels)
    # 训练/验证 8:2 划分
    n_train = int(0.8 * n_samples)
    train_ds = TensorDataset(uav_f[:n_train], clu_f[:n_train],
                             edge_f[:n_train], labels[:n_train])
    val_ds = TensorDataset(uav_f[n_train:], clu_f[n_train:],
                           edge_f[n_train:], labels[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 3. 初始化模型
    model = GCNMatcher(uav_dim=UAV_FEAT_DIM, cluster_dim=CLUSTER_FEAT_DIM,
                       edge_dim=EDGE_FEAT_DIM, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. 训练循环（完全向量化，无逐样本 Python 循环）
    best_val_acc = 0.0
    print(f"[GCN训练] 开始训练 ({epochs} epochs)")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_u, batch_c, batch_e, batch_y in train_loader:
            batch_u = batch_u.to(device)    # [B, N_uav, d_u]
            batch_c = batch_c.to(device)    # [B, N_cluster, d_c]
            batch_e = batch_e.to(device)    # [B, N_uav, N_cluster, d_e]
            batch_y = batch_y.to(device)    # [B, N_uav]

            B = batch_u.shape[0]
            # 一次前向处理整个 batch
            scores = model(batch_u, batch_c, batch_e)     # [B, N_uav, N_cluster]
            loss = nn.CrossEntropyLoss()(
                scores.view(B * n_uav, n_cluster),         # [B*N_uav, N_cluster]
                batch_y.view(B * n_uav),                   # [B*N_uav]
            )
            pred = scores.argmax(dim=-1)                   # [B, N_uav]
            correct = (pred == batch_y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * B
            train_correct += correct
            train_total += B * n_uav

        scheduler.step()

        # 验证（同样向量化）
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch_u, batch_c, batch_e, batch_y in val_loader:
                    batch_u = batch_u.to(device)
                    batch_c = batch_c.to(device)
                    batch_e = batch_e.to(device)
                    batch_y = batch_y.to(device)
                    B = batch_u.shape[0]
                    scores = model(batch_u, batch_c, batch_e)  # [B, N_uav, N_cluster]
                    pred = scores.argmax(dim=-1)                # [B, N_uav]
                    val_correct += (pred == batch_y).sum().item()
                    val_total += B * n_uav

            val_acc = val_correct / val_total
            train_acc = train_correct / train_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f"  Epoch {epoch:3d}: "
                  f"train_loss={train_loss / n_train:.4f}, "
                  f"train_acc={train_acc:.4f}, "
                  f"val_acc={val_acc:.4f}")

    # 5. 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'uav_dim': UAV_FEAT_DIM,
        'cluster_dim': CLUSTER_FEAT_DIM,
        'edge_dim': EDGE_FEAT_DIM,
        'hidden': hidden,
        'val_acc': best_val_acc,
    }, model_save_path)
    print(f"[GCN训练] 模型已保存: {model_save_path} (best val_acc={best_val_acc:.4f})")

    return model, best_val_acc


# ============================================================
#  主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GCN 匹配网络 — 数据收集 & 训练")
    parser.add_argument("--collect", action="store_true", default=True,
                        help="先收集数据")
    parser.add_argument("--skip-collect", action="store_true",
                        help="跳过数据收集，直接训练已有数据")
    parser.add_argument("--episodes", type=int, default=300,
                        help="数据收集的 episode 数")
    parser.add_argument("--data-path", type=str,
                        default="models/gcn_training_data.npz")
    parser.add_argument("--model-path", type=str,
                        default="models/gcn_matcher.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    # Step 1: 数据收集
    if not args.skip_collect:
        print("=" * 60)
        print("  Step 1/2: 数据收集（匈牙利标注）")
        print("=" * 60)
        collect_training_data(
            num_episodes=args.episodes,
            save_path=args.data_path,
            use_random_actions=True,
        )

    # Step 2: GCN 训练
    print("\n" + "=" * 60)
    print("  Step 2/2: GCN 离线训练")
    print("=" * 60)
    train_gcn(
        data_path=args.data_path,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )

    print("\n[完成] GCN 模型已就绪: " + args.model_path)
    print("  使用时导入: from experiment.algorithm.gcn_matcher import GCNAssigner")
    print("  推理示例:   assigner = GCNAssigner('" + args.model_path + "')")
    print("             assign_vector = assigner.assign(uavs, env.clusters)")
