import os
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import radius_neighbors_graph

from environment import Environment


# =========================================================
# 0. 全局配置
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_USERS_PER_CLUSTER = 100.0

POSITIVE_FEATURE_INDICES = [4, 5, 6, 7, 10, 11, 12, 13]


# =========================================================
# 1. 单集群 GCN 模型
# =========================================================
class ClusterGCN(nn.Module):
    """
    单集群 GCN。

    输入：
        一个 DBSCAN 集群内部的用户图。
        节点是用户。
        节点特征是 [x, y, vx, vy]。

    输出：
        该集群的 14 维特征。
    """

    def __init__(self, in_channels=4, hidden_channels=64, out_channels=14, max_users=100.0):
        super().__init__()

        self.max_users = max_users

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # 单个集群内部做图级池化：
        # h_mean     : hidden
        # h_sum_norm : hidden
        # h_max      : hidden
        # x_mean     : 4
        # count_norm : 1
        head_in_channels = hidden_channels * 3 + in_channels + 1

        self.head = nn.Sequential(
            nn.Linear(head_in_channels, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index):
        """
        :param x: [N, 4]，单个集群内部 N 个用户的节点特征
        :param edge_index: 单个集群内部用户图的边
        :return: [1, 14]，该集群的 14 维预测特征
        """

        h = self.conv1(x, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # 单个集群内部池化，不再需要 cluster_labels
        h_mean = h.mean(dim=0, keepdim=True)

        h_sum = h.sum(dim=0, keepdim=True)
        h_sum_norm = h_sum / self.max_users

        h_max = h.max(dim=0, keepdim=True).values

        # 原始输入均值，帮助模型稳定学习中心和速度
        x_mean = x.mean(dim=0, keepdim=True)

        # 显式加入用户数量
        count_norm = torch.tensor(
            [[x.size(0) / self.max_users]],
            dtype=x.dtype,
            device=x.device
        )

        cluster_repr = torch.cat(
            [h_mean, h_sum_norm, h_max, x_mean, count_norm],
            dim=1
        )

        out = self.head(cluster_repr)

        # 注意：训练阶段不要对输出做 ReLU。
        # 非负约束只在测试展示时做后处理。
        return out


# =========================================================
# 2. 从环境中提取用户状态
# =========================================================
def extract_environment_state(env):
    """
    提取当前环境中所有用户的位置和速度。
    """

    positions = np.vstack([
        c.get_all_user_positions()
        for c in env.clusters
    ])

    velocities = np.vstack([
        np.array([u.velocity for u in c.users])
        for c in env.clusters
    ])

    return positions, velocities


# =========================================================
# 3. DBSCAN 后按簇切分
# =========================================================
def split_by_dbscan_labels(positions, velocities, labels):
    """
    根据 DBSCAN 标签，把所有用户切分成多个集群。

    返回：
        clusters = [
            {
                "label": 原始DBSCAN标签,
                "positions": 该簇用户位置,
                "velocities": 该簇用户速度
            },
            ...
        ]
    """

    clusters = []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label

        cluster_positions = positions[mask]
        cluster_velocities = velocities[mask]

        if len(cluster_positions) == 0:
            continue

        clusters.append({
            "label": label,
            "positions": cluster_positions,
            "velocities": cluster_velocities
        })

    return clusters


# =========================================================
# 4. 单集群 Ground Truth 计算
# =========================================================
def compute_single_cluster_ground_truth(
    cluster_positions,
    cluster_velocities,
    env_area_size,
    max_users=MAX_USERS_PER_CLUSTER,
    speed=5.0
):
    """
    对单个 DBSCAN 集群计算 14 维真实特征。
    输出是归一化后的 [14]。
    """

    pts = cluster_positions
    vels = cluster_velocities

    n = len(pts)

    if n == 0:
        return None

    center = np.mean(pts, axis=0)

    x_mean = center[0] / env_area_size
    y_mean = center[1] / env_area_size

    vx_mean = np.mean(vels[:, 0]) / speed if speed > 0 else np.mean(vels[:, 0])
    vy_mean = np.mean(vels[:, 1]) / speed if speed > 0 else np.mean(vels[:, 1])

    norm_n = n / max_users

    distances = np.linalg.norm(pts - center, axis=1)
    radius = np.max(distances) / (env_area_size / 2.0)

    sigma_x = np.std(pts[:, 0]) / env_area_size
    sigma_y = np.std(pts[:, 1]) / env_area_size

    left_bias = (np.median(pts[:, 0]) - center[0]) / env_area_size
    up_bias = (np.median(pts[:, 1]) - center[1]) / env_area_size

    q1 = np.sum((pts[:, 0] > center[0]) & (pts[:, 1] > center[1])) / n
    q2 = np.sum((pts[:, 0] < center[0]) & (pts[:, 1] > center[1])) / n
    q3 = np.sum((pts[:, 0] < center[0]) & (pts[:, 1] < center[1])) / n
    q4 = np.sum((pts[:, 0] > center[0]) & (pts[:, 1] < center[1])) / n

    features = [
        x_mean, y_mean,
        vx_mean, vy_mean,
        norm_n,
        radius,
        sigma_x, sigma_y,
        left_bias, up_bias,
        q1, q2, q3, q4
    ]

    return torch.tensor(features, dtype=torch.float32)


# =========================================================
# 5. 单集群图构建
# =========================================================
def build_single_cluster_graph(
    cluster_positions,
    cluster_velocities,
    env_area_size,
    speed,
    radius_threshold=150.0
):
    """
    为单个 DBSCAN 集群内部的用户构建 PyG 图。

    注意：
        这里只构建某一个集群内部的图。
        不再把所有用户一起输入 GCN。
    """

    if len(cluster_positions) == 0:
        return None

    norm_pos = cluster_positions / env_area_size
    norm_vel = cluster_velocities / speed if speed > 0 else cluster_velocities

    x_np = np.hstack([norm_pos, norm_vel])
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    if len(cluster_positions) == 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        adj = radius_neighbors_graph(
            cluster_positions,
            radius=radius_threshold,
            mode="connectivity",
            include_self=False
        )

        sources, targets = adj.nonzero()

        if len(sources) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                np.vstack([sources, targets]),
                dtype=torch.long
            )

    data = Data(x=x_tensor, edge_index=edge_index)

    return data


# =========================================================
# 6. 构造随机训练环境
# =========================================================
def make_random_env(num_slots):
    """
    每个 epoch 随机生成环境，提高泛化能力。
    """

    num_clusters = np.random.randint(4, 10)
    num_users_per_cluster = np.random.randint(30, 81)

    env = Environment(
        num_slots=num_slots,
        num_clusters=num_clusters,
        num_users_per_cluster=num_users_per_cluster
    )

    return env


# =========================================================
# 7. 预测后处理
# =========================================================
def postprocess_prediction(y_pred):
    """
    只在测试展示阶段使用。
    训练阶段不要调用。
    """

    out = y_pred.clone()

    # 用户数、半径、离散度、象限比例不能为负
    out[:, POSITIVE_FEATURE_INDICES] = torch.clamp(
        out[:, POSITIVE_FEATURE_INDICES],
        min=0.0
    )

    # 象限比例限制在 [0, 1]
    out[:, 10:14] = torch.clamp(
        out[:, 10:14],
        min=0.0,
        max=1.0
    )

    # 四象限归一化，让 q1 + q2 + q3 + q4 = 1
    q_sum = out[:, 10:14].sum(dim=1, keepdim=True)
    valid = q_sum.squeeze(1) > 1e-6

    out[valid, 10:14] = out[valid, 10:14] / q_sum[valid]

    return out


def denormalize_features(y, env_area_size, speed, max_users=MAX_USERS_PER_CLUSTER):
    """
    将 14 维归一化特征还原为物理量。
    """

    multipliers = torch.tensor(
        [
            env_area_size, env_area_size,
            speed, speed,
            max_users,
            env_area_size / 2.0,
            env_area_size, env_area_size,
            env_area_size, env_area_size,
            1.0, 1.0, 1.0, 1.0
        ],
        dtype=y.dtype,
        device=y.device
    )

    return y * multipliers


# =========================================================
# 8. 训练函数
# =========================================================
def train_gcn(
    epochs=200,
    slots_per_epoch=40,
    lr=0.003,
    radius_threshold=150.0,
    dbscan_eps_pos=120.0,
    dbscan_eps_vel=1.0,
    min_samples=15
):
    model = ClusterGCN(
        in_channels=4,
        hidden_channels=64,
        out_channels=14,
        max_users=MAX_USERS_PER_CLUSTER
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    criterion = nn.SmoothL1Loss(reduction="none")

    feature_weights = torch.tensor(
        [
            6.0, 6.0,          # x, y
            4.0, 4.0,          # vx, vy
            4.0,               # N
            2.0,               # radius
            2.0, 2.0,          # sigma_x, sigma_y
            1.0, 1.0,          # bias_x, bias_y
            1.5, 1.5, 1.5, 1.5 # quadrant ratios
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    model.train()

    for epoch in range(epochs):
        env = make_random_env(num_slots=slots_per_epoch)

        epoch_loss = 0.0
        valid_cluster_samples = 0

        for slot in range(slots_per_epoch):
            env.step()

            positions, velocities = extract_environment_state(env)

            # 1. 先对全部用户做 DBSCAN
            labels = env.dbscan_user_level(
                positions,
                velocities,
                eps_pos=dbscan_eps_pos,
                eps_vel=dbscan_eps_vel,
                min_samples=min_samples
            )

            # 2. 再按照 DBSCAN 结果切分成多个集群
            dbscan_clusters = split_by_dbscan_labels(
                positions,
                velocities,
                labels
            )

            # 3. 每个集群单独输入 GCN
            for cluster in dbscan_clusters:
                cluster_positions = cluster["positions"]
                cluster_velocities = cluster["velocities"]

                y_true = compute_single_cluster_ground_truth(
                    cluster_positions,
                    cluster_velocities,
                    env_area_size=env.area_size,
                    max_users=MAX_USERS_PER_CLUSTER,
                    speed=env.speed
                )

                if y_true is None:
                    continue

                data = build_single_cluster_graph(
                    cluster_positions,
                    cluster_velocities,
                    env_area_size=env.area_size,
                    speed=env.speed,
                    radius_threshold=radius_threshold
                )

                if data is None:
                    continue

                data = data.to(DEVICE)
                y_true = y_true.unsqueeze(0).to(DEVICE)

                optimizer.zero_grad()

                y_pred = model(
                    data.x,
                    data.edge_index
                )

                loss_matrix = criterion(y_pred, y_true)
                loss = (loss_matrix * feature_weights).mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                epoch_loss += loss.item()
                valid_cluster_samples += 1

        scheduler.step()

        avg_loss = epoch_loss / valid_cluster_samples if valid_cluster_samples > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"- LR: {current_lr:.6f} "
                f"- Avg Loss: {avg_loss:.6f} "
                f"- Valid Cluster Samples: {valid_cluster_samples}"
            )

    return model


# =========================================================
# 9. 单帧测试验证
# =========================================================
def test_and_denormalize(
    model,
    env,
    radius_threshold=150.0,
    dbscan_eps_pos=120.0,
    dbscan_eps_vel=1.0,
    min_samples=15
):
    model.eval()

    print("\n" + "=" * 70)
    print("单帧推理验证：DBSCAN 切分集群后，每个集群单独输入 GCN")
    print("=" * 70)

    env.step()

    positions, velocities = extract_environment_state(env)

    labels = env.dbscan_user_level(
        positions,
        velocities,
        eps_pos=dbscan_eps_pos,
        eps_vel=dbscan_eps_vel,
        min_samples=min_samples
    )

    dbscan_clusters = split_by_dbscan_labels(
        positions,
        velocities,
        labels
    )

    print(f"DBSCAN 识别到 {len(dbscan_clusters)} 个有效集群。")
    print(f"物理环境中实际存在 {len(env.clusters)} 个底层 cluster。")
    print()

    if len(dbscan_clusters) == 0:
        print("当前帧没有有效 DBSCAN 集群。")
        return

    env_true_centers = np.array([c.center for c in env.clusters])
    env_true_velocities = np.array([c.velocity for c in env.clusters])

    feature_names = [
        "1. 中心 X (m)",
        "2. 中心 Y (m)",
        "3. 速度 X (m/s)",
        "4. 速度 Y (m/s)",
        "5. 用户数 (人)",
        "6. 集群半径 (m)",
        "7. 离散度 X (m)",
        "8. 离散度 Y (m)",
        "9. 左右偏置 (m)",
        "10. 上下偏置 (m)",
        "11. 第一象限占比",
        "12. 第二象限占比",
        "13. 第三象限占比",
        "14. 第四象限占比"
    ]

    total_center_error = 0.0
    total_vel_error = 0.0
    valid_count = 0

    for idx, cluster in enumerate(dbscan_clusters):
        cluster_positions = cluster["positions"]
        cluster_velocities = cluster["velocities"]

        y_true = compute_single_cluster_ground_truth(
            cluster_positions,
            cluster_velocities,
            env_area_size=env.area_size,
            max_users=MAX_USERS_PER_CLUSTER,
            speed=env.speed
        )

        data = build_single_cluster_graph(
            cluster_positions,
            cluster_velocities,
            env_area_size=env.area_size,
            speed=env.speed,
            radius_threshold=radius_threshold
        )

        if y_true is None or data is None:
            continue

        data = data.to(DEVICE)
        y_true = y_true.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            y_pred_raw = model(
                data.x,
                data.edge_index
            )
            y_pred = postprocess_prediction(y_pred_raw)

        real_true = denormalize_features(
            y_true,
            env_area_size=env.area_size,
            speed=env.speed,
            max_users=MAX_USERS_PER_CLUSTER
        ).cpu()

        real_pred = denormalize_features(
            y_pred,
            env_area_size=env.area_size,
            speed=env.speed,
            max_users=MAX_USERS_PER_CLUSTER
        ).cpu()

        true_center = np.array([
            real_true[0, 0].item(),
            real_true[0, 1].item()
        ])

        pred_center = np.array([
            real_pred[0, 0].item(),
            real_pred[0, 1].item()
        ])

        true_vel = np.array([
            real_true[0, 2].item(),
            real_true[0, 3].item()
        ])

        pred_vel = np.array([
            real_pred[0, 2].item(),
            real_pred[0, 3].item()
        ])

        distances_to_physical = np.linalg.norm(
            env_true_centers - true_center,
            axis=1
        )

        matched_idx = int(np.argmin(distances_to_physical))
        phys_center = env_true_centers[matched_idx]
        phys_vel = env_true_velocities[matched_idx]

        center_error = np.linalg.norm(true_center - pred_center)
        vel_error = np.linalg.norm(true_vel - pred_vel)

        total_center_error += center_error
        total_vel_error += vel_error
        valid_count += 1

        print(f"================== DBSCAN 集群 {idx + 1} 详细报告 ==================")
        print(f"DBSCAN Label: {cluster['label']}")
        print(f"集群内部用户数: {len(cluster_positions)}")
        print()
        print(f"DBSCAN 真实中心: ({true_center[0]:.2f}, {true_center[1]:.2f})")
        print(f"GCN 预测中心:    ({pred_center[0]:.2f}, {pred_center[1]:.2f})")
        print(f"中心误差:        {center_error:.4f} m")
        print()
        print(f"DBSCAN 真实速度: ({true_vel[0]:.4f}, {true_vel[1]:.4f})")
        print(f"GCN 预测速度:    ({pred_vel[0]:.4f}, {pred_vel[1]:.4f})")
        print(f"速度误差:        {vel_error:.4f} m/s")
        print()
        print(
            f"该 DBSCAN 集群最接近底层物理 Cluster {matched_idx}，"
            f"物理中心=({phys_center[0]:.2f}, {phys_center[1]:.2f})，"
            f"物理速度=({phys_vel[0]:.4f}, {phys_vel[1]:.4f})"
        )
        print()

        print("14维特征预测对比：DBSCAN真实统计值 vs GCN预测输出值")
        print(f"{'特征名称':<16} | {'DBSCAN真实值':>16} | {'GCN预测值':>16}")
        print("-" * 60)

        for i in range(14):
            true_value = real_true[0, i].item()
            pred_value = real_pred[0, i].item()

            if i >= 10:
                true_str = f"{true_value * 100:>14.2f} %"
                pred_str = f"{pred_value * 100:>14.2f} %"
            elif i == 4:
                true_str = f"{true_value:>16.0f}"
                pred_str = f"{pred_value:>16.0f}"
            else:
                true_str = f"{true_value:>16.4f}"
                pred_str = f"{pred_value:>16.4f}"

            print(f"{feature_names[i]:<16} | {true_str} | {pred_str}")

        print()

    if valid_count > 0:
        avg_center_error = total_center_error / valid_count
        avg_vel_error = total_vel_error / valid_count

        print("=" * 70)
        print("整体误差统计")
        print(f"平均中心误差: {avg_center_error:.4f} m")
        print(f"平均速度误差: {avg_vel_error:.4f} m/s")
        print("=" * 70)


# =========================================================
# 10. 主程序入口
# =========================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"当前设备: {DEVICE}")

    print("\n==== 阶段 1：启动单集群 GCN 训练 ====")

    trained_model = train_gcn(
        epochs=200,
        slots_per_epoch=40,
        lr=0.003,
        radius_threshold=150.0,
        dbscan_eps_pos=120.0,
        dbscan_eps_vel=1.0,
        min_samples=15
    )

    print("\n==== 阶段 2：保存模型权重 ====")

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "cluster_gcn_weights.pth")
    torch.save(trained_model.state_dict(), save_path)

    print(f"模型权重已保存至: {save_path}")

    print("\n==== 阶段 3：全新随机环境测试验证 ====")

    test_model = ClusterGCN(
        in_channels=4,
        hidden_channels=64,
        out_channels=14,
        max_users=MAX_USERS_PER_CLUSTER
    ).to(DEVICE)

    test_model.load_state_dict(
        torch.load(save_path, map_location=DEVICE)
    )

    test_env = Environment(
        num_slots=100,
        num_clusters=8,
        num_users_per_cluster=45
    )

    test_and_denormalize(
        test_model,
        test_env,
        radius_threshold=150.0,
        dbscan_eps_pos=120.0,
        dbscan_eps_vel=1.0,
        min_samples=15
    )