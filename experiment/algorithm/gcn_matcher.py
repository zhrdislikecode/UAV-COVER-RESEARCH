"""
GCN-based UAV-Cluster 匹配模块

通过图神经网络学习匈牙利算法的分配策略，实现无人机与集群的宏观调度。
训练与 RL 完全解耦：先用匈牙利收集标注数据，离线训练 GCN，再用于推理。

架构:
  UAV 节点 ──→ UAV Encoder ──→ UAV Embeddings ┐
                                                ├──→ Pairwise Scoring ──→ Score Matrix
  集群节点 ──→ Cluster Encoder ──→ Cluster Emb. ┘

节点数: N_uav + N_cluster (小图，无需深层 GCN)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#  特征维度常量
# ============================================================
UAV_FEAT_DIM = 6
CLUSTER_FEAT_DIM = 6
EDGE_FEAT_DIM = 4  # 第4维: per-UAV 距离排名（匹配匈牙利逻辑）


# ============================================================
#  特征提取函数（输入原始对象，输出 numpy 特征向量）
# ============================================================
def build_uav_features(uavs, scene_size=15.0):
    """为每架 UAV 构建特征向量

    返回: np.ndarray shape [N_uav, UAV_FEAT_DIM]
    """
    num_uavs = len(uavs)
    feats = np.zeros((num_uavs, UAV_FEAT_DIM), dtype=np.float32)
    for i, uav in enumerate(uavs):
        feats[i, 0] = uav.position[0] / scene_size
        feats[i, 1] = uav.position[1] / scene_size
        feats[i, 2] = uav.current_battery_capacity / max(uav.total_battery_capacity, 1e-6)
        feats[i, 3] = 1.0 if uav.follow_cluster is not None else 0.0
        feats[i, 4] = (uav.follow_cluster.id / 10.0
                       if uav.follow_cluster is not None else -0.1)
        feats[i, 5] = min(uav.cluster_coverage_time / 50.0, 1.0)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_cluster_features(clusters, scene_size=15.0):
    """为每个集群构建特征向量

    返回: np.ndarray shape [N_cluster, CLUSTER_FEAT_DIM]
    """
    num = len(clusters)
    feats = np.zeros((num, CLUSTER_FEAT_DIM), dtype=np.float32)
    scores = np.array([c.score for c in clusters], dtype=np.float32)
    max_score = max(scores.max(), 1.0)

    for i, c in enumerate(clusters):
        feats[i, 0] = c.center[0] / scene_size
        feats[i, 1] = c.center[1] / scene_size
        if c.direction is not None:
            feats[i, 2] = c.direction[0] / 0.1
            feats[i, 3] = c.direction[1] / 0.1
        feats[i, 4] = scores[i] / max_score
        feats[i, 5] = 1.0 if c.is_selected else 0.0
    # 消除集群到达路径点时 direction 除零产生的 NaN
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_edge_features(uavs, clusters, scene_size=15.0):
    """为每对 (UAV, 集群) 构建边特征

    边特征:
      0: 全局归一化距离 (dist / scene_size)
      1: 通信速率代理 (1 / (1 + dist*10))
      2: 是否当前已分配 (0/1)
      3: per-UAV 归一化距离 (1 - dist/max_dist_for_this_uav)
         ↑ 关键特征：匹配匈牙利算法的 per-UAV 距离归一化逻辑

    返回: np.ndarray shape [N_uav, N_cluster, EDGE_FEAT_DIM]
    """
    n_u, n_c = len(uavs), len(clusters)
    feats = np.zeros((n_u, n_c, EDGE_FEAT_DIM), dtype=np.float32)

    # 先算所有距离
    dists = np.zeros((n_u, n_c), dtype=np.float32)
    for i, uav in enumerate(uavs):
        for j, c in enumerate(clusters):
            dists[i, j] = float(np.linalg.norm(uav.position[:2] - c.center[:2]))

    # 构建边特征
    for i in range(n_u):
        max_d = max(dists[i].max(), 1e-6)
        for j in range(n_c):
            d = dists[i, j]
            feats[i, j, 0] = d / scene_size
            feats[i, j, 1] = 1.0 / (1.0 + d * 10.0)
            feats[i, j, 2] = 1.0 if (uavs[i].follow_cluster is not None
                                     and uavs[i].follow_cluster.id == clusters[j].id) else 0.0
            feats[i, j, 3] = 1.0 - d / max_d  # per-UAV 归一化距离收益

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_graph_state(uavs, clusters, scene_size=15.0):
    """一次性提取完整图状态

    返回: (uav_feats, cluster_feats, edge_feats)
    """
    return (
        build_uav_features(uavs, scene_size),
        build_cluster_features(clusters, scene_size),
        build_edge_features(uavs, clusters, scene_size),
    )


# ============================================================
#  GCN 匹配网络
# ============================================================
class GCNMatcher(nn.Module):
    """图神经网络：UAV-集群匹配评分

    核心设计：
      - 匈牙利残差连接: alpha*score + (1-alpha)*(1-dist/max_dist)
        直接编码匈牙利算法的线性组合逻辑，模型只需学习修正项
      - 小隐层 (32) 防止过参数化
      - 完全向量化，无 Python 循环
    """

    def __init__(self, uav_dim=UAV_FEAT_DIM, cluster_dim=CLUSTER_FEAT_DIM,
                 edge_dim=EDGE_FEAT_DIM, hidden=32):
        super().__init__()

        # 匈牙利残差：可学习权重 α（score vs distance_benefit）
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 轻量节点编码器
        self.uav_enc = nn.Sequential(
            nn.Linear(uav_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.cluster_enc = nn.Sequential(
            nn.Linear(cluster_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # 配对评分器（小容量，只学非线性修正）
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 边特征 → 标量修正
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, uav_feats, cluster_feats, edge_feats):
        """
        Args:
            uav_feats:     [B, N_uav, uav_dim]  或  [N_uav, uav_dim]
            cluster_feats: [B, N_cluster, cluster_dim]  或  [N_cluster, cluster_dim]
            edge_feats:    [B, N_uav, N_cluster, edge_dim]  或  [N_uav, N_cluster, edge_dim]

        Returns:
            scores: [B, N_uav, N_cluster]  或  [N_uav, N_cluster]
        """
        no_batch = (uav_feats.dim() == 2)
        if no_batch:
            uav_feats = uav_feats.unsqueeze(0)
            cluster_feats = cluster_feats.unsqueeze(0)
            edge_feats = edge_feats.unsqueeze(0)

        B, N_uav, _ = uav_feats.shape
        _, N_cluster, _ = cluster_feats.shape

        # 1. 匈牙利残差：直接编码线性组合
        #    cluster_feats[:,:,4] = score/max_score (归一化集群分值)
        #    edge_feats[:,:,:,3] = 1 - dist/max_dist (per-UAV 归一化距离)
        score_feat = cluster_feats[:, :, 4].unsqueeze(1)      # [B,1,Nc]
        dist_feat = edge_feats[:, :, :, 3]                     # [B,Nu,Nc]
        residual = (self.alpha * score_feat +
                    (1.0 - self.alpha) * dist_feat)            # [B,Nu,Nc]

        # 2. 神经网络修正项（小容量，学匈牙利线性组合之外的模式）
        u_h = self.uav_enc(uav_feats)            # [B, Nu, h]
        c_h = self.cluster_enc(cluster_feats)    # [B, Nc, h]
        u_h = self.dropout(u_h)

        u_exp = u_h.unsqueeze(2).expand(B, N_uav, N_cluster, -1)
        c_exp = c_h.unsqueeze(1).expand(B, N_uav, N_cluster, -1)
        pair = torch.cat([u_exp, c_exp], dim=-1)
        neural = self.pair_scorer(pair).squeeze(-1)           # [B,Nu,Nc]

        # 3. 边特征修正
        edge_bias = self.edge_enc(edge_feats).squeeze(-1)     # [B,Nu,Nc]

        scores = residual + neural + edge_bias

        if no_batch:
            scores = scores.squeeze(0)
        return scores


# ============================================================
#  GCN 分配器（推理用，接口兼容匈牙利）
# ============================================================
class GCNAssigner:
    """加载训练好的 GCN 模型，替代匈牙利算法进行 UAV-集群分配

    用法:
        assigner = GCNAssigner("models/gcn_matcher.pth")
        ...
        assign_vector = assigner.assign(uavs, env.clusters)
        # assign_vector: np.ndarray shape [N_uav], 每架 UAV 被分配的集群 ID
    """

    def __init__(self, model_path, uav_dim=UAV_FEAT_DIM,
                 cluster_dim=CLUSTER_FEAT_DIM, edge_dim=EDGE_FEAT_DIM,
                 hidden=32):
        self.model = GCNMatcher(uav_dim, cluster_dim, edge_dim, hidden).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

    @torch.no_grad()
    def get_scores(self, uavs, clusters, scene_size=15.0):
        """返回 GCN 对每对 UAV-集群的原始评分矩阵（用于日志/调试）

        Returns:
            scores: np.ndarray shape [N_uav, N_cluster]
        """
        uav_f, clu_f, edge_f = build_graph_state(uavs, clusters, scene_size)
        uav_t = torch.FloatTensor(uav_f).to(device)
        clu_t = torch.FloatTensor(clu_f).to(device)
        edge_t = torch.FloatTensor(edge_f).to(device)
        return self.model(uav_t, clu_t, edge_t).cpu().numpy()

    @torch.no_grad()
    def assign(self, uavs, clusters, scene_size=15.0):
        """给定当前 UAV 和集群状态，返回一对一分配向量

        对 GCN 评分矩阵应用匈牙利算法，强制每架 UAV 分配不同集群。
        GCN 提供"偏好评分"，匈牙利保证"互斥约束"。

        Returns:
            assignment: np.ndarray shape [N_uav], 每架 UAV 分配的集群索引
        """
        from scipy.optimize import linear_sum_assignment
        scores = self.get_scores(uavs, clusters, scene_size)
        # 匈牙利最小化 cost，GCN 分数越高越好 → cost = -scores
        cost = -scores.astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(cost)
        return col_ind.astype(np.int64)


# ============================================================
#  数据收集器（用于离线生成训练数据）
# ============================================================
class GCNDataCollector:
    """在环境中运行匈牙利算法，收集 (图状态, 匈牙利分配标签) 样本"""

    def __init__(self):
        self.samples = []  # list of (uav_feats, cluster_feats, edge_feats, assignment)

    def collect_at_trigger(self, uavs, clusters, assignment_vector, scene_size=15.0):
        """在匈牙利触发步调用，保存当前图状态和匈牙利分配结果"""
        uav_f, clu_f, edge_f = build_graph_state(uavs, clusters, scene_size)
        self.samples.append({
            'uav_features': uav_f,
            'cluster_features': clu_f,
            'edge_features': edge_f,
            'assignment': np.array(assignment_vector, dtype=np.int64),
        })

    def save(self, filepath):
        """保存收集的数据到 .npz 文件"""
        n = len(self.samples)
        if n == 0:
            print("[GCNDataCollector] 无样本，跳过保存")
            return

        n_u = self.samples[0]['uav_features'].shape[0]
        n_c = self.samples[0]['cluster_features'].shape[0]
        u_dim = self.samples[0]['uav_features'].shape[1]
        c_dim = self.samples[0]['cluster_features'].shape[1]
        e_dim = self.samples[0]['edge_features'].shape[2]

        uav_arr = np.zeros((n, n_u, u_dim), dtype=np.float32)
        clu_arr = np.zeros((n, n_c, c_dim), dtype=np.float32)
        edge_arr = np.zeros((n, n_u, n_c, e_dim), dtype=np.float32)
        assign_arr = np.zeros((n, n_u), dtype=np.int64)

        for i, s in enumerate(self.samples):
            uav_arr[i] = s['uav_features']
            clu_arr[i] = s['cluster_features']
            edge_arr[i] = s['edge_features']
            assign_arr[i] = s['assignment']

        np.savez_compressed(filepath,
                            uav_features=uav_arr,
                            cluster_features=clu_arr,
                            edge_features=edge_arr,
                            assignments=assign_arr)
        print(f"[GCNDataCollector] 保存 {n} 条样本到 {filepath}")

    @staticmethod
    def load(filepath):
        """加载保存的数据"""
        data = np.load(filepath, allow_pickle=True)
        return {
            'uav_features': data['uav_features'],
            'cluster_features': data['cluster_features'],
            'edge_features': data['edge_features'],
            'assignments': data['assignments'],
        }
