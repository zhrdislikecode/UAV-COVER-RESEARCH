import numpy as np
from sklearn.cluster import DBSCAN
from experiment.domain.cluster import Cluster


class Environment:
    """环境，仅管理用户和集群。每次 reset() 自动随机化配置。"""

    # 预定义的兴趣点（固定不变）
    INTEREST_POINTS = np.array([
        [1, 1], [7, 1], [14, 1],
        [1, 13], [7, 13], [14, 13],
        [4, 5], [4, 9], [10, 5], [10, 9], [7, 7]
    ], dtype=np.float64)

    def __init__(self, slot, cluster_num, scene_size=15, cluster_radius=0.3,
                 users_per_cluster=30):
        self.scene_size = scene_size
        self.slot = slot
        self.users_per_cluster = users_per_cluster
        self.interest_points = self.INTEREST_POINTS.copy()

        if cluster_num >= len(self.interest_points):
            raise ValueError(
                f"集群数量 ({cluster_num}) 必须小于 POI 数量 ({len(self.interest_points)})"
            )

        # 创建集群占位
        self.cluster_num = cluster_num
        self.cluster_radius = cluster_radius
        self.clusters = []
        for i in range(cluster_num):
            cluster = Cluster(np.zeros(2), cluster_radius, users_per_cluster, slot)
            cluster.id = i
            self.clusters.append(cluster)

        # 自动随机化初始配置
        self.randomize()

    # ================================================================
    #  随机化
    # ================================================================
    def randomize(self):
        """随机分配集群起始 POI 和路径，强制执行无重叠同向约束"""
        pois = self.interest_points
        num_pois = len(pois)
        cluster_num = self.cluster_num
        radius = self.cluster_radius
        rng = np.random

        # 1. 每个集群选不同起始 POI
        start_indices = rng.choice(num_pois, size=cluster_num, replace=False)

        # 2. 生成满足约束的路径
        paths = self._build_valid_paths(pois, start_indices, cluster_num, radius)

        # 3. 更新集群
        self.paths = paths
        for i, cluster in enumerate(self.clusters):
            cluster.center = paths[i][0].copy()
            cluster.path = paths[i]
            cluster.path_index = 0
            cluster.score = 0
            cluster.is_selected = False
            cluster.users = cluster._initialize_users(self.users_per_cluster)
            cluster.initial_user_positions = np.array(
                [u.position.copy() for u in cluster.users], dtype=np.float32
            )

        self._initial_state = self._capture_state()

    def _build_valid_paths(self, pois, start_indices, cluster_num, radius):
        """生成满足约束的路径集合"""
        for _ in range(200):
            paths = []
            for i in range(cluster_num):
                path_len = np.random.randint(2, min(6, len(pois)))
                indices = [start_indices[i]]
                for _ in range(path_len - 1):
                    available = [j for j in range(len(pois)) if j != indices[-1]]
                    indices.append(np.random.choice(available))
                paths.append(pois[indices])
            if self._paths_satisfy_constraints(paths, radius):
                return paths
        return paths

    def _paths_satisfy_constraints(self, paths, radius):
        """检查：任意两个集群不能同时重叠且同向"""
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(paths[i][0] - paths[j][0])
                if dist >= radius * 2:
                    continue
                d_i = paths[i][1] - paths[i][0]
                d_j = paths[j][1] - paths[j][0]
                n_i, n_j = np.linalg.norm(d_i), np.linalg.norm(d_j)
                if n_i == 0 or n_j == 0:
                    continue
                if np.dot(d_i, d_j) / (n_i * n_j) > 0.95:
                    return False
        return True

    # ================================================================
    #  状态管理
    # ================================================================
    def _capture_state(self):
        return [{
            'center': c.center.copy(),
            'path_index': c.path_index,
            'score': c.score,
            'is_selected': c.is_selected,
            'user_positions': [u.position.copy() for u in c.users],
            'user_cover_nums': [u.cover_num for u in c.users],
        } for c in self.clusters]

    def reset(self):
        """重置为新的随机配置（每个 epoch 用户分布都不同）"""
        self.randomize()

    def step(self, flag):
        """推进一个时隙"""
        for cluster in self.clusters:
            cluster.move(flag)

    # ================================================================
    #  工具方法
    # ================================================================
    def perform_clustering(self):
        all_positions = np.concatenate(
            [np.array([u.position for u in c.users]) for c in self.clusters], axis=0)
        dbscan = DBSCAN(eps=5.0, min_samples=2)
        labels = dbscan.fit_predict(all_positions)
        centers = []
        for label in np.unique(labels):
            if label == -1:
                continue
            centers.append(np.mean(all_positions[labels == label], axis=0))
        return np.array(centers)

    def set_positions_from_array(self, user_positions, cluster_positions,
                                  cluster_directions):
        for ci, cluster in enumerate(self.clusters):
            cluster.center = cluster_positions[ci]
            cluster.direction = cluster_directions[ci]
            for ui, user in enumerate(cluster.users):
                gidx = sum(len(c.users) for c in self.clusters[:ci]) + ui
                user.position[:2] = user_positions[gidx]
