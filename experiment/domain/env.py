from experiment.domain.cluster import *
from experiment.domain.uav import *
from sklearn.cluster import DBSCAN

class Environment:
    def __init__(self, slot, cluster_num, scene_size=15, cluster_radius=3, users_per_cluster=50, uav_num=2, uav_fly_multiple = 2):
        self.scene_size = scene_size
        self.slot = slot
        self.uav_fly_multiple = uav_fly_multiple
        # 定义用户的起点
        self.cluster_centers = np.array([(1, 4), (4, 13), (7, 13), (10, 5), (14, 1), (7, 10), (4, 5), (3, 1)], dtype=np.float64)
        # 定义兴趣点
        self.interest_points = np.array([
            [1, 1],  # 0
            [7, 1],  # 1
            [14, 1], # 2
            [1, 13], # 3
            [7, 13], # 4
            [14, 13],# 5
            [4, 5],  # 6
            [4, 9],  # 7
            [10, 5], # 8
            [10, 9], # 9
            [7, 7]   # 10
        ])
        # 定义集群的路径
        self.paths = [
            np.array([(1, 1), (14, 1), (14, 13)], dtype=np.float64),
            np.array([(1, 13), (7, 13), (14, 13)], dtype=np.float64),
            np.array([(14, 13), (14, 1)], dtype=np.float64),
            np.array([(4, 9), (4, 5), (10, 5), (4, 9)], dtype=np.float64),
            np.array([(10, 9), (4, 9), (10, 5)], dtype=np.float64),
            np.array([(10, 5), (10, 9), (14, 13), (14, 1)], dtype=np.float64),
            np.array([(7, 7), (7, 1), (1, 1), (1, 13)], dtype=np.float64),
            np.array([(4, 9), (14, 1), (1, 1)], dtype=np.float64),

        ]
        # 初始化集群
        self.clusters = []
        for i in range(cluster_num):
            cluster = Cluster(self.cluster_centers[i], cluster_radius, users_per_cluster, slot)
            cluster.center = self.cluster_centers[i].copy()
            cluster.id = i
            cluster.path = self.paths[i]
            self.clusters.append(cluster)
        # 初始化 UAV 集合
        self.uavs = [UAV(uav_id=i + 1, slot=self.slot, multiple=uav_fly_multiple) for i in range(uav_num)]

    def reset(self):
        for cluster in self.clusters:
            cluster.center = self.cluster_centers[cluster.id].copy()
            cluster.path_index = 0
            cluster.score = 0
            for i in range(len(cluster.users)):
                cluster.users[i].position = np.array(cluster.initial_user_positions[i])

        for i, uav in enumerate(self.uavs):
            uav.id = i + 1
            uav.position = np.array([0, 0], dtype=np.float32)
            uav.current_battery_capacity = uav.total_battery_capacity
            uav.follow_cluster = None
            uav.cluster_coverage_time = 0
            uav.last_cluster_id = None
            uav.macro_fly_time = 0

    def step(self, flag):
        for cluster in self.clusters:
            # 更新集群位置
            cluster.move(flag)

    def perform_clustering(self):
        all_positions = []
        for cluster in self.clusters:
            predicted_positions = np.array([user.position for user in cluster.users])
            all_positions.append(predicted_positions)

        all_positions = np.concatenate(all_positions, axis=0)

        dbscan = DBSCAN(eps=5.0, min_samples=2)
        labels = dbscan.fit_predict(all_positions)

        cluster_centers = []
        for label in np.unique(labels):
            if label == -1:
                continue  # -1 是噪声点
            cluster_points = all_positions[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)

        return np.array(cluster_centers)

    # 得到每个集群的score值集合
    def get_all_cluster_scores(self):
        scores = np.array([cluster.score for cluster in self.clusters])  # 获取每个集群的得分
        return scores

    # 得到无人机到各个簇心的距离，返回 m x n 的矩阵
    # 输入聚类后的用户集群中心位置
    def calculate_uav_to_cluster_distances(self, cluster_centers):
        num_uavs = len(self.uavs)
        num_clusters = cluster_centers.shape[0]
        distances = np.zeros((num_uavs, num_clusters))
        # 调度中心拿到无人机的位置
        uav_positions = np.array([uav.get_position() for uav in self.uavs])
        # 生成该矩阵
        for i, uav_pos in enumerate(uav_positions):
            for j in range(num_clusters):
                # 将簇心从二维转换为三维，z = 0
                cluster_center = np.array([cluster_centers[j][0], cluster_centers[j][1]])
                distances[i, j] = np.linalg.norm(uav_pos - cluster_center)

        return distances

    def assign_uavs_to_clusters(self, assignment_vector, cluster_centers):
        # 先把所有 cluster 状态重置
        for cluster in self.clusters:
            cluster.is_selected = False

        # 遍历每个 UAV，根据 assignment_vector 更新状态
        for uav_idx, cluster_id in enumerate(assignment_vector):
            # 找到该 UAV 分配到的集群
            assigned_cluster = self.clusters[cluster_id]
            assigned_cluster.is_selected = True
            assigned_cluster.score = 0

            # 设置 UAV 的位置和绑定的集群
            self.uavs[uav_idx].position = np.array([cluster_centers[cluster_id][0],
                                                    cluster_centers[cluster_id][1]])
            self.uavs[uav_idx].follow_cluster = assigned_cluster

    def set_positions_from_array(self, user_positions, cluster_positions, cluster_directions):
        for cluster_idx, cluster in enumerate(self.clusters):
            # 设置 cluster 的中心点和方向
            cluster.center = cluster_positions[cluster_idx]
            cluster.direction = cluster_directions[cluster_idx]

            # 设置 cluster 内用户的位置
            for user_idx, user in enumerate(cluster.users):
                # 这里需要用全局的 user index
                global_user_idx = sum(len(c.users) for c in self.clusters[:cluster_idx]) + user_idx
                user.position[:2] = user_positions[global_user_idx]

    def get_all_uav_positions(self):
        return np.array([uav.get_position()[:2] for uav in self.uavs], dtype=np.float64)
