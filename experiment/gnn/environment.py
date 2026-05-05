import os
import numpy as np
import matplotlib.pyplot as plt


# ================= 1. User 类 =================
class User:
    def __init__(self, offset, velocity):
        """
        用户类，依附于集群存在
        :param offset: 用户相对于集群中心的二维偏移量 np.array([x, y]) (单位：米)
        :param velocity: 用户的初始速度矢量 np.array([vx, vy]) (单位：米/时隙)
        """
        self.offset = offset
        self.velocity = velocity.copy()


# ================= 2. Cluster 类 =================
class Cluster:
    def __init__(self, cluster_id, start_center, radius, target_poi_idx, num_users, pois, speed):
        self.cluster_id = cluster_id
        self.center = start_center.copy()
        self.radius = radius
        self.target_poi_idx = target_poi_idx
        self.num_users = num_users

        target_pos = pois[self.target_poi_idx]
        direction = target_pos - self.center
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.velocity = (direction / dist) * speed
        else:
            self.velocity = np.zeros(2)

        self.users = []
        self._generate_users()

    def _generate_users(self):
        r = self.radius * np.sqrt(np.random.uniform(0, 1, self.num_users))
        theta = np.random.uniform(0, 2 * np.pi, self.num_users)

        for i in range(self.num_users):
            offset = np.array([r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])])
            self.users.append(User(offset, self.velocity))

    def move_towards_target(self, pois, speed, area_size):
        """执行单个集群的移动逻辑"""
        target = pois[self.target_poi_idx]
        direction = target - self.center
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target > 0:
            direction = direction / distance_to_target

            if distance_to_target <= speed:
                self.center = target.copy()
                available_idx = [j for j in range(len(pois)) if j != self.target_poi_idx]
                self.target_poi_idx = np.random.choice(available_idx)

                new_target = pois[self.target_poi_idx]
                new_dir = new_target - self.center
                new_dist = np.linalg.norm(new_dir)
                if new_dist > 0:
                    self.velocity = (new_dir / new_dist) * speed
            else:
                self.center += direction * speed
                self.velocity = direction * speed

        # 边界检测
        cx, cy = self.center
        self.center[0] = np.clip(cx, self.radius, area_size - self.radius)
        self.center[1] = np.clip(cy, self.radius, area_size - self.radius)

        for user in self.users:
            user.velocity = self.velocity.copy()

    def get_all_user_positions(self):
        if not self.users:
            return np.empty((0, 2))
        offsets = np.array([user.offset for user in self.users])
        return self.center + offsets


# ================= 3. Environment 类 =================
class Environment:
    def __init__(self, num_slots, num_clusters, num_users_per_cluster=50):
        self.all_time_slots = 900
        # 【修改点 1】：总区域大小 15.0 -> 1500.0 米 (1.5km x 1.5km)
        self.area_size = 1500.0
        self.num_slots = num_slots
        # 【修改点 2】：速度 0.05 -> 5.0 米/时隙
        self.speed = 5.0
        self.num_users_per_cluster = num_users_per_cluster

        # 【修改点 3】：所有兴趣点的坐标放大 100 倍
        self.pois = np.array([
            [200.0, 200.0], [750.0, 200.0], [1300.0, 200.0],
            [1050.0, 450.0], [450.0, 750.0], [1050.0, 750.0],
            [450.0, 1050.0], [200.0, 1300.0], [750.0, 1300.0], [1300.0, 1300.0],
            [200.0, 750.0], [1300.0, 750.0], [750.0, 750.0], [450.0, 450.0], [1050.0, 1050.0]
        ])

        if num_clusters > len(self.pois):
            raise ValueError(f"集群数量({num_clusters})不能大于POI数量({len(self.pois)})，否则无法保证初始时不重叠。")

        self.clusters = []

        initial_poi_indices = np.random.choice(len(self.pois), size=num_clusters, replace=False)

        # 【修改点 4】：集群半径范围 0.2~0.5 -> 20.0~50.0 米
        cluster_radii = np.random.uniform(20.0, 50.0, num_clusters)

        for i in range(num_clusters):
            start_center = self.pois[initial_poi_indices[i]]

            available_idx = [j for j in range(len(self.pois)) if j != initial_poi_indices[i]]
            target_idx = np.random.choice(available_idx)

            cluster = Cluster(
                cluster_id=i,
                start_center=start_center,
                radius=cluster_radii[i],
                target_poi_idx=target_idx,
                num_users=self.num_users_per_cluster,
                pois=self.pois,
                speed=self.speed
            )
            self.clusters.append(cluster)

        self._resolve_constraints()

    def _resolve_constraints(self):
        """核心约束机制：检查并处理空间重叠且方向相同的集群"""
        n = len(self.clusters)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = self.clusters[i]
                c2 = self.clusters[j]

                dist = np.linalg.norm(c1.center - c2.center)
                if dist < (c1.radius + c2.radius):

                    v1_norm = np.linalg.norm(c1.velocity)
                    v2_norm = np.linalg.norm(c2.velocity)

                    same_direction = False

                    if v1_norm > 0 and v2_norm > 0:
                        cos_sim = np.dot(c1.velocity, c2.velocity) / (v1_norm * v2_norm)
                        if cos_sim > 0.95:
                            same_direction = True
                    elif v1_norm == 0 and v2_norm == 0:
                        if c1.target_poi_idx == c2.target_poi_idx:
                            same_direction = True

                    if same_direction:
                        available_idx = [k for k in range(len(self.pois)) if k != c2.target_poi_idx]
                        new_target = np.random.choice(available_idx)
                        c2.target_poi_idx = new_target

                        new_dir = self.pois[new_target] - c2.center
                        new_dist = np.linalg.norm(new_dir)
                        if new_dist > 0:
                            c2.velocity = (new_dir / new_dist) * self.speed

                        for user in c2.users:
                            user.velocity = c2.velocity.copy()

    def step(self):
        """环境推进一个时隙"""
        for cluster in self.clusters:
            cluster.move_towards_target(self.pois, self.speed, self.area_size)

        self._resolve_constraints()

    # 【修改点 5】：DBSCAN 的判别阈值同步放大。eps_pos 1.2 -> 120.0米, eps_vel 0.01 -> 1.0 米/时隙
    def dbscan_user_level(self, positions, velocities, eps_pos=120.0, eps_vel=1.0, min_samples=15):
        """对所有用户的坐标和速度向量直接进行DBSCAN计算"""
        n = len(positions)
        if n == 0:
            return np.array([])

        pos_dist = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
        vel_dist = np.linalg.norm(velocities[:, np.newaxis, :] - velocities[np.newaxis, :, :], axis=2)

        neighbors_matrix = (pos_dist < eps_pos) & (vel_dist < eps_vel)

        labels = np.full(n, -2)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -2:
                continue

            neighbors = np.where(neighbors_matrix[i])[0]
            if len(neighbors) < min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seed_set = list(neighbors)
            if i in seed_set:
                seed_set.remove(i)

            while seed_set:
                q = seed_set.pop(0)
                if labels[q] == -1:
                    labels[q] = cluster_id
                if labels[q] != -2:
                    continue

                labels[q] = cluster_id
                q_neighbors = np.where(neighbors_matrix[q])[0]

                if len(q_neighbors) >= min_samples:
                    seed_set.extend([k for k in q_neighbors if labels[k] == -2])

            cluster_id += 1

        return labels

    def plot_distribution(self, current_slot, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_xlabel("X Coordinate (m)")
        ax.set_ylabel("Y Coordinate (m)")

        ax.scatter(self.pois[:, 0], self.pois[:, 1], c='red', marker='*', s=100, alpha=0.3, label='POIs')

        all_user_positions = []
        all_user_velocities = []

        for cluster in self.clusters:
            pos = cluster.get_all_user_positions()
            vel = np.array([user.velocity for user in cluster.users])

            all_user_positions.append(pos)
            all_user_velocities.append(vel)

        if all_user_positions:
            all_user_positions = np.vstack(all_user_positions)
            all_user_velocities = np.vstack(all_user_velocities)

            ax.scatter(all_user_positions[:, 0], all_user_positions[:, 1], c='black', s=2, label='Users')

            # 使用放大后的 DBSCAN 阈值进行聚类
            cluster_labels = self.dbscan_user_level(
                all_user_positions, all_user_velocities,
                eps_pos=120.0, eps_vel=1.0, min_samples=15
            )

            unique_labels = np.unique(cluster_labels)
            current_centroids = []

            for label in unique_labels:
                if label == -1:
                    continue

                cluster_points = all_user_positions[cluster_labels == label]
                centroid = np.mean(cluster_points, axis=0)
                current_centroids.append(np.round(centroid, 2))

                max_distance = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
                # 【修改点 6】：画图时的边界留白 0.1 -> 10.0 米
                draw_radius = max_distance + 10.0

                circle = plt.Circle((centroid[0], centroid[1]), draw_radius, color='red',
                                    linestyle='--', fill=False, linewidth=2, alpha=0.8)
                ax.add_artist(circle)

            print(
                f"[{current_slot:04d} 时隙] DBSCAN识别出 {len(current_centroids)} 个群体。簇心坐标: {current_centroids}")

        ax.set_title(f"User Distribution - Slot {current_slot}/{self.num_slots}")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def run(self, plot_interval, save_dir='frames'):
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"开始生成图片，将保存到 {save_dir}/ 目录...")

        for slot in range(self.num_slots + 1):
            if slot % plot_interval == 0:
                save_path = os.path.join(save_dir, f'slot_{slot:04d}.png')
                self.plot_distribution(slot, save_path=save_path)
            if slot < self.num_slots:
                self.step()

        print(f"\n图片生成完成！共保存 {(self.num_slots // plot_interval) + 1} 张图片")

# ================= 使用示例 =================
if __name__ == "__main__":
    env = Environment(num_slots=300, num_clusters=10, num_users_per_cluster=60)
    env.run(plot_interval=20)