from experiment.domain.user import *
np.random.seed(4)

class Cluster:
    def __init__(self, center, radius, users_per_cluster, speed):
        self.center = np.array(center)
        self.radius = radius
        self.users = self._initialize_users(users_per_cluster)
        self.initial_user_positions = np.array([u.position.copy() for u in self.users], dtype=np.float32)
        self.score = 0
        self.is_selected = False
        self.id = 0
        self.path = None
        self.path_index = 0
        self.speed = speed / 100
        self.perturbation_probability = 0.5
        self.direction = None

    def _initialize_users(self, users_per_cluster):
        users = []
        half = users_per_cluster // 2

        # —— 小圆内的用户（半径为 0.5 * self.radius） ——
        angles1 = np.random.uniform(0, 2 * np.pi, half)
        radii1 = 0.5 * self.radius * np.sqrt(np.random.uniform(0, 1, half))
        x1 = radii1 * np.cos(angles1)
        y1 = radii1 * np.sin(angles1)

        # —— 大圆内的用户（半径为 self.radius） ——
        angles2 = np.random.uniform(0, 2 * np.pi, users_per_cluster - half)
        radii2 = self.radius * np.sqrt(np.random.uniform(0, 1, users_per_cluster - half))
        x2 = radii2 * np.cos(angles2)
        y2 = radii2 * np.sin(angles2)

        # —— 合并两个区域的坐标偏移 ——
        x_offsets = np.concatenate([x1, x2])
        y_offsets = np.concatenate([y1, y2])

        # —— 加上中心坐标，形成最终的位置数组 ——
        positions = np.stack((self.center[0] + x_offsets, self.center[1] + y_offsets), axis=1)

        for i in range(users_per_cluster):
            users.append(User(positions[i]))

        return users

    def move(self, flag = False):
        if flag:
            target = self.path[self.path_index]
            direction_vector = target - self.center
            distance = np.linalg.norm(direction_vector)

            if distance < self.speed:
                # 到达当前目标点，切换到下一个
                self.center = target.copy()
                self.path_index = (self.path_index + 1) % len(self.path)
                # 重新计算新方向
                target = self.path[self.path_index]
                direction_vector = target - self.center
                distance = np.linalg.norm(direction_vector)

            # 归一化方向向量
            direction_unit = direction_vector / distance if distance != 0 else np.zeros_like(direction_vector)

            # 生成扰动，10%概率
            if np.random.rand() < self.perturbation_probability:
                # 生成随机扰动，幅度可调，比如最多 +/- 5 米
                perturb = np.random.uniform(-self.speed, self.speed, size=direction_unit.shape)
            else:
                perturb = np.zeros_like(direction_unit)

            # 移动中心，扰动叠加到方向上
            move_vector = direction_unit * self.speed + perturb
            self.center += move_vector

            # 移动所有用户
            for user in self.users:
                user.move(move_vector)
        else:
            target = self.path[self.path_index]
            direction_vector = target - self.center
            distance = np.linalg.norm(direction_vector)

            if distance < self.speed:
                # 到达当前目标点，切换到下一个
                self.center = target.copy()
                self.path_index = (self.path_index + 1) % len(self.path)
                # 重新计算新方向
                target = self.path[self.path_index]
                direction_vector = target - self.center
                distance = np.linalg.norm(direction_vector)

            # 归一化方向向量
            direction_unit = direction_vector / distance if distance != 0 else np.zeros_like(direction_vector)
            self.center += direction_unit * self.speed
            self.direction = direction_unit * self.speed

            # 移动所有用户
            for user in self.users:
                user.move(direction_unit * self.speed)
        self.direction = (self.path[self.path_index] - self.center) / np.linalg.norm(self.path[self.path_index] - self.center) * self.speed
        return self.direction

    def get_positions(self):
        return np.array([user.position for user in self.users])


