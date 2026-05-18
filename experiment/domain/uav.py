import math
import numpy as np


# ============================================================
#  常量
# ============================================================
COVERAGE_RADIUS = 0.5          # 覆盖半径（归一化单位，1单位=100m → 50m）
SCENE_SIZE = 15.0              # 场景大小
COORD_SCALE = 100.0            # 坐标缩放：1 归一化单位 = 100 米
UAV_HEIGHT = 10.0              # 无人机飞行高度（米）
HEIGHT_SQ = UAV_HEIGHT ** 2    # 高度平方，用于 3D 距离计算

# 通信参数
BANDWIDTH_HZ = 200e6
TRANSMIT_POWER_WATT = 1.0
NOISE_DENSITY_W_PER_HZ = 1e-17
LOS_LOSS_DB = 1.0
NLOS_LOSS_DB = 50.0
A_PARAM = 15.0                 # LoS 概率公式参数 a
B_PARAM = 0.2                  # LoS 概率公式参数 b
FREQ_HZ = 2.4e9                # 载波频率


# ============================================================
#  能耗模型
# ============================================================
class PowerModel:
    """无人机功率与能耗计算"""

    def __init__(self, slot_seconds: float):
        self.slot = slot_seconds

    @staticmethod
    def power_watt(speed_m_s: float) -> float:
        """给定速度 (m/s) 下的功率 (W)"""
        v = speed_m_s
        return (
            0.0092 * v ** 3
            + 0.0166 * v ** 2
            + 88.6279 * (math.sqrt(math.sqrt(1 + v ** 4 / 1055.0673) - v ** 2 / 32.4818))
            + 79.8563
        )

    def energy_per_slot(self, speed_m_s: float) -> float:
        """给定速度下单时隙能耗"""
        return self.power_watt(speed_m_s) * self.slot / 3600.0


# ============================================================
#  通信模型
# ============================================================
def _los_probability(height: float, distance_3d: float) -> float:
    """计算视距 (LoS) 概率"""
    theta = math.degrees(math.asin(height / distance_3d))
    return 1.0 / (1.0 + A_PARAM * math.exp(-B_PARAM * (theta - A_PARAM)))


def _path_loss_db(height: float, distance_3d: float) -> float:
    """计算路径损耗 (dB)"""
    p_los = _los_probability(height, distance_3d)
    fspl = 20.0 * math.log10(4.0 * math.pi * FREQ_HZ * distance_3d / 3e8)
    return fspl + p_los * LOS_LOSS_DB + (1.0 - p_los) * NLOS_LOSS_DB


def compute_comm_rate(height: float, distance_2d: float, num_users: int = 30) -> float:
    """计算 UAV 到用户的通信速率 (Mbps)

    Args:
        height: 无人机高度 (m)
        distance_2d: 水平距离（归一化单位，会 *COORD_SCALE 转为米）
        num_users: 集群内用户数
    """
    dis_3d = math.sqrt((distance_2d * COORD_SCALE) ** 2 + height ** 2)
    pl_db = _path_loss_db(height, dis_3d)

    bw_per_user = BANDWIDTH_HZ / num_users
    noise_power = NOISE_DENSITY_W_PER_HZ * bw_per_user
    channel_gain = 10.0 ** (-pl_db / 10.0)
    snr = (TRANSMIT_POWER_WATT * channel_gain) / noise_power
    rate_bps = (BANDWIDTH_HZ / num_users) * math.log2(1.0 + snr)
    return rate_bps / 1e6  # Mbps


# ============================================================
#  UAV 类
# ============================================================
class UAV:
    # 动作空间：0=悬停, 1-4=低速移动, 5-8=中速移动
    ACTION_HOVER = 0
    ACTION_LOW_MAX = 4
    ACTION_MID_MAX = 8

    def __init__(self, uav_id: int, slot: float = 4.0, multiple: float = 2.0,
                 scene_size: float = SCENE_SIZE):
        self.id = uav_id
        self.slot = slot
        self.scene_size = scene_size
        self.position = np.array([0.0, 0.0], dtype=np.float32)
        self.follow_cluster = None
        self.follow_cluster_list: list = []
        self.total_battery_capacity = 85.0
        self.current_battery_capacity = 85.0
        self.macro_fly_time = 0
        self.cluster_coverage_time = 0
        self.last_cluster_id = None
        self.is_consume_energy = False

        # 能耗模型
        self._power = PowerModel(slot)
        self.hover_energy = self._power.energy_per_slot(0.0)
        self.low_speed_energy = self._power.energy_per_slot(1.0)
        self.mid_speed_energy = self._power.energy_per_slot(multiple)
        self.high_speed_energy = self._power.energy_per_slot(30.0)

        # 动作对应的位移量（归一化单位）
        step_low = slot / 100.0
        step_mid = multiple * slot / 100.0
        self.step_mid = step_mid          # 供 step_continuous 使用
        self.moves = np.array([
            [0.0, 0.0],            # 0: 悬停
            [0.0, step_low],       # 1: 北慢
            [0.0, -step_low],      # 2: 南慢
            [step_low, 0.0],       # 3: 东慢
            [-step_low, 0.0],      # 4: 西慢
            [0.0, step_mid],       # 5: 北中
            [0.0, -step_mid],      # 6: 南中
            [step_mid, 0.0],       # 7: 东中
            [-step_mid, 0.0],      # 8: 西中
        ], dtype=np.float32)

        # 动作 → 能耗映射
        self._action_energy = self._build_action_energy()

    def _build_action_energy(self):
        e = [0.0] * 9
        e[0] = self.hover_energy
        for i in range(1, 5):
            e[i] = self.low_speed_energy
        for i in range(5, 9):
            e[i] = self.mid_speed_energy
        return e

    # ---- 基础操作 ----
    def get_position(self):
        return self.position

    def get_state(self, all_uavs=None):
        """获取状态向量: [cx, cy, cvx, cvy, ux, uy] + [其他UAV的x,y]...

        Args:
            all_uavs: 所有 UAV 列表，用于添加其他 UAV 位置
        """
        if self.follow_cluster is None:
            base_dim = 6
            if all_uavs is not None:
                base_dim += 2 * (len(all_uavs) - 1)
            return np.zeros(base_dim, dtype=np.float32)
        cluster_pos = np.array(self.follow_cluster.center)
        cluster_dir = np.array(self.follow_cluster.direction)
        uav_pos = self.get_position()
        state = [cluster_pos[0], cluster_pos[1], cluster_dir[0], cluster_dir[1],
                 uav_pos[0], uav_pos[1]]
        if all_uavs is not None:
            for other in all_uavs:
                if other.id != self.id:
                    op = other.get_position()
                    state.extend([op[0], op[1]])
        return np.array(state, dtype=np.float32)

    # ---- 微观移动 ----
    def step(self, action: int):
        """执行单步微观移动

        Returns: (next_state, reward, done, covered_count, comm_quality)
        """
        old_position = self.position.copy()

        # 1. 扣除能耗
        self._deduct_energy(action)

        # 2. 移动
        self.position += self.moves[action]
        self.position = np.clip(self.position, 0.0, self.scene_size)

        # 3. 计算奖励
        reward, covered, comm_quality = self._compute_reward(old_position, action)

        return self.get_state(), reward, False, covered, comm_quality

    def _deduct_energy(self, action: int):
        """扣除动作对应的能耗"""
        self.current_battery_capacity -= self._action_energy[action]

    # ---- 连续动作（供 DDPG 使用） ----
    def step_continuous(self, action_2d: np.ndarray):
        """执行连续动作 (dx, dy) ∈ [-1, 1]×[-1, 1]，缩放后移动

        Returns: (next_state, reward, done, covered_count, comm_quality)
        """
        old_position = self.position.copy()
        step_mid = self.step_mid  # 中速步长作为最大步长
        move = np.clip(action_2d, -1.0, 1.0) * step_mid

        # 能耗：根据移动距离判定
        dist = float(np.linalg.norm(move))
        if dist < 1e-6:
            energy = self.hover_energy
            equiv_action = self.ACTION_HOVER
        elif dist <= step_mid * 0.5:
            energy = self.low_speed_energy
            equiv_action = 1
        else:
            energy = self.mid_speed_energy
            equiv_action = 5

        self.current_battery_capacity -= energy
        self.position += move.astype(np.float32)
        self.position = np.clip(self.position, 0.0, self.scene_size)

        reward, covered, comm_quality = self._compute_reward(old_position, equiv_action)
        return self.get_state(), reward, False, covered, comm_quality

    def _compute_reward(self, old_position: np.ndarray, action: int):
        """计算微观移动的奖励值"""
        if self.follow_cluster is None:
            return -10.0, 0, 0.0

        cluster = self.follow_cluster
        new_pos = self.position
        cluster_center = cluster.center[:2]

        new_dist_to_center = float(np.linalg.norm(new_pos - cluster_center))
        old_dist_to_center = float(np.linalg.norm(old_position[:2] - cluster_center))

        covered = 0
        comm_quality = 0.0
        num_users = len(cluster.users)

        for user in cluster.users:
            user_pos = user.position[:2]
            distance = float(np.linalg.norm(new_pos - user_pos))
            if distance <= COVERAGE_RADIUS:
                covered += 1
                user.cover_num += 1
            comm_quality += compute_comm_rate(UAV_HEIGHT, distance, num_users=num_users)

        comm_quality = comm_quality / 40.0 * (self.slot / 4.0)

        if covered == 0:
            return -new_dist_to_center, 0, comm_quality

        # 基础奖励：通信质量 + 覆盖数
        reward = 0.5 * comm_quality + 0.5 * covered

        # 动作惩罚
        if action == self.ACTION_HOVER:
            pass  # 悬停无惩罚
        elif action <= self.ACTION_LOW_MAX:
            reward -= 1.0
        else:
            reward -= 2.0

        # 远离集群惩罚
        if old_dist_to_center < new_dist_to_center:
            reward -= 5.0

        return reward, covered, comm_quality / 25.6

    # ---- 方向修正 ----
    def revise_direction(self, action: int) -> int:
        """将动作修正为朝向目标集群的方向"""
        if self.follow_cluster is None:
            return action
        target = self.follow_cluster.center + self.follow_cluster.direction
        best_action, best_dist = 0, float("inf")
        for a, move in enumerate(self.moves):
            if np.linalg.norm(move) == 0:
                continue
            new_pos = self.position + move
            dist = float(np.linalg.norm(new_pos - target))
            if dist < best_dist:
                best_dist = dist
                best_action = a
        return best_action

    # ============================================================
    #  PK 变体方法（供 DMTD 等算法使用）
    # ============================================================
    def get_dqn_pk_state(self, env):
        state = np.array([])
        uav_pos = self.get_position()
        state = np.append(state, uav_pos[0] / SCENE_SIZE)
        state = np.append(state, uav_pos[1] / SCENE_SIZE)

        if self.follow_cluster is not None:
            cluster = self.follow_cluster
            state = np.append(state, cluster.center[0] / SCENE_SIZE)
            state = np.append(state, cluster.center[1] / SCENE_SIZE)
            if cluster.direction is not None:
                state = np.append(state, cluster.direction[0])
                state = np.append(state, cluster.direction[1])
            else:
                state = np.append(state, [0.0, 0.0])
        else:
            state = np.append(state, [0.0, 0.0, 0.0, 0.0])

        max_score = max([c.score for c in env.clusters]) if env.clusters else 1.0
        for cluster in env.clusters:
            state = np.append(state, cluster.center[0] / SCENE_SIZE)
            state = np.append(state, cluster.center[1] / SCENE_SIZE)
            if cluster.direction is not None:
                state = np.append(state, cluster.direction[0])
                state = np.append(state, cluster.direction[1])
            else:
                state = np.append(state, [0.0, 0.0])
            score_norm = cluster.score / max(max_score, 1.0)
            state = np.append(state, min(score_norm, 1.0))

        for uav in env.uavs:
            if uav.id != self.id:
                other_pos = uav.get_position()
                state = np.append(state, other_pos[0] / SCENE_SIZE)
                state = np.append(state, other_pos[1] / SCENE_SIZE)

        coverage_time_norm = min(self.cluster_coverage_time / 100.0, 1.0)
        state = np.append(state, coverage_time_norm)
        return state.astype(np.float32)

    def get_dqn_pk_reward(self, env, is_macro_switch=False):
        total_comm = 0.0
        total_cover = 0
        competition_penalty = 0.0

        if self.follow_cluster is None:
            return -10.0, 0, 0.0

        cluster = self.follow_cluster
        uav_pos = self.get_position()

        if self.last_cluster_id == cluster.id:
            self.cluster_coverage_time += 1
        else:
            self.cluster_coverage_time = 1
            self.last_cluster_id = cluster.id

        covered_users = []
        for user in cluster.users:
            distance = math.sqrt(
                (uav_pos[0] - user.position[0]) ** 2
                + (uav_pos[1] - user.position[1]) ** 2
            )
            if distance <= COVERAGE_RADIUS:
                covered_users.append(user)
                user.cover_num += 1
                total_cover += 1
                dis_3d = math.sqrt((distance * COORD_SCALE) ** 2 + HEIGHT_SQ)
                total_comm += compute_comm_rate(UAV_HEIGHT, distance, num_users=len(cluster.users))

        # 竞争惩罚：与其他 UAV 覆盖同一用户
        for other_uav in env.uavs:
            if other_uav.id == self.id or other_uav.follow_cluster != cluster:
                continue
            other_pos = other_uav.get_position()
            for user in covered_users:
                dist = math.sqrt(
                    (other_pos[0] - user.position[0]) ** 2
                    + (other_pos[1] - user.position[1]) ** 2
                )
                if dist <= COVERAGE_RADIUS:
                    competition_penalty += 1.0

        # 碰撞惩罚
        collision_penalty = 0.0
        for other_uav in env.uavs:
            if other_uav.id != self.id:
                dist = math.sqrt(
                    (uav_pos[0] - other_uav.position[0]) ** 2
                    + (uav_pos[1] - other_uav.position[1]) ** 2
                )
                if dist < COVERAGE_RADIUS:
                    collision_penalty = -50.0

        # 公平性惩罚：覆盖同一集群超过 25 步
        fairness_penalty = 0.0
        if self.cluster_coverage_time > 25:
            fairness_penalty = -(self.cluster_coverage_time - 25) * 0.5

        macro_switch_penalty = -5.0 if is_macro_switch else 0.0
        total_comm = total_comm / 1000.0

        reward = (
            0.5 * total_comm
            + 0.5 * total_cover
            - competition_penalty
            + collision_penalty
            + fairness_penalty
            + macro_switch_penalty
        )
        return reward, total_cover, total_comm

    def step_pk(self, action: int, env):
        """PK 变体的 step：支持宏观集群切换 + 微观移动"""
        is_macro_switch = False

        if action >= 9:
            # 宏观动作：切换到目标集群
            target_cluster_id = action - 9
            if target_cluster_id < len(env.clusters):
                target_cluster = env.clusters[target_cluster_id]
                if self.cluster_coverage_time >= 25 or self.follow_cluster is None:
                    move_dis = float(np.linalg.norm(target_cluster.center - self.position))
                    self.macro_fly_time = int(move_dis * COORD_SCALE / (20.0 * self.slot))
                    self.current_battery_capacity -= self.macro_fly_time * self.high_speed_energy
                    self.follow_cluster = target_cluster
                    self.position = np.array(
                        [target_cluster.center[0], target_cluster.center[1]],
                        dtype=np.float32
                    )
                    self.cluster_coverage_time = 0
                    self.last_cluster_id = target_cluster.id
                    is_macro_switch = True
                else:
                    self.current_battery_capacity -= self.hover_energy
            else:
                self.current_battery_capacity -= self.hover_energy
        else:
            # 微观动作
            if self.macro_fly_time > 0:
                self.macro_fly_time -= 1
                reward, covered, comm_quality = self.get_dqn_pk_reward(env, False)
                return self.get_dqn_pk_state(env), reward, False, covered, comm_quality

            self._deduct_energy(action)
            self.position += self.moves[action]
            self.position = np.clip(self.position, 0.0, self.scene_size)

        reward, covered, comm_quality = self.get_dqn_pk_reward(env, is_macro_switch)
        return self.get_dqn_pk_state(env), reward, False, covered, comm_quality
