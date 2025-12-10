import math

import numpy as np

# α：建成用地面积与总用地面积之比
# β：单位面积的平均建筑数量
# γ：根据瑞利概率密度函数描述建筑物高度分布的尺度参数
# (α, β, γ)为三元组，郊区(0.1, 750, 8)，城市(0.3, 500, 15)，密集城市(0.5, 300, 20)，高层城市(0.5, 300, 50)
# a 和 b 是由多项式计算而来
a = 15
b = 0.2
los_loss_value = 1
nlos_loss_value = 50

def calculate_LoS_possibility(height, distance):
    # 计算角度 θ
    theta = math.asin(height / distance)
    theta = math.degrees(theta)
    # 计算 P(LoS, θ)
    p_los = 1 / (1 + a * math.exp(-b * (theta - a)))
    return p_los

def calculate_path_loss(height, distance):
    los_possibility = calculate_LoS_possibility(height, distance)
    return 20 * math.log10(4 * math.pi * 2.4e9 * distance / 3e8) + los_possibility * los_loss_value + (1 - los_possibility) * nlos_loss_value

def calculate_uav_rate_from_path_loss(
    pl_a2g_db,
    bandwidth_hz=200e6,
    time_fraction=1.0,
    transmit_power_watt=1.0,
    noise_density_w_per_hz=1e-17,
    num_users=30
):
    # 每个用户分到的带宽
    bandwidth_per_user = bandwidth_hz / num_users

    # 总噪声功率
    noise_power = noise_density_w_per_hz * bandwidth_per_user

    # 信道增益（线性）
    channel_gain = 10 ** (-pl_a2g_db / 10)

    # 信噪比 SNR（线性）
    snr = (transmit_power_watt * channel_gain) / noise_power

    # 速率计算（bit/s）
    rate_bps = (bandwidth_hz * time_fraction / num_users) * np.log2(1 + snr)

    # Mbps 输出
    rate_mbps = rate_bps / 1e6
    return rate_mbps

def calculate_comm_rate(height, distance, num_users = 30):
    path_loss = calculate_path_loss(height, distance)
    return calculate_uav_rate_from_path_loss(path_loss, num_users = num_users)

def P(v):
    return (0.0092 * v ** 3
            + 0.0166 * v ** 2
            + 88.6279 * (math.sqrt(math.sqrt(1 +  v ** 4 / 1055.0673) -  v ** 2 / 32.4818))
            + 79.8563)



class UAV:
    def __init__(self, uav_id, position=(0, 0),  slot = 4, multiple = 2):
        self.id = uav_id
        self.position = np.array(position, dtype=np.float32)
        self.follow_cluster = None
        self.follow_cluster_list = list()
        self.total_battery_capacity = 85
        self.current_battery_capacity = 85
        self.hover_power = P(0) * slot / 3600
        self.low_speed_power = P(1) * slot / 3600
        self.middle_speed_power = P(multiple) * slot / 3600
        self.slot = slot
        self.high_speed_power = P(30) * slot / 3600
        self.moves = [
            np.array([0, 0]),
            np.array([0, slot / 100]),
            np.array([0, -slot / 100]),
            np.array([slot / 100, 0]),
            np.array([-slot / 100, 0]),
            np.array([0, multiple * slot / 100]),
            np.array([0, -multiple * slot / 100]),
            np.array([multiple * slot / 100, 0]),
            np.array([-multiple * slot / 100, 0])
        ]
        self.macro_fly_time = 0
        self.cluster_coverage_time = 0  # 追踪当前集群的覆盖时间
        self.last_cluster_id = None  # 记录上一个集群ID

    def move(self, direction):
        self.position += np.array(direction, dtype=np.float32)

    def revise_direction(self, action):
        target_position = self.follow_cluster.center + self.follow_cluster.direction
        best_action, best_dist = 0, float("inf")
        for action, move in enumerate(self.moves):
            if np.linalg.norm(move) == 0:
                continue
            new_position = self.position + move
            dist = np.linalg.norm(new_position - target_position)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action

    def get_position(self):
        return self.position

    def get_state(self):
        cluster_pos = np.array(self.follow_cluster.center)
        cluster_dir = np.array(self.follow_cluster.direction)
        uav_pos = self.get_position()

        state = np.concatenate([cluster_pos, cluster_dir, uav_pos], axis=0)
        return state.astype(np.float32)

    def step(self, action):
        if action == 0:
            self.current_battery_capacity -= self.hover_power
        elif action <= 4:
            self.current_battery_capacity -= self.low_speed_power
        else:
            self.current_battery_capacity -= self.middle_speed_power

        position = np.array(self.position[:2], dtype=np.float32)

        self.position[:2] += self.moves[action]

        self.position[:2] = np.clip(self.position[:2], 0, 15)

        reward, covered, comm_quality = self.get_reward(position, action)

        done = False

        return self.get_state(), reward, done, covered, comm_quality

    def get_reward(self, position=None, action=None):
        covered = 0
        comm_quality = 0
        cluster = self.follow_cluster
        uav_pos = np.array(self.get_position()[:2], dtype=np.float32)
        cluster_center = np.array(cluster.center[:2], dtype=np.float32)
        dis_uav_cluster = np.linalg.norm(uav_pos - cluster_center)
        dis_uav_cluster_old = np.linalg.norm(position[:2] - cluster_center)
        for user in cluster.users:
            user_pos = np.array(user.position[:2], dtype=np.float32)
            distance = np.linalg.norm(uav_pos - user_pos)
            if distance <= 0.5:
                covered += 1
                user.cover_num += 1
            dis = math.sqrt(distance * distance * 10000 + 100)
            comm_quality += calculate_comm_rate(10, dis, num_users=len(self.follow_cluster.users))
        comm_quality = comm_quality / 40 * (self.slot / 4)
        if covered == 0:
            return -dis_uav_cluster, 0, comm_quality

        if action == 0:
            reward = 0.5 * comm_quality + 0.5 * covered
        elif action <= 4:
            reward = 0.5 * comm_quality + 0.5 * covered - 1
        else:
            reward = 0.5 * comm_quality + 0.5 * covered - 2

        if dis_uav_cluster_old < dis_uav_cluster:
            reward -= 5

        return reward, covered, comm_quality / 25.6

    def get_dqn_pk_state(self, env):
        state = np.array([])
        
        # 1. 当前UAV位置 (归一化到[0,1])
        uav_pos = self.get_position()
        state = np.append(state, uav_pos[0] / 15.0)
        state = np.append(state, uav_pos[1] / 15.0)
        
        # 2. 当前跟随的集群信息
        if self.follow_cluster is not None:
            cluster = self.follow_cluster
            state = np.append(state, cluster.center[0] / 15.0)
            state = np.append(state, cluster.center[1] / 15.0)
            if cluster.direction is not None:
                state = np.append(state, cluster.direction[0])
                state = np.append(state, cluster.direction[1])
            else:
                state = np.append(state, 0.0)
                state = np.append(state, 0.0)
        else:

            state = np.append(state, [0.0, 0.0, 0.0, 0.0])
        

        max_score = max([c.score for c in env.clusters]) if env.clusters else 1.0
        for cluster in env.clusters:
            state = np.append(state, cluster.center[0] / 15.0)
            state = np.append(state, cluster.center[1] / 15.0)
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
                state = np.append(state, other_pos[0] / 15.0)
                state = np.append(state, other_pos[1] / 15.0)
        

        coverage_time_norm = min(self.cluster_coverage_time / 100.0, 1.0)
        state = np.append(state, coverage_time_norm)
        
        return state.astype(np.float32)

    def get_dqn_pk_reward(self, env, is_macro_switch=False):
        total_comm = 0
        total_cover = 0
        competition_penalty = 0
        
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
            distance = math.sqrt((uav_pos[0] - user.position[0]) ** 2 + (uav_pos[1] - user.position[1]) ** 2)
            if distance <= 0.5:
                covered_users.append(user)
                user.cover_num += 1
                total_cover += 1
                dis_m = distance * 100
                dis_3d = math.sqrt(dis_m * dis_m + 100)
                total_comm += calculate_comm_rate(10, dis_3d, num_users=len(cluster.users))

        for other_uav in env.uavs:
            if other_uav.id != self.id and other_uav.follow_cluster == cluster:
                other_pos = other_uav.get_position()
                for user in covered_users:
                    distance = math.sqrt((other_pos[0] - user.position[0]) ** 2 + (other_pos[1] - user.position[1]) ** 2)
                    if distance <= 0.5:
                        competition_penalty += 1

        collision_penalty = 0
        for other_uav in env.uavs:
            if other_uav.id != self.id:
                distance = math.sqrt((uav_pos[0] - other_uav.position[0]) ** 2 + 
                                   (uav_pos[1] - other_uav.position[1]) ** 2)
                if distance < 0.5:  # 距离过近
                    collision_penalty = -50
        

        fairness_penalty = 0
        if self.cluster_coverage_time > 25:
            excess_time = self.cluster_coverage_time - 25
            fairness_penalty = -excess_time * 0.5

        macro_switch_penalty = 0
        if is_macro_switch:
            macro_switch_penalty = -5.0

        total_comm = total_comm / 1000

        reward = (0.5 * total_comm + 0.5 * total_cover + 
                 competition_penalty * (-1.0) + 
                 collision_penalty + 
                 fairness_penalty + 
                 macro_switch_penalty)
        
        return reward, total_cover, total_comm

    def step_pk(self, action, env):

        is_macro_switch = False

        if action >= 9:
            target_cluster_id = action - 9
            if target_cluster_id < len(env.clusters):
                target_cluster = env.clusters[target_cluster_id]

                if self.cluster_coverage_time >= 25 or self.follow_cluster is None:
                    if self.follow_cluster is not None:
                        move_dis = np.linalg.norm(target_cluster.center - self.position)
                    else:
                        move_dis = np.linalg.norm(target_cluster.center - self.position)

                    self.macro_fly_time = int(move_dis * 100 / (20 * self.slot))
                    self.current_battery_capacity -= self.macro_fly_time * self.high_speed_power

                    self.follow_cluster = target_cluster
                    self.position = np.array([target_cluster.center[0], target_cluster.center[1]])
                    self.cluster_coverage_time = 0
                    self.last_cluster_id = target_cluster.id
                    is_macro_switch = True
                else:

                    action = 0
                    self.current_battery_capacity -= self.hover_power
            else:
                action = 0
                self.current_battery_capacity -= self.hover_power
        else:
            # 微观移动动作
            if self.macro_fly_time > 0:
                # 如果正在宏观飞行，不能执行微观动作
                self.macro_fly_time -= 1
                reward, covered, comm_quality = self.get_dqn_pk_reward(env, False)
                done = False
                return self.get_dqn_pk_state(env), reward, done, covered, comm_quality

            if action == 0:
                self.current_battery_capacity -= self.hover_power
            elif action <= 4:
                self.current_battery_capacity -= self.low_speed_power
            else:
                self.current_battery_capacity -= self.middle_speed_power
            
            self.position = np.array(self.position, dtype=np.float32)
            self.position += self.moves[action]
            self.position = np.clip(self.position, 0, 15)

        reward, covered, comm_quality = self.get_dqn_pk_reward(env, is_macro_switch)
        done = False
        return self.get_dqn_pk_state(env), reward, done, covered, comm_quality

