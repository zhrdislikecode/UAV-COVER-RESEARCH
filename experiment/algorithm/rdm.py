from experiment.domain.env import *
from experiment.dqn import *
from experiment.utils import *
from matplotlib import rcParams

np.random.seed(10)

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

uav_num = 2
slot = 6
all_user_num = 120
cluster_num = 4
episodes = 300
steps = 1800 // slot
step_change = steps / 10
env = Environment(
    scene_size=15,
    slot=slot,
    cluster_num=cluster_num,
    cluster_radius=0.3,
    users_per_cluster=all_user_num // cluster_num,
    uav_num=uav_num,
    uav_fly_multiple=3
)

def random_method(env):
    user_positions_online, cluster_positions_online, cluster_direction_online = generate_positions(env, True, steps)
    total_com, total_step = 0, 0
    env.reset()
    uav_position = np.zeros((steps, len(env.uavs), 2), dtype=np.float64)
    for step in range(steps):
        env.set_positions_from_array(user_positions_online[step], cluster_positions_online[step], cluster_direction_online[step])
        for i in range(len(env.uavs)):
            uav = env.uavs[i]
            uav_position[step][i] = uav.position
            if uav.current_battery_capacity <= 0:
                continue
            uav = env.uavs[i]
            total_step += 1
            action = random.randint(0, len(uav.moves) - 1)
            if action == 0:
                uav.current_battery_capacity -= uav.hover_power
            elif action <= 4:
                uav.current_battery_capacity -= uav.low_speed_power
            else:
                uav.current_battery_capacity -= uav.middle_speed_power
            uav.position[:2] += uav.moves[action] * 10
            uav.position[:2] = np.clip(uav.position[:2], 1, 14)
            com = 0
            for cluster in env.clusters:
                for user in cluster.users:
                    user_pos = np.array(user.position[:2], dtype=np.float32)
                    distance = np.linalg.norm(uav.position - user_pos)
                    if distance <= 0.5:
                        user.cover_num += 1
                        dis = math.sqrt(distance * distance * 10000 + 100)
                        com += calculate_comm_rate(10, dis, num_users=50)
            com = com / 40 * (uav.slot / 4)
            if uav.current_battery_capacity <= 0:
                uav.is_consume_energy = True
                continue
            total_com += com
    jain_index = calculate_jain_index(env, total_step)
    plot_uav_trajectory_3d(uav_position, cluster_positions_online)
    return jain_index * total_com, total_com, jain_index

value, total_com, jain_index = random_method(env)
