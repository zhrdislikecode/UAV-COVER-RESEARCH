from experiment.domain.env import *
from experiment.dqn import *
from experiment.hungarian import HungarianAssigner
from experiment.utils import *
from matplotlib import rcParams
np.random.seed(10)

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def main_method(env, weight = 0.25, threshold = 1):
    trigger_steps = []
    covered_rate_list, total_com_list, jain_index_list = [], [], []
    agents = [DQNAgent(state_dim=6, action_dim=9) for _ in range(len(env.uavs))]
    user_positions, cluster_positions, cluster_direction = generate_positions(env, False, steps)

    for episode in range(episodes):
        if episode == 0:
            for step in range(steps):
                env.set_positions_from_array(user_positions[step], cluster_positions[step], cluster_direction[step])
                cluster_centers = np.array([cluster.center for cluster in env.clusters])
                distance_matrix = env.calculate_uav_to_cluster_distances(cluster_centers)
                distance_matrix = 1 - distance_matrix / distance_matrix.max(axis=1, keepdims=True)
                score_matrix = env.get_all_cluster_scores()
                if step != 0:
                    score_matrix = score_matrix / score_matrix.max()
                hungarian = HungarianAssigner(distance_matrix, score_matrix, step, step_change, weight, threshold)
                if hungarian.should_assign():
                    trigger_steps.append(step)
                    assign_vector = hungarian.assign()
                    env.assign_uavs_to_clusters(assign_vector, cluster_centers)
                    for uav in env.uavs:
                        uav.follow_cluster_list.append(uav.follow_cluster.id)
                for cluster in env.clusters:
                    if not cluster.is_selected:
                        cluster.score += 1
        elif episode != episodes - 1:
            uav_deploy_idx, total_covered, total_com, total_step, sa_steps = 0, 0, 0, 1, 0
            env.reset()
            for step in range(steps):
                env.set_positions_from_array(user_positions[step], cluster_positions[step], cluster_direction[step])
                if step in trigger_steps:
                    for uav in env.uavs:
                        deploy_cluster = env.clusters[uav.follow_cluster_list[uav_deploy_idx]]
                        move_dis = np.linalg.norm(deploy_cluster.center - uav.position)
                        uav.macro_fly_time = int(move_dis * 100 / (20 * slot))
                        uav.current_battery_capacity -= uav.macro_fly_time * uav.high_speed_power
                        uav.follow_cluster = deploy_cluster
                        uav.position = np.array([deploy_cluster.center[0], deploy_cluster.center[1]])
                    uav_deploy_idx += 1
                for i in range(len(env.uavs)):
                    uav = env.uavs[i]
                    if uav.macro_fly_time > 0:
                        uav.macro_fly_time -= 1
                        continue
                    if uav.current_battery_capacity <= 0:
                        continue
                    total_step += 1
                    agent = agents[i]
                    state = env.uavs[i].get_state()
                    action = agent.choose_action(state)
                    next_state, reward, done, covered, com = uav.step(action)
                    agent.store_transition((state, action, reward, next_state, float(done)))
                    if episode <= episodes / 2:
                        agent.train()
                    else:
                        if step % 5 == 0:
                            agent.train()
                    if uav.current_battery_capacity <= 0:
                        uav.is_consume_energy = True
                        continue
                    total_covered += covered
                    total_com += com
                    if step % 30 == 0:
                        agent.update_target()
                    if (step + 1) % 100 == 0:
                        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            jain_index = calculate_jain_index(env, total_step)
            covered_rate = total_covered / (total_step * (all_user_num / cluster_num))
            covered_rate_list.append(covered_rate)
            jain_index_list.append(jain_index)
            total_com_list.append(total_com * jain_index)
            if episode % 10 == 0:
                print(f"Episode : {episode},epsilon : {agents[0].epsilon} total_covered : {total_covered} total_com: {total_com} cover_rate：{covered_rate} jain_index：{jain_index}")
        else:
            uav_position = np.zeros((steps, len(env.uavs), 2), dtype=np.float64)
            user_positions_online, cluster_positions_online, cluster_direction_online = generate_positions(env, True, steps)
            uav_deploy_idx, total_covered, total_com, total_step, sa_steps = 0, 0, 0, 1, 0
            env.reset()
            for agent in agents:
                agent.epsilon = 0
                agent.memory = []
            for step in range(steps):
                env.set_positions_from_array(user_positions_online[step], cluster_positions_online[step], cluster_direction_online[step])
                if step in trigger_steps:
                    for uav in env.uavs:
                        deploy_cluster = env.clusters[uav.follow_cluster_list[uav_deploy_idx]]
                        move_dis = np.linalg.norm(deploy_cluster.center - uav.position)
                        uav.macro_fly_time = int(move_dis * 100 / (20 * slot))
                        uav.current_battery_capacity -= uav.macro_fly_time * uav.high_speed_power
                        uav.follow_cluster = deploy_cluster
                        uav.position = np.array([deploy_cluster.center[0], deploy_cluster.center[1]])
                    uav_deploy_idx += 1
                total_step += 1
                for i in range(len(env.uavs)):
                    uav = env.uavs[i]
                    uav_position[step][i] = uav.follow_cluster.center
                    if uav.macro_fly_time > 0:
                        uav.macro_fly_time -= 1
                        continue
                    if uav.current_battery_capacity <= 0:
                        continue
                    agent = agents[i]
                    state = env.uavs[i].get_state()
                    action = agent.choose_action(state)
                    if np.random.rand() > 0:
                        action = uav.revise_direction(action)
                    next_state, reward, done, covered, com = uav.step(action)
                    if uav.current_battery_capacity <= 0:
                        uav.is_consume_energy = True
                        continue
                    total_covered += covered
                    total_com += com
            jain_index = calculate_jain_index(env, total_step)
            covered_rate = total_covered / (total_step * (all_user_num / cluster_num))
            covered_rate_list.append(covered_rate)
            jain_index_list.append(jain_index)
            total_com_list.append(total_com * jain_index)
            plot_uav_trajectory_3d(uav_position, cluster_positions_online)
            return jain_index * total_com, total_com, jain_index


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

value, total_com, jain_index = main_method(env)
