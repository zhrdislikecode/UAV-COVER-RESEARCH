from experiment.domain.env import *
from experiment.dqn import *
from experiment.utils import *
from matplotlib import rcParams

np.random.seed(10)

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def DMTD(env, weight=0.25, threshold=1):
    covered_rate_list, total_com_list, jain_index_list = [], [], []

    state_dim = 7 + 5 * len(env.clusters) + 2 * (len(env.uavs) - 1)
    action_dim = 9 + len(env.clusters)

    agents = [DQNAgent(state_dim=state_dim, action_dim=action_dim) for _ in range(len(env.uavs))]
    user_positions, cluster_positions, cluster_direction = generate_positions(env, False, steps)

    for episode in range(episodes):
        if episode == 0:
            for step in range(steps):
                env.set_positions_from_array(user_positions[step], cluster_positions[step], cluster_direction[step])
                if step == 0:
                    for uav in env.uavs:
                        random_cluster = np.random.choice(env.clusters)
                        uav.follow_cluster = random_cluster
                        uav.position = np.array([random_cluster.center[0], random_cluster.center[1]])
                        uav.cluster_coverage_time = 0
                        uav.last_cluster_id = random_cluster.id
                for cluster in env.clusters:
                    is_covered = any(uav.follow_cluster == cluster for uav in env.uavs)
                    if not is_covered:
                        cluster.score += 1
                    else:
                        cluster.score = 0

        elif episode != episodes - 1:
            total_covered, total_com, total_step = 0, 0, 1
            env.reset()
            for uav in env.uavs:
                random_cluster = np.random.choice(env.clusters)
                uav.follow_cluster = random_cluster
                uav.position = np.array([random_cluster.center[0], random_cluster.center[1]])
                uav.cluster_coverage_time = 0
                uav.last_cluster_id = random_cluster.id

            for step in range(steps):
                env.set_positions_from_array(user_positions[step], cluster_positions[step], cluster_direction[step])

                for cluster in env.clusters:
                    is_covered = any(uav.follow_cluster == cluster for uav in env.uavs)
                    if not is_covered:
                        cluster.score += 1
                    else:
                        cluster.score = 0

                for i in range(len(env.uavs)):
                    uav = env.uavs[i]
                    if uav.macro_fly_time > 0:
                        uav.macro_fly_time -= 1
                        continue
                    if uav.current_battery_capacity <= 0:
                        continue
                    total_step += 1
                    agent = agents[i]
                    state = uav.get_dqn_pk_state(env)
                    action = agent.choose_action(state)
                    next_state, reward, done, covered, com = uav.step_pk(action, env)
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
                print(
                    f"Episode : {episode},epsilon : {agents[0].epsilon} total_covered : {total_covered} total_com: {total_com} cover_rate：{covered_rate} jain_index：{jain_index}")

        else:
            uav_position = np.zeros((steps, len(env.uavs), 2), dtype=np.float64)
            user_positions_online, cluster_positions_online, cluster_direction_online = generate_positions(env, True,
                                                                                                           steps)
            total_covered, total_com, total_step = 0, 0, 1
            env.reset()
            for uav in env.uavs:
                random_cluster = np.random.choice(env.clusters)
                uav.follow_cluster = random_cluster
                uav.position = np.array([random_cluster.center[0], random_cluster.center[1]])
                uav.cluster_coverage_time = 0
                uav.last_cluster_id = random_cluster.id

            for agent in agents:
                agent.epsilon = 0
                agent.memory = []

            for step in range(steps):
                env.set_positions_from_array(user_positions_online[step], cluster_positions_online[step],
                                             cluster_direction_online[step])

                for cluster in env.clusters:
                    is_covered = any(uav.follow_cluster == cluster for uav in env.uavs)
                    if not is_covered:
                        cluster.score += 1
                    else:
                        cluster.score = 0

                total_step += 1
                for i in range(len(env.uavs)):
                    uav = env.uavs[i]
                    if uav.follow_cluster is not None:
                        uav_position[step][i] = uav.follow_cluster.center
                    else:
                        uav_position[step][i] = uav.position

                    if uav.macro_fly_time > 0:
                        uav.macro_fly_time -= 1
                        continue
                    if uav.current_battery_capacity <= 0:
                        continue
                    agent = agents[i]
                    state = uav.get_dqn_pk_state(env)
                    action = agent.choose_action(state)
                    next_state, reward, done, covered, com = uav.step_pk(action, env)
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

value, total_com, jain_index = DMTD(env)
