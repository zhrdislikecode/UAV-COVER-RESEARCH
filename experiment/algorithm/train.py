import os
from experiment.utils import generate_positions, calculate_jain_index
from experiment.algorithm.trigger import (compute_trigger_steps,
                                           deploy_uavs_at_trigger_step)


def run_training(env, uavs, agent, config, save_dir="models"):
    """训练通用模型：每隔 rerandomize_interval 轮切换一次随机环境"""
    covered_rate_list, total_com_list, jain_index_list = [], [], []

    user_pos = cluster_pos = cluster_dir = None
    trigger_steps = None

    for episode in range(1, config.episodes - 1):
        # 每隔 N 轮重新随机化环境，其余轮沿用同一配置
        if (episode - 1) % config.rerandomize_interval == 0:
            env.randomize()
            for uav in uavs:
                uav.follow_cluster_list = []
            user_pos, cluster_pos, cluster_dir = \
                generate_positions(env, False, config.steps)
            trigger_steps = compute_trigger_steps(
                env, uavs, user_pos, cluster_pos, cluster_dir, config
            )

        deploy_idx = 0
        total_covered, total_com, total_step = 0, 0, 0
        env.reset()
        for uav in uavs:
            uav.position[:] = [0, 0]
            uav.current_battery_capacity = uav.total_battery_capacity
            uav.follow_cluster = None
            uav.cluster_coverage_time = 0
            uav.last_cluster_id = None
            uav.macro_fly_time = 0

        for step in range(config.steps):
            env.set_positions_from_array(
                user_pos[step], cluster_pos[step], cluster_dir[step]
            )
            if step in trigger_steps:
                deploy_idx = deploy_uavs_at_trigger_step(env, uavs, deploy_idx)

            for uav in uavs:
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                state = uav.get_state()
                action = agent.choose_action(state)
                next_state, reward, done, covered, com = uav.step(action)
                agent.store_transition((state, action, reward, next_state, float(done)))

                if episode <= config.episodes / 2:
                    agent.train()
                elif step % config.train_freq_late == 0:
                    agent.train()

                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue
                total_covered += covered
                total_com += com

            if step % config.target_update_interval == 0:
                agent.update_target()
            if (step + 1) % config.epsilon_decay_interval == 0:
                agent.epsilon = max(agent.epsilon * agent.epsilon_decay,
                                    agent.epsilon_min)

        jain_index = calculate_jain_index(env, total_step)
        users_per_cluster = config.all_user_num / config.cluster_num
        covered_rate = total_covered / (total_step * users_per_cluster)
        covered_rate_list.append(covered_rate)
        jain_index_list.append(jain_index)
        total_com_list.append(total_com * jain_index)

        if episode % config.log_interval == 0:
            print(
                f"Episode : {episode}, "
                f"epsilon : {agent.epsilon:.4f} "
                f"total_covered : {total_covered} "
                f"total_com: {total_com} "
                f"cover_rate：{covered_rate:.4f} "
                f"jain_index：{jain_index:.4f}"
            )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "drl_tpwsp_model.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list
