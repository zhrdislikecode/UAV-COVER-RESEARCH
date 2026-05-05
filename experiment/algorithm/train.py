import os
import numpy as np
from experiment.utils import calculate_jain_index
from experiment.algorithm.trigger import try_trigger_deployment


def _is_continuous(agent):
    """检测是否为连续动作 agent（DDPG）"""
    return hasattr(agent, 'action_dim') and agent.action_dim == 2


def run_training(env, uavs, agent, config, save_dir="models"):
    """训练通用模型：每轮自动随机化环境，支持 DQN / PPO / DDPG"""
    covered_rate_list, total_com_list, jain_index_list = [], [], []
    continuous = _is_continuous(agent)

    for episode in range(1, config.episodes - 1):
        deploy_idx = 0
        total_covered, total_com, total_step = 0, 0, 0

        # 每隔 rerandomize_interval 轮重新随机化环境
        if (episode - 1) % config.rerandomize_interval == 0:
            env.randomize()
            for uav in uavs:
                uav.follow_cluster_list = []

        env.reset()
        for uav in uavs:
            uav.position[:] = [0, 0]
            uav.current_battery_capacity = uav.total_battery_capacity
            uav.follow_cluster = None
            uav.follow_cluster_list = []
            uav.cluster_coverage_time = 0
            uav.last_cluster_id = None
            uav.macro_fly_time = 0

        for step in range(config.steps):
            env.step(False)  # 确定性移动

            # 在线匈牙利触发检查 + 部署
            deploy_idx, _ = try_trigger_deployment(
                env, uavs, step, deploy_idx, config
            )

            for uav in uavs:
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                state = uav.get_state()
                action = agent.choose_action(state)

                if continuous:
                    next_state, reward, done, covered, com = \
                        uav.step_continuous(action)
                else:
                    next_state, reward, done, covered, com = uav.step(action)

                agent.store_transition(
                    (state, action, reward, next_state, float(done))
                )

                # 训练（PPO 内部会累积到 horizon 再更新）
                if episode <= config.episodes / 2:
                    agent.train()
                elif step % config.train_freq_late == 0:
                    agent.train()

                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue
                total_covered += covered
                total_com += com

            # DQN 专属：target 更新 + epsilon 衰减
            if hasattr(agent, 'update_target'):
                if step % config.target_update_interval == 0:
                    agent.update_target()
            if hasattr(agent, 'epsilon'):
                if (step + 1) % config.epsilon_decay_interval == 0:
                    agent.epsilon = max(
                        agent.epsilon * agent.epsilon_decay, agent.epsilon_min
                    )

        jain_index = calculate_jain_index(env, total_step)
        users_per_cluster = config.all_user_num / config.cluster_num
        covered_rate = total_covered / max(total_step * users_per_cluster, 1)
        covered_rate_list.append(covered_rate)
        jain_index_list.append(jain_index)
        total_com_list.append(total_com * jain_index)

        if episode % config.log_interval == 0:
            eps = getattr(agent, 'epsilon', float('nan'))
            print(
                f"Episode : {episode}, "
                f"epsilon : {eps:.4f} "
                f"total_covered : {total_covered} "
                f"total_com: {total_com} "
                f"cover_rate：{covered_rate:.4f} "
                f"jain_index：{jain_index:.4f}"
            )

    os.makedirs(save_dir, exist_ok=True)
    # 根据 agent 类型命名模型文件
    if _is_continuous(agent):
        model_name = "drl_tpwsp_ddpg.pth"
    elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
        model_name = "drl_tpwsp_ppo.pth"
    else:
        model_name = "drl_tpwsp_dqn.pth"
    save_path = os.path.join(save_dir, model_name)
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list
