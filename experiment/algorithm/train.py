import os
import numpy as np
from experiment.utils import calculate_jain_index
from experiment.algorithm.trigger import try_trigger_deployment


# ============================================================
#  Agent 类型检测
# ============================================================
def _agent_type(agent):
    name = agent.__class__.__name__
    if 'PPO' in name:
        return 'ppo'
    elif 'DDPG' in name:
        return 'ddpg'
    elif 'DQN' in name:
        return 'dqn'
    return 'unknown'


# ============================================================
#  通用工具
# ============================================================
def _reset_uavs(uavs):
    for uav in uavs:
        uav.position[:] = [0, 0]
        uav.current_battery_capacity = uav.total_battery_capacity
        uav.follow_cluster = None
        uav.follow_cluster_list = []
        uav.cluster_coverage_time = 0
        uav.last_cluster_id = None
        uav.macro_fly_time = 0


def _log_episode(episode, agent, total_covered, total_com, covered_rate, jain_index):
    eps = getattr(agent, 'epsilon', float('nan'))
    print(
        f"Episode : {episode}, "
        f"epsilon : {eps:.4f} "
        f"total_covered : {total_covered} "
        f"total_com: {total_com} "
        f"cover_rate：{covered_rate:.4f} "
        f"jain_index：{jain_index:.4f}"
    )


# ============================================================
#  DQN 训练
# ============================================================
def run_training_dqn(env, uavs, agent, config, save_dir="models"):
    covered_rate_list, total_com_list, jain_index_list = [], [], []

    for episode in range(1, config.episodes - 1):
        deploy_idx = 0
        total_covered, total_com, total_step = 0, 0, 0

        if (episode - 1) % config.rerandomize_interval == 0:
            env.randomize()
            for uav in uavs:
                uav.follow_cluster_list = []

        env.reset()
        _reset_uavs(uavs)

        for step in range(config.steps):
            env.step(False)
            deploy_idx, _ = try_trigger_deployment(
                env, uavs, step, deploy_idx, config)

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
                agent.store_transition(
                    (state, action, reward, next_state, float(done)))

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
                agent.epsilon = max(
                    agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        jain_index = calculate_jain_index(env, total_step)
        users_per_cluster = config.all_user_num / config.cluster_num
        covered_rate = total_covered / max(total_step * users_per_cluster, 1)
        covered_rate_list.append(covered_rate)
        jain_index_list.append(jain_index)
        total_com_list.append(total_com * jain_index)

        if episode % config.log_interval == 0:
            _log_episode(episode, agent, total_covered, total_com,
                         covered_rate, jain_index)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "drl_tpwsp_dqn.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list


# ============================================================
#  PPO 训练
# ============================================================
def run_training_ppo(env, uavs, agent, config, save_dir="models"):
    covered_rate_list, total_com_list, jain_index_list = [], [], []

    for episode in range(1, config.episodes - 1):
        deploy_idx = 0
        total_covered, total_com, total_step = 0, 0, 0

        if (episode - 1) % config.rerandomize_interval == 0:
            env.randomize()
            for uav in uavs:
                uav.follow_cluster_list = []

        env.reset()
        _reset_uavs(uavs)

        for step in range(config.steps):
            env.step(False)
            deploy_idx, _ = try_trigger_deployment(
                env, uavs, step, deploy_idx, config)

            for i, uav in enumerate(uavs):
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                agent.set_uav_index(i)
                state = uav.get_state()
                action = agent.choose_action(state)
                next_state, reward, done, covered, com = uav.step(action)
                agent.store_transition(
                    (state, action, reward, next_state, float(done)))

                agent.train()

                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue
                total_covered += covered
                total_com += com

        agent.end_episode(len(uavs))

        jain_index = calculate_jain_index(env, total_step)
        users_per_cluster = config.all_user_num / config.cluster_num
        covered_rate = total_covered / max(total_step * users_per_cluster, 1)
        covered_rate_list.append(covered_rate)
        jain_index_list.append(jain_index)
        total_com_list.append(total_com * jain_index)

        if episode % config.log_interval == 0:
            _log_episode(episode, agent, total_covered, total_com,
                         covered_rate, jain_index)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "drl_tpwsp_ppo.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list


# ============================================================
#  DDPG 训练
# ============================================================
def run_training_ddpg(env, uavs, agent, config, save_dir="models"):
    covered_rate_list, total_com_list, jain_index_list = [], [], []

    for episode in range(1, config.episodes - 1):
        deploy_idx = 0
        total_covered, total_com, total_step = 0, 0, 0

        if (episode - 1) % config.rerandomize_interval == 0:
            env.randomize()
            for uav in uavs:
                uav.follow_cluster_list = []

        env.reset()
        _reset_uavs(uavs)

        for step in range(config.steps):
            env.step(False)
            deploy_idx, _ = try_trigger_deployment(
                env, uavs, step, deploy_idx, config)

            for uav in uavs:
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                state = uav.get_state()
                action = agent.choose_action(state)
                next_state, reward, done, covered, com = uav.step_continuous(action)
                agent.store_transition(
                    (state, action, reward, next_state, float(done)))

                if episode <= config.episodes / 2:
                    agent.train()
                elif step % config.train_freq_late == 0:
                    agent.train()

                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue
                total_covered += covered
                total_com += com

        jain_index = calculate_jain_index(env, total_step)
        users_per_cluster = config.all_user_num / config.cluster_num
        covered_rate = total_covered / max(total_step * users_per_cluster, 1)
        covered_rate_list.append(covered_rate)
        jain_index_list.append(jain_index)
        total_com_list.append(total_com * jain_index)

        if episode % config.log_interval == 0:
            _log_episode(episode, agent, total_covered, total_com,
                         covered_rate, jain_index)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "drl_tpwsp_ddpg.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list


# ============================================================
#  统一入口（向后兼容）
# ============================================================
def run_training(env, uavs, agent, config, save_dir="models"):
    at = _agent_type(agent)
    if at == 'ppo':
        return run_training_ppo(env, uavs, agent, config, save_dir)
    elif at == 'ddpg':
        return run_training_ddpg(env, uavs, agent, config, save_dir)
    else:
        return run_training_dqn(env, uavs, agent, config, save_dir)
