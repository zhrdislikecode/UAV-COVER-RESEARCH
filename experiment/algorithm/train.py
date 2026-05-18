import os
import numpy as np
from experiment.utils import calculate_jain_index
from experiment.algorithm.trigger import try_trigger_deployment as _hungarian_trigger
from experiment.ppo import make_ppo_state
from experiment.ddpg import make_ddpg_state


# ============================================================
#  宏观调度选择
# ============================================================
_SCHEDULER_MAP = {
    'hungarian':  None,  # 用默认的 _hungarian_trigger
    'gcn':         None,  # 懒加载
    'macro_ddqn':  None,
}


def _get_trigger_fn(macro_scheduler='hungarian'):
    """根据参数返回触发函数（训练时自动注入 training=True）"""
    if macro_scheduler == 'hungarian':
        base_fn = _hungarian_trigger
        return lambda env, uavs, step, d_idx, cfg: \
            base_fn(env, uavs, step, d_idx, cfg, training=True)
    elif macro_scheduler == 'macro_ddqn':
        from experiment.algorithm.trigger_macro_ddqn import \
            try_trigger_deployment_macro_ddqn
        return try_trigger_deployment_macro_ddqn
    elif macro_scheduler == 'greedy':
        from experiment.algorithm.trigger_greedy import \
            try_trigger_deployment_greedy
        return try_trigger_deployment_greedy
    else:
        raise ValueError(f"Unknown macro_scheduler: {macro_scheduler}, "
                         f"choose from 'hungarian'|'macro_ddqn'|'greedy'")


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
def run_training_dqn(env, uavs, agent, config, save_dir="models", macro_scheduler='hungarian'):
    trigger_fn = _get_trigger_fn(macro_scheduler)
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
            deploy_idx, _ = trigger_fn(
                env, uavs, step, deploy_idx, config)

            for uav in uavs:
                if hasattr(uav, 'steps_since_last_switch'):
                    uav.steps_since_last_switch += 1
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                state = uav.get_state(uavs)
                action = agent.choose_action(state)
                _, reward, done, covered, com = uav.step(action)
                next_state = uav.get_state(uavs)
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
    save_path = os.path.join(save_dir, f"drl_tpwsp_dqn_uav_{len(uavs)}.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list


# ============================================================
#  PPO 训练
# ============================================================
def run_training_ppo(env, uavs, agent, config, save_dir="models", macro_scheduler='hungarian'):
    trigger_fn = _get_trigger_fn(macro_scheduler)
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
            deploy_idx, _ = trigger_fn(
                env, uavs, step, deploy_idx, config)

            for i, uav in enumerate(uavs):
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                agent.set_uav_index(i)
                state = make_ppo_state(uav, uavs)
                action = agent.choose_action(state)
                next_state, reward, done, covered, com = uav.step(action)
                # PPO reward shaping: 连续距离惩罚，引导 UAV 紧贴集群中心
                if uav.follow_cluster is not None:
                    dist_to_center = float(np.linalg.norm(
                        uav.position[:2] - uav.follow_cluster.center[:2]))
                    reward -= 0.5 * dist_to_center
                agent.store_transition(
                    (state, action, reward, make_ppo_state(uav, uavs), float(done)))

                if uav.current_battery_capacity <= 0:
                    uav.is_consume_energy = True
                    continue
                total_covered += covered
                total_com += com

            # 每步训练一次（PPO 内部 horizon 守卫，buffer 不够时直接返回）
            if step % config.train_freq_late == 0:
                agent.train()

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
    save_path = os.path.join(save_dir, f"drl_tpwsp_ppo_uav_{len(uavs)}.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list


# ============================================================
#  DDPG 训练
# ============================================================
def run_training_ddpg(env, uavs, agent, config, save_dir="models", macro_scheduler='hungarian'):
    trigger_fn = _get_trigger_fn(macro_scheduler)
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
            deploy_idx, _ = trigger_fn(
                env, uavs, step, deploy_idx, config)

            for uav in uavs:
                if hasattr(uav, 'steps_since_last_switch'):
                    uav.steps_since_last_switch += 1
                if uav.macro_fly_time > 0:
                    uav.macro_fly_time -= 1
                    continue
                if uav.current_battery_capacity <= 0:
                    continue
                total_step += 1
                state = make_ddpg_state(uav, uavs)
                action = agent.choose_action(state)
                _, reward, done, covered, com = uav.step_continuous(action)
                # DDPG reward shaping: 连续距离惩罚，引导 UAV 贴紧集群中心
                next_state = make_ddpg_state(uav, uavs)
                if uav.follow_cluster is not None:
                    dist_to_center = float(np.linalg.norm(
                        uav.position[:2] - uav.follow_cluster.center[:2]))
                    reward -= 0.5 * dist_to_center
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

        # 每个 episode 结束时衰减噪声（而非每次 train 都衰减）
        agent.decay_noise()

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
    save_path = os.path.join(save_dir, f"drl_tpwsp_ddpg_uav_{len(uavs)}.pth")
    agent.save_model(save_path)
    print(f"Model saved to: {save_path}")

    return covered_rate_list, total_com_list, jain_index_list


# ============================================================
#  统一入口（向后兼容）
# ============================================================
def run_training(env, uavs, agent, config, save_dir="models", macro_scheduler='hungarian'):
    at = _agent_type(agent)
    if at == 'ppo':
        return run_training_ppo(env, uavs, agent, config, save_dir, macro_scheduler=macro_scheduler)
    elif at == 'ddpg':
        return run_training_ddpg(env, uavs, agent, config, save_dir, macro_scheduler=macro_scheduler)
    else:
        return run_training_dqn(env, uavs, agent, config, save_dir, macro_scheduler=macro_scheduler)
