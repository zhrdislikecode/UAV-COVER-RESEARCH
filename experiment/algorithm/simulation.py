"""
仿真实验：N 个不同环境，评估三种 RL agent + 匈牙利调度，输出平均指标

用法:
  python -m experiment.algorithm.simulation
  python -m experiment.algorithm.simulation --runs 20 --clusters 6 --users 30 --uavs 3
  python -m experiment.algorithm.simulation -n 10 -c 4 -u 25 -v 2
"""
import numpy as np
from experiment.config import EnvConfig, TrainConfig, DQNConfig
from experiment.algorithm.evaluate import run_evaluation


def run(n_runs=30, cluster_num=6, users_per_cluster=30, uav_num=3,
        steps=300, scheduler='hungarian', agents=None):
    """仿真实验

    Args:
        n_runs: 环境数量
        cluster_num: 集群数
        users_per_cluster: 每集群用户数
        uav_num: 无人机数
        scheduler: 宏观调度器
        agents: RL agent 列表，默认 ['ppo', 'dqn', 'ddpg']
    """
    if agents is None:
        agents = ['ppo', 'dqn', 'ddpg']

    env_config = EnvConfig()
    env_config.cluster_num = cluster_num
    env_config.users_per_cluster = users_per_cluster
    env_config.uav_num = uav_num

    train_config = TrainConfig()
    train_config.steps = steps
    train_config.all_user_num = cluster_num * users_per_cluster

    dqn_config = DQNConfig()

    model_names = {'dqn': 'drl_tpwsp_dqn.pth', 'ppo': 'drl_tpwsp_ppo.pth',
                   'ddpg': 'drl_tpwsp_ddpg.pth'}

    print(f"Config: {n_runs} runs, {cluster_num} clusters, "
          f"{users_per_cluster} users/cluster, {uav_num} UAVs, "
          f"steps={steps}, scheduler={scheduler}")

    results = {a: {'value': [], 'total_com': [], 'jain_index': []} for a in agents}

    for run_id in range(1, n_runs + 1):
        seed = 100 + run_id
        np.random.seed(seed)
        print(f"\n{'='*60}")
        print(f"  Run {run_id}/{n_runs} (seed={seed})")
        print(f"{'='*60}")

        for idx, agent_type in enumerate(agents):
            model_path = f"models/{model_names[agent_type]}"
            value, total_com, jain_index = run_evaluation(
                env_config, train_config, dqn_config,
                model_path=model_path,
                randomize=(idx == 0),
                agent_type=agent_type,
                macro_scheduler=scheduler,
            )
            results[agent_type]['value'].append(value)
            results[agent_type]['total_com'].append(total_com)
            results[agent_type]['jain_index'].append(jain_index)

    # ── 输出平均 ──
    print(f"\n{'='*60}")
    print(f"  Results (avg over {n_runs} runs)")
    print(f"{'='*60}")
    print(f"{'Agent':>8s}  {'value':>12s}  {'total_com':>12s}  {'jain_index':>12s}")
    print(f"{'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")

    for agent_type in agents:
        avg_v = np.mean(results[agent_type]['value'])
        avg_c = np.mean(results[agent_type]['total_com'])
        avg_j = np.mean(results[agent_type]['jain_index'])
        print(f"{agent_type:>8s}  {avg_v:12.4f}  {avg_c:12.4f}  {avg_j:12.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--runs", type=int, default=5)
    parser.add_argument("-c", "--clusters", type=int, default=1)
    parser.add_argument("-u", "--users", type=int, default=30)
    parser.add_argument("-v", "--uavs", type=int, default=1)
    parser.add_argument("-t", "--steps", type=int, default=450)
    parser.add_argument("-s", "--scheduler", type=str, default="hungarian")
    args = parser.parse_args()

    run(n_runs=args.runs, cluster_num=args.clusters,
        users_per_cluster=args.users, uav_num=args.uavs,
        steps=args.steps, scheduler=args.scheduler)
