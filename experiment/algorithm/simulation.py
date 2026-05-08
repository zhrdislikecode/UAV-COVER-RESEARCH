"""
仿真实验：30 个不同环境，评估三种 RL agent + 匈牙利调度，输出平均指标
"""
import numpy as np
from experiment.config import EnvConfig, TrainConfig, DQNConfig
from experiment.algorithm.evaluate import run_evaluation

np.random.seed(42)

N_RUNS = 30
AGENTS = ['ppo', 'dqn', 'ddpg']
SCHEDULER = 'hungarian'


def run():
    env_config = EnvConfig()
    train_config = TrainConfig()
    dqn_config = DQNConfig()

    model_names = {'dqn': 'drl_tpwsp_dqn.pth', 'ppo': 'drl_tpwsp_ppo.pth',
                   'ddpg': 'drl_tpwsp_ddpg.pth'}

    results = {a: {'value': [], 'total_com': [], 'jain_index': []} for a in AGENTS}

    for run_id in range(1, N_RUNS + 1):
        seed = 100 + run_id
        np.random.seed(seed)
        print(f"\n{'='*60}")
        print(f"  Run {run_id}/{N_RUNS} (seed={seed})")
        print(f"{'='*60}")

        for idx, agent_type in enumerate(AGENTS):
            model_path = f"models/{model_names[agent_type]}"
            value, total_com, jain_index = run_evaluation(
                env_config, train_config, dqn_config,
                model_path=model_path,
                randomize=(idx == 0),   # 仅第一个 agent 随机化环境
                agent_type=agent_type,
                macro_scheduler=SCHEDULER,
            )
            results[agent_type]['value'].append(value)
            results[agent_type]['total_com'].append(total_com)
            results[agent_type]['jain_index'].append(jain_index)

    # ── 输出平均 ──
    print(f"\n{'='*60}")
    print(f"  Results (avg over {N_RUNS} runs)")
    print(f"{'='*60}")
    print(f"{'Agent':>8s}  {'value':>12s}  {'total_com':>12s}  {'jain_index':>12s}")
    print(f"{'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")

    for agent_type in AGENTS:
        avg_v = np.mean(results[agent_type]['value'])
        avg_c = np.mean(results[agent_type]['total_com'])
        avg_j = np.mean(results[agent_type]['jain_index'])
        print(f"{agent_type:>8s}  {avg_v:12.4f}  {avg_c:12.4f}  {avg_j:12.4f}")


if __name__ == "__main__":
    run()
