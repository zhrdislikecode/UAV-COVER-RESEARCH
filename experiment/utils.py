from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_uav_trajectory_3d(uav_position, cluster_position, uav_height=10,
                           smooth=True, dash_interval=30, pois=None):
    font = font_manager.FontProperties(family='Arial', weight='bold', size=16)

    uav_position[:, :, 0:2] *= 100
    cluster_position[:, :, 0:2] *= 100
    T, m, _ = uav_position.shape
    _, n, _ = cluster_position.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 颜色映射#1f77b4
    uav_colors = ['red', '#1f77b4', '#9467bd', '#8c564b', '#7f7f7f']
    cluster_colors = ['#ff69b4', 'orange', 'purple', '#2ca02c', 'gray']

    # ---------- 绘制无人机轨迹 ----------
    coverage_radius = 100  # UAV 覆盖半径

    for i in range(m):
        x = uav_position[:, i, 0]
        y = uav_position[:, i, 1]

        if smooth and len(x) > 3:
            t = np.linspace(0, 1, len(x))
            t_smooth = np.linspace(0, 1, len(x) * 5)
            spline_x = make_interp_spline(t, x, k=3)(t_smooth)
            spline_y = make_interp_spline(t, y, k=3)(t_smooth)
        else:
            spline_x, spline_y = x, y

        # 从 (0,0,10) 到第一个点虚线
        ax.plot([0, x[0]], [0, y[0]], [10, uav_height],
                color=uav_colors[i % len(uav_colors)],
                linewidth=1.5,
                linestyle='--')

        for t in range(1, T):
            ax.plot([x[t - 1], x[t]], [y[t - 1], y[t]], [uav_height, uav_height],
                    color=uav_colors[i % len(uav_colors)],
                    linewidth=1.5,
                    linestyle='--')

        interval = 5
        if uav_colors[i % len(uav_colors)] == 'red':
            marker_style = 'x'
            marker_color = 'red'
        elif uav_colors[i % len(uav_colors)] == '#1f77b4':
            marker_style = '^'
            marker_color = '#1f77b4'
        else:
            marker_style = 'o'
            marker_color = uav_colors[i % len(uav_colors)]

        ax.scatter(
            x[::interval], y[::interval], np.full(len(x[::interval]), uav_height),
            color=marker_color, marker=marker_style, s=60, alpha=1
        )

        circle_theta = np.linspace(0, 2 * np.pi, 100)
        for t in range(0, T, interval):
            cx, cy = x[t], y[t]
            circle_x = cx + coverage_radius * np.cos(circle_theta)
            circle_y = cy + coverage_radius * np.sin(circle_theta)
            cz = np.zeros_like(circle_theta)
            ax.plot(circle_x, circle_y, cz,
                    color=marker_color,
                    linestyle='--',
                    linewidth=1)

    num_users_per_cluster = 30
    user_range = 50
    user_interval = 15

    for j in range(n):
        ax.plot(cluster_position[:, j, 0],
                cluster_position[:, j, 1],
                np.zeros(T),
                color=cluster_colors[j % len(cluster_colors)],
                linestyle='--',
                linewidth=2.5)

        for t in range(T):
            if t % user_interval == 0:
                cx, cy = cluster_position[t, j, 0], cluster_position[t, j, 1]
                ux = cx + np.random.uniform(-user_range, user_range, num_users_per_cluster)
                uy = cy + np.random.uniform(-user_range, user_range, num_users_per_cluster)
                uz = np.zeros(num_users_per_cluster)
                ax.scatter(ux, uy, uz, color='k', marker='o', s=1, alpha=0.6)

    if pois is not None:
        for px, py in pois:
            ax.scatter(px * 100, py * 100, 0,
                      c='red', marker='x', s=80, linewidths=2)

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_zlabel("Z", fontsize=14)

    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1500)
    ax.set_xticks(np.arange(0, 1500, 300))
    ax.set_yticks(np.arange(0, 1500, 300))
    ax.set_zlim(0, uav_height + 2)

    ax.tick_params(axis='both', which='major', labelsize=16)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()

    # ---------- 保存为SVG ----------
    # plt.rcParams['svg.fonttype'] = 'none'
    # plt.savefig("uav_trajectory.svg", format='svg', bbox_inches='tight', dpi=600)
    # print("✅ 图像已保存至当前目录: uav_trajectory.svg")

    plt.show()

def generate_positions(env_template, flag, steps):
    env_template.reset()
    for cluster in env_template.clusters:
        cluster.path_index = 0

    env = env_template
    total_users = sum(len(cluster.users) for cluster in env.clusters)
    num_clusters = len(env.clusters)
    all_user_positions = np.zeros((steps, total_users, 2), dtype=np.float64)
    all_cluster_positions = np.zeros((steps, num_clusters, 2), dtype=np.float64)
    all_cluster_directions = np.zeros((steps, num_clusters, 2), dtype=np.float64)

    for cluster in env.clusters:
        cluster.path_index = 0

    for step in range(steps):
        env.step(flag)
        user_idx = 0
        cluster_idx = 0
        for cluster in env.clusters:
            all_cluster_positions[step][cluster_idx] = cluster.center
            all_cluster_directions[step][cluster_idx] = cluster.direction
            cluster_idx += 1
            for user in cluster.users:
                all_user_positions[step, user_idx, :] = user.position[:2]
                user_idx += 1
    return all_user_positions, all_cluster_positions, all_cluster_directions

def plot_cluster_paths(all_cluster_positions):
    steps, num_clusters, _ = all_cluster_positions.shape

    plt.figure(figsize=(8, 6))
    for cluster_idx in range(num_clusters):
        x = all_cluster_positions[:, cluster_idx, 0]
        y = all_cluster_positions[:, cluster_idx, 1]
        plt.plot(x, y, marker='o', markersize=1, label=f'Cluster {cluster_idx}')

    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.title("各集群中心轨迹")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_positions(positions, cluster_centers=None, uav_positions=None, title="Positions", count=0, pois=None):
    interest_points = np.array([
        [1, 1],
        [7, 1],
        [14, 1],
        [1, 13],
        [7, 13],
        [14, 13],
        [4, 5],
        [4, 9],
        [10, 5],
        [10, 9],
        [7, 7]
    ])

    x = positions[:, 0]
    y = positions[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='black', s=10, marker='o', label='Users')

    if cluster_centers is not None:
        cx = cluster_centers[:, 0]
        cy = cluster_centers[:, 1]
        plt.scatter(cx, cy, c='red', s=50, marker='x', label='Cluster Centers')

    if uav_positions is not None:
        ux = uav_positions[:, 0]
        uy = uav_positions[:, 1]
        plt.scatter(ux, uy, c='blue', s=60, marker='v', label='UAVs')

    plt.scatter(interest_points[:, 0], interest_points[:, 1],
                c='green', s=40, marker='s', label='Interest Points')


    plt.title(f"Allocation Round {count}")
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xlim(0, 15)
    plt.ylim(0, 15)

    plt.xticks(np.linspace(0, 15, 16))
    plt.yticks(np.linspace(0, 15, 16))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_coverage_rate(coverage_rates):
    episodes = list(range(1, len(coverage_rates) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, coverage_rates, marker='o',  markersize=1, linestyle='-', color='black', label='Coverage Rate')
    plt.xlabel('Episode')
    plt.ylabel('Coverage Rate')
    plt.title('UAV Coverage Rate per Episode')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.show()

def calculate_jain_index(env, steps):
    cover_ratios = []
    for cluster in env.clusters:
        for ue in cluster.users:
            cover_ratios.append(ue.cover_num / steps)
    numerator = sum(cover_ratios) ** 2
    denominator = len(cover_ratios) * sum(r ** 2 for r in cover_ratios)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def plot_train_result(data_list, title="覆盖率变化图"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data_list)), data_list, marker='.', linestyle='-', color='k', markersize=4)
    plt.xlabel("时隙")
    plt.ylabel("覆盖率")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 当所有无人机没电了则停止
def check_is_over(uav_list):
    for uav in uav_list:
        if uav.current_battery_capacity > 0:
            return False
    return True

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    cos_theta = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def plot_parameter_comparison(value, com, jain, energy, algo_labels=None, save_path="figure.svg"):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.unicode_minus'] = False

    num_params = 4
    num_algos = len(value)

    if algo_labels is None:
        algo_labels = ['DRL-TRWSP', 'DWTD', 'Greedy', 'Random']

    param_labels = [r'$F(S)$', r'$D(S)$', r'$J(S)$', r'$E(S)$']
    x = np.arange(num_params)
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(num_algos):
        values = [value[i], com[i], jain[i], energy[i]]
        ax.bar(x + (i - num_algos / 2) * width + width / 2,
               values, width, label=algo_labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, fontsize=16)

    ax.set_ylim(0, 1)
    ax.set_ylabel('Normalized Performance Metric', fontsize=14)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.legend(fontsize=13, loc='upper right')

    plt.tight_layout()
    fig.savefig(save_path, format='svg')
    plt.show()