# Macro DDQN 宏观调度设计文档

## 1. 核心思路

将集中式的匈牙利分配替换为**分布式 DDQN**：每架 UAV 独立运行一个 DDQN 模型，在每个决策步观察全局状态，决定 **keep（原地服务）** 还是 **switch to top-N（切换到高收益集群）**。

```
匈牙利:    全局最优匹配（集中式，一次决策3架UAV）
Macro DDQN: 各UAV独立决策（分布式，同构共享模型参数）
```

## 2. 决策节奏

| 阶段 | 决策间隔 | 最小停留 | 首次分配 |
|------|---------|---------|---------|
| 训练 | 每 1 步（stay 外） | 10 步 | 各 UAV 独立选 top-1 |
| 推理 | 每 10 步 | 10 步 | 匈牙利保证互斥 |

切换后强制停留 MIN_STAY 步，停留期内仅累加 reward，不做决策。

## 3. State（15 维）

对每架 UAV，为其 6 个集群中的 3 个最佳候选提取特征：

```
top-3 候选 × 5 维 = 15 维:

[score_norm]    集群等待时间 / max_score     ← 等越久越大，越该去
[dist_benefit]  1 - dist/max_dist_uav        ← UAV 越近越大
[is_current]    0/1, 是否我正服务的集群      ← keep 语义
[covered_by_other] 0/1, 是否已被兄弟覆盖    ← 避让机制
[density_norm]  用户数 / max_users           ← 密度高的覆盖价值大
```

候选按**匈牙利 benefit 公式**排序：`0.5 × score_norm + 0.5 × dist_benefit`。排除已被兄弟 UAV 覆盖的集群，当前集群强制保留在候选列表。

## 4. Action（4 个离散动作）

| Action | 含义 | 执行 |
|--------|------|------|
| 0 | keep | 保持当前集群，继续服务 |
| 1 | switch to top-1 | 切换到 benefit 最高的候选集群 |
| 2 | switch to top-2 | 切换到 benefit 次高的候选集群 |
| 3 | switch to top-3 | 切换到 benefit 第三的候选集群 |

训练时瞬移到目标集群中心，推理时通过 `execute_macro_switch` 计算宏飞行距离。

## 5. Score 更新 — 自然驱动切换

每个 step 对 6 个集群的 score 做如下更新：

```
被覆盖的集群: score = max(0, score - 2)    ← 每步 -2，正在被服务的集群
未被覆盖的集群: score += 1                   ← 每步 +1，等待中的集群
```

**Score 如何驱动切换？**

| 时刻 | UAV 所在集群 C0 | 空闲集群 C3 |
|------|-----------|------------|
| 切换瞬间 | score=30, score_norm=0.5 | score=100, score_norm=1.0 |
| 服务 5 步后 | score=20, score_norm=0.33 | score=105, score_norm=1.0 |
| 服务 15 步后 | score=0, score_norm=0 | score=115, score_norm=1.0 |

在服务过程中 C0 的 score 缓慢从 30 降到 0，而 C3 从 100 涨到 115。DDQN 的 state 看到 C0 的 score_norm 降到了 0、C3 的 score_norm 是 1.0，切换带来的 benefit 跳升足够大，自然做出 switch 决策。

**核心设计**：不设显式的"你该走了"惩罚，而是通过 score 的自然衰减让 DDQN 自己判断"什么时候该切"。这是和匈牙利算法**完全同构**的——匈牙利的 benefit 矩阵也是用 score 和 distance 算的。

## 6. Reward — 完全对齐匈牙利

```python
# 统一公式, 不区分 keep/switch:
reward = 0.5 * score_norm + 0.5 * dist_benefit

# keep:
#   dist_benefit = 1.0 (已在集群中心)
#   score_norm 随服务逐步衰减
#   → reward 从高到低自然下降

# switch:
#   dist_benefit = 1 - 实际距离/max_dist (考虑飞行成本)
#   score_norm = 目标集群的当前分值 (锁定切换瞬间)
#   → reward 一次性跳升
```

无人为缩放、无切换惩罚。DDQN 直接最大化匈牙利 benefit 的累积和。

## 7. 训练流程

```
每个 episode (300 步):
  env.randomize()         ← 换环境配置
  每个 step:
    env.step(False)       ← 集群确定性移动
    更新 cluster.score    ← 被覆盖-2, 未覆盖+1

    每个 UAV (不在 stay 中的):
      构建 state → 计算 top3 候选
      首次分配: 直接选 top-1
      epsilon-greedy 决策:
        keep → reward = 0.5*sn + 0.5*1.0, push transition
        switch → 锁定目标分数, reward = 0.5*sn + 0.5*db, 瞬移, 进入 min_stay

      停留期内: 每步累加 reward, 不做决策, 不 push
      停留结束: push (state_before, action, sum_rewards, state_now)

    每步 DDQN.train()  ← 在线更新
  
  epsilon 逐 episode 衰减 (1.0 → 0.05)
```

**关键特性**：
- 3 架 UAV 共享同一个 replay buffer 和 DDQN 模型
- 训练时瞬移（不计宏飞行时间），让 DDQN 专注学"何时切、切去哪"
- 停留期延迟 reward 推送，DDQN 看到完整服务周期的总收益

## 8. 推理流程

```
Step 0: 匈牙利全局最优首次分配（保证互斥）
Step 10, 20, 30, ...:
  每架 UAV 独立:
    停留中 → 保持, 不加决策
    不在停留 → DDQN 推理选 action
      keep → 继续当前集群
      switch → 宏飞行到目标, 重新进入 min_stay=10
```

## 9. 和匈牙利的关系

| 维度 | 匈牙利 | Macro DDQN |
|------|--------|------------|
| 决策方式 | 集中式、全局最优 | 分布式、独立决策 |
| 目标函数 | 最大化 benefit 矩阵 | 最大化匈牙利 benefit |
| 触发 | 每 30 步强制触发 | 每 10 步 DDQN 自主判断 |
| 冷却 | 无 | min_stay = 10 步 |
| 训练 | 无（确定性算法） | 在线 RL（epsilon-greedy） |

Macro DDQN 训练时用的是匈牙利的 benefit 公式做 reward，训练目标**不是**超越匈牙利，而是**逼近匈牙利的决策质量**，同时在分布式执行中获得更好的可扩展性。

## 10. 当前性能差距分析

从 60% → 目标 90%+，可能瓶颈：

1. **训练不充分** — 500 episode × 300 步 × 3 UAV = 45 万决策中 DDQN 只训练了一小部分。考虑增大 episode 数或提高训练频率
2. **首次分配冲突** — 训练时 3 架 UAV 各自独立选 top-1，可能选到同一集群。改为训练时首次也用匈牙利分配
3. **停留期数据稀疏** — MIN_STAY=10 意味着每 UAV 每 episode 只有 ~30 条完整 transition，可考虑降低到 5 步
4. **state 缺少自身信息** — 当前 state 只有 top-3 候选的特征,没有 UAV 自己的位置/电量/当前集群等待时间。加回这些特征可以让 DDQN 做出更精准的 keep vs switch 判断
