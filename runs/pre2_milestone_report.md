# Pre2 / Week 9 Milestone 报告草稿 — OOD 检测与选择性预测

日期：2026-10-25 ｜ 环境：AutoDL 单卡 RTX 3080 Ti (12GB)，Python 3.12，PyTorch 2.5.1+cu124，seed=42
代码：本地仓库 `ood-selective-prediction`（AutoDL 实例 `~/Documents/Trae-CN/default/ood-selective-prediction`，数据与 runs/ 不入库）

## 1. 问题与目标

在 CIFAR-10 / CIFAR-100 两个 ID 任务上训练 ResNet-18 分类器，比较 Standard 与 Energy-aware 两种训练方式、MSP 与 Energy 两种置信度分数（统一为"越大越像 ID"），在 95% 目标 coverage 下评估选择性预测，并在 SVHN、DTD 上测 OOD 检测（AUROC / AUPR / FPR@95）。

## 2. 数据与无泄漏划分（全部就绪）

- **ID**：CIFAR-10、CIFAR-100 官方 train 50,000 各自按 seed=42 划分 45,000 训练 / 5,000 验证（验证集不用增强）；官方 test 各 10,000 仅最终评估。
- **辅助异常**：Tiny ImageNet-200 仅 train split（100,000 张，64×64）。经 WordNet 类名与 CIFAR 类别人工语义对照（非自动），CIFAR-10 任务排除 23 类（剩 88,500 张）、CIFAR-100 任务排除 39 类（剩 80,500 张），清单逐条依据见 `data/tinyimagenet_exclude_cifar10.txt` / `_cifar100.txt`。
- **OOD 测试**：SVHN test（26,032 张）、DTD test（1,880 张 = 官方 partition-1：47 类×40），评估前不参与任何调参/选阈值。
- 数据说明：Stanford/Oxford 原站在实例内不可达，Tiny ImageNet/SVHN/DTD 改用 HuggingFace 镜像获取，SVHN/DTD 以 parquet 自定义 Dataset 加载，来源与校验记录见 `runs/DATA_PREPARATION.md`、`runs/data_preparation.json`。

## 3. 模型与训练

- 主干：CIFAR 版 ResNet-18（3×3 stem、无 maxpool），SGD lr=0.1、momentum 0.9、wd 5e-4、cosine 退火、20 epochs、batch 128。
- **Energy-aware 微调（主方法，对应开题承诺）**：从训练好的 Standard baseline 权重出发（`--init-checkpoint`），lr=0.01 微调 5 epochs；CE 保留，另加 margin-based energy separation loss：ReLU(E_in−(−5))² + ReLU((−1)−E_out)²，辅助样本（Tiny ImageNet 排除后）batch 与 ID batch 逐 iter 交替。energy_weight 仅用 **ID 验证集精度** 网格选定（0.05/0.02/0.01/0.005）：CIFAR-10 选 w=0.02（val 0.905）、CIFAR-100 选 w=0.005（val 0.690）——SVHN/DTD 全程未参与任何选择。
- 试点对照：相同 loss 从零训练 20 epochs（w=0.1），作为方法演进的负结果记录。所有配置逐项存于 `runs/*/config.json`。

## 4. 结果（均为实测，证据在 runs/）

### 4.1 主表：Standard vs Energy-aware(FT)

CIFAR-10（阈值来自 ID 验证集 95% coverage 分位数；w=0.02）：

| 设置 | ID acc | ECE | Sel.Err@95cov | SVHN AUROC | SVHN FPR95 | DTD AUROC | DTD FPR95 |
|---|---|---|---|---|---|---|---|
| Standard+MSP | **0.912** | **0.028** | **0.064** | **0.876** | **0.772** | **0.867** | 0.686 |
| Standard+Energy | 0.912 | 0.028 | 0.068 | 0.892 | 0.765 | 0.883 | **0.606** |
| Energy-aware(FT)+MSP | 0.895 | 0.093 | 0.076 | 0.765 | 0.913 | 0.796 | 0.773 |
| Energy-aware(FT)+Energy | 0.895 | 0.093 | 0.089 | 0.496 | 0.983 | 0.735 | 0.803 |

CIFAR-100（w=0.005）：

| 设置 | ID acc | ECE | Sel.Err@95cov | SVHN AUROC | SVHN FPR95 | DTD AUROC | DTD FPR95 |
|---|---|---|---|---|---|---|---|
| Standard+MSP | **0.721** | **0.070** | **0.251** | 0.609 | 0.887 | 0.711 | 0.867 |
| Standard+Energy | 0.721 | 0.070 | 0.254 | 0.707 | 0.907 | 0.745 | 0.861 |
| Energy-aware(FT)+MSP | 0.692 | 0.117 | 0.278 | **0.787** | **0.834** | **0.748** | 0.849 |
| Energy-aware(FT)+Energy | 0.692 | 0.117 | 0.286 | 0.728 | 0.986 | 0.698 | 0.938 |

### 4.2 试点：从零训练 energy-aware（w=0.1, 20 ep）——负结果

| 任务 | 指标 | Standard | Energy-aware(scratch) |
|---|---|---|---|
| CIFAR-10 | ID acc / ECE | 0.912 / 0.028 | 0.784 / 0.204 |
| CIFAR-10 | SVHN AUROC@MSP | 0.876 | 0.680 |
| CIFAR-100 | ID acc / ECE | 0.721 / 0.070 | 0.561 / 0.252 |
| CIFAR-100 | SVHN AUROC@MSP | 0.609 | 0.697 |

CE 与能量损失从零联合优化时目标冲突，精度与校准双双崩溃（ECE 0.20–0.25）；唯一亮点是 CIFAR-100 scratch+energy 的 SVHN AUROC 0.812，但同设置 DTD 0.596 全场最差，对 OOD 集不一致。这促成了主方法改为"先 Standard 后微调"。

### 4.3 主要观察

1. **微调保住了绝大部分精度，同时修好了 CIFAR-100 的 OOD 检测（正面结果）**：CIFAR-100 上 Energy-aware(FT)+MSP 以 −2.8 pt test acc（72.1→69.2）换 SVHN AUROC +17.8 pt（0.609→0.787，甚至高于 Standard+Energy 的 0.707）、DTD +3.7 pt；精度收益-检测收益的交易比单一调 score 显著得多（见 `figures/finetune_tradeoff.png` 的权衡曲线）。
2. **CIFAR-10 上同样做法是负结果（诚实报告）**：CIFAR-10 baseline 的 OOD 已经很强（SVHN 0.876），微调后 MSP 检测反而全面下降（SVHN 0.765、DTD 0.796），能量分数更是崩到 0.496（≈随机）。w 扫描（0.02→0.05）显示 acc 与 SVHN AUROC 在 C10 上同向变化、无免费权衡区间。
3. **微调后最优分数从 Energy 翻转为 MSP**：margin 损失把 ID 能量压向 −5、辅助能量推向 −1，logsumexp 的动态范围被压缩，能量分数近似只反映特征范数，对近邻 OOD（如 C100 的 SVHN，能量分数聚成单峰，FPR95 0.986）失效；而 MSP 仍保持与能量的相关性且更稳。这解释了主表中 FT 行的列间反转，是值得讨论的机制性发现。
4. **Standard 下结论与文献一致**：Energy 分数稳定优于 MSP（两任务、两 OOD 集一致，SVHN +1.6/+9.8 pt）；ECE 0.03–0.07 远好于任何 energy-aware 变体（0.09–0.12），精度-校准-检测三者不可兼得是当前 20+5 epoch 预算下的主要 trade-off。
5. **选择性预测**：95% coverage 下 Standard 的 sel-err 最低；CIFAR-100 全体设置 sel-err 都高（0.25–0.43），与 ID acc 一致，说明瓶颈在精度而非拒绝策略（见 `figures/risk_coverage.png`）。

## 5. 问题与限制

- CIFAR-10 微调负结果尚未解决：可能需要更小 w/温度调度/margin 调度，或只对能量分支做正则而非 CE 联合。
- 微调后 ECE 恶化（0.028→0.093、0.070→0.117）；温度缩放校准可作为零成本补救，尚未实施。
- DTD 原始 300×300 resize 到 32×32 会显著丢失纹理信息，可能影响其作为 OOD 的难度；保留原协议待讨论。
- Tiny ImageNet 语义排除为人工清单，存在边界类判定（bison/lion/teddy 等保留理由已记录）。

## 6. 下一步计划

1. Pre3：margin/温度调度 + 40–60 epochs（从零或更长微调），目标是把 C100 的 +17.8 SVHN 收益保持的同时把 acc 差距压到 −1 pt 以内，并解决 C10 负结果。
2. 温度缩放后处理校准（仅用 ID 验证集拟合 T），修复 energy-aware 变体 ECE。
3. 报告与图表定稿（主表、finetune_tradeoff、score_distributions、risk-coverage、training curves 均已在 `runs/figures/`）。
4. 可选扩展：CIFAR-10-C 鲁棒性评估（`src/evaluate_cifar10c.py` 已实现）。

## 证据文件清单

- 训练：`runs/{cifar10,cifar100}-baseline/`（Standard 20 ep）、`runs/{cifar10,cifar100}-energy-ft{,2,3,4}/`（微调扫描）、`runs/{cifar10,cifar100}-energy/`（scratch 试点）
- 评估：`runs/*-{msp,energy}/`（metrics.json、scores.npz、risk_coverage.csv）×24 组
- 调参：`runs/tuning_w002.csv`（w=0.02 从零试验）、微调扫描 val 结果见 history.json
- 汇总：`runs/comparison_tables.md`（主表+试点副表）、`runs/results_summary.csv`、`runs/figures/*.png`（training_curves 含微调接续段、finetune_tradeoff 权衡曲线、score_distributions 8 面板、risk_coverage）
- 数据：`runs/DATA_PREPARATION.md`、`runs/data_preparation.json`、`runs/environment_info.txt`、`runs/smoke_check.json`（24/24 通过）

